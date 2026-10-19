"""
metrics.py
──────────
Evaluation metrics for both the flow model and a deterministic baseline.

Three metrics that expose the difference between the models:

  clean_accuracy   : standard exact-match accuracy on unambiguous inputs
                     (both models should be similar here — control condition)

  coverage@K       : on ambiguous inputs, fraction of samples where BOTH
                     valid orderings appear in K draws
                     → GS is always 0% (deterministic)
                     → Flow should find both modes

  calibration_error: |empirical_freq(mode_a) - alpha|
                     does the model's sampling frequency match the
                     actual blend weight?
                     → perfect calibration = 0.0
"""

import torch
import numpy as np
from config import CFG

try:
    from torch_linear_assignment import batch_linear_assignment
    HAS_HUNGARIAN = True
except ImportError:
    HAS_HUNGARIAN = False
    print("[metrics.py] torch_linear_assignment not found — using scipy linear_sum_assignment (exact, slower)")


# ══════════════════════════════════════════════════════════════════
# 1.  ROUNDING: soft DS matrix → hard permutation
# ══════════════════════════════════════════════════════════════════

def hungarian_round(X: torch.Tensor) -> torch.Tensor:
    """
    Round a real matrix to the nearest permutation matrix in Frobenius norm,
    i.e. solve the linear assignment problem max <X, P> (Hungarian).

    No clipping: X lives on the affine set {X1 = 1, Xᵀ1 = 1} and may have
    negative entries; the assignment is computed on X itself.

    Args:
        X: [B, N, N]  real matrix
    Returns:
        P: [B, N, N]  hard permutation matrix (binary)
    """
    B, n, _ = X.shape
    X_safe  = X.nan_to_num(0.)

    asgn = None
    if HAS_HUNGARIAN:
        try:
            asgn = batch_linear_assignment(-X_safe)     # maximise sum
        except Exception:                               # kernel not built for this GPU
            asgn = None
    if asgn is None:
        from scipy.optimize import linear_sum_assignment
        Xn = X_safe.detach().float().cpu().numpy()
        asgn = torch.stack([torch.as_tensor(linear_sum_assignment(-Xn[b])[1])
                            for b in range(B)]).to(X.device)

    P    = torch.zeros_like(X)
    rows = torch.arange(n, device=X.device).unsqueeze(0).expand(B, -1)
    P[torch.arange(B, device=X.device).unsqueeze(1), rows, asgn.long()] = 1.
    return P


def is_correct_perm(P_pred: torch.Tensor,
                    P_star: torch.Tensor) -> torch.Tensor:
    """
    Element-wise exact-match check.
    Returns [B] bool tensor: True where P_pred == P_star exactly.
    """
    return (P_pred - P_star).abs().sum(dim=(-2, -1)) < 0.5


# ══════════════════════════════════════════════════════════════════
# 2.  CLEAN ACCURACY  (standard single-sample evaluation)
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def clean_accuracy(model,
                   cln_loader,
                   device: torch.device,
                   k_samples: int = 1) -> float:
    """
    Exact-match accuracy on clean (unambiguous) sequences.

    For the flow model: draws k_samples=1 (one ODE integration).
    For a deterministic model: k_samples is ignored.

    Args:
        model:      FlowModel  (must have a .sample() method)
        cln_loader: DataLoader yielding (images, P_a, P_b, alpha, is_ambig)
                    with is_ambig=False for all items
        device:     target device
        k_samples:  number of ODE samples (only the first is used here)

    Returns:
        accuracy in [0, 1]
    """
    model.eval()
    n_correct = n_total = 0

    for images, P_a, P_b, alpha, is_ambig in cln_loader:
        images = images.to(device)
        P_star = P_a.to(device)     # P_a == P_b for clean sequences

        samples = model.sample(images, k_samples=1)   # list[1] of [B,N,N]
        P_hard  = hungarian_round(samples[0].float())

        n_correct += is_correct_perm(P_hard, P_star).sum().item()
        n_total   += images.shape[0]

    return n_correct / max(n_total, 1)


# ══════════════════════════════════════════════════════════════════
# 3.  AMBIGUOUS METRICS  (coverage + calibration)
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def ambiguous_metrics(model,
                      amb_loader,
                      device: torch.device,
                      k_samples: int = CFG.K_SAMPLES) -> dict:
    """
    Evaluate on ambiguous sequences only (is_ambig=True).

    Metrics:
    ─────────
    coverage@K:
        For each ambiguous input, draw K samples and round each to a
        hard permutation.  Coverage = 1 if BOTH P*_a and P*_b appear
        in the K samples.  Report mean coverage over the test set.

        GS (deterministic): always 0.0 — by definition it can only
        output one permutation regardless of K.

        Flow: should approach 1.0 as K → ∞ and training improves.

    calibration_error:
        alpha = blend weight.  If the flow is well-calibrated, it should
        produce P*_a approximately alpha·K times out of K.
        Error = |freq(mode_a in K samples) - alpha|
        Perfect calibration = 0.0.

    any_correct@K:
        Fraction of ambiguous inputs where at least one of the K samples
        matches either P*_a or P*_b.  Measures raw recall.

    Args:
        model:      FlowModel
        amb_loader: DataLoader with 100% ambiguous sequences
        device:     target device
        k_samples:  K — number of stochastic draws per input

    Returns:
        dict with keys: coverage, calibration, any_correct
    """
    model.eval()
    coverages     = []
    calibrations  = []
    any_correct_l = []

    for images, P_a, P_b, alpha, is_ambig in amb_loader:
        images = images.to(device)
        P_a    = P_a.to(device)
        P_b    = P_b.to(device)
        B      = images.shape[0]

        # Draw K samples and round each
        all_preds = []
        for P_soft in model.sample(images, k_samples=k_samples):
            all_preds.append(hungarian_round(P_soft.float()))   # [B, N, N]

        # Per-sample analysis
        for b in range(B):
            Pa_b = P_a[b]                   # [N, N]
            Pb_b = P_b[b]                   # [N, N]
            al   = alpha[b].item()

            hit_a = hit_b = 0
            for pred in all_preds:
                p = pred[b]                 # [N, N]
                if is_correct_perm(p.unsqueeze(0),
                                   Pa_b.unsqueeze(0)).item():
                    hit_a += 1
                if is_correct_perm(p.unsqueeze(0),
                                   Pb_b.unsqueeze(0)).item():
                    hit_b += 1

            # Coverage: both modes found?
            coverages.append(float(hit_a > 0 and hit_b > 0))

            # Calibration: empirical freq(mode_a) vs alpha
            freq_a = hit_a / k_samples
            calibrations.append(abs(freq_a - al))

            # Any correct
            any_correct_l.append(float(hit_a + hit_b > 0))

    return {
        "coverage"   : float(np.mean(coverages)),
        "calibration": float(np.mean(calibrations)),
        "any_correct": float(np.mean(any_correct_l)),
        "k_samples"  : k_samples,
    }


# ══════════════════════════════════════════════════════════════════
# 4.  COMBINED EVALUATION
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def full_eval(model,
              cln_loader,
              amb_loader,
              device: torch.device,
              k_samples: int = CFG.K_SAMPLES) -> dict:
    """
    Run both clean accuracy and ambiguous metrics.

    Returns:
        dict with keys: clean_acc, coverage, calibration, any_correct
    """
    cln_acc = clean_accuracy(model, cln_loader, device)
    amb     = ambiguous_metrics(model, amb_loader, device, k_samples)

    return {
        "clean_acc"  : cln_acc,
        **amb,
    }
