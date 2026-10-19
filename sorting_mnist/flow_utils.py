"""
flow_utils.py
─────────────
Tangent space operations and OT coupling — the mathematical core
of the OT-CFM approach for permutation matrices.

Key concepts
────────────
The doubly-stochastic polytope DS_n = {M ∈ R^{n×n} : M1=1, M^T1=1, M≥0}
has a tangent space at every interior point:

    T = {V ∈ R^{n×n} : V1=0, V^T1=0}

i.e. matrices with zero row-sums AND zero column-sums.

Any noise or velocity we add to a point in DS_n must live in T,
otherwise we immediately leave the doubly-stochastic constraint set.

OT Coupling  (the key trick for bimodal targets)
───────────────────────────────────────────────
Given a noisy start X0_noisy = X0 + σ·ε and two candidate targets
P*_a, P*_b, we couple X0_noisy to whichever target it is CLOSER to
in Frobenius norm.  This partitions the noise sphere around X0 into
two regions, one per target.  The learned velocity field is then
consistent in each region — no gradient cancellation.
"""

import torch


# ══════════════════════════════════════════════════════════════════
# 1.  DOUBLY-STOCHASTIC PROJECTIONS
# ══════════════════════════════════════════════════════════════════

def tangent_project(U: torch.Tensor) -> torch.Tensor:
    """
    Project U ∈ R^{B×n×n} onto the tangent space of DS_n:
        zero row-sums AND zero column-sums.

    Formula (Gram-Schmidt on the constraint space):
        T(U) = U - row_mean - col_mean + global_mean

    This is idempotent: T(T(U)) = T(U).
    """
    row_mean    = U.mean(dim=-1, keepdim=True)       # [B, n, 1]
    col_mean    = U.mean(dim=-2, keepdim=True)       # [B, 1, n]
    global_mean = U.mean(dim=(-2, -1), keepdim=True) # [B, 1, 1]
    return U - row_mean - col_mean + global_mean


# ══════════════════════════════════════════════════════════════════

def sample_tangent_noise(B: int, n: int,
                         device: torch.device,
                         dtype: torch.dtype) -> torch.Tensor:
    """
    Sample a uniformly random unit vector in the tangent space T of DS_n.

    Steps:
      1. Sample a random Gaussian matrix ε ∈ R^{n×n}
      2. Project onto T via tangent_project (enforces row/col sum = 0)
      3. Normalise to unit Frobenius norm

    The resulting ε has ||ε||_F = 1 and lies exactly in T, so
    X0_noisy = X0 + σ·ε satisfies the DS constraints up to σ.

    Args:
        B:      batch size
        n:      sequence length
        device, dtype: match the model tensors

    Returns:
        ε: [B, n, n]  unit tangent vector for each item in batch
    """
    eps   = torch.randn(B, n, n, device=device, dtype=dtype)
    eps   = tangent_project(eps)
    norms = eps.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    return eps / norms


# ══════════════════════════════════════════════════════════════════
# 3.  NOISY STARTING POINT
# ══════════════════════════════════════════════════════════════════

def make_noisy_X0(B: int, n: int, sigma: float,
                  device: torch.device,
                  dtype: torch.dtype) -> tuple:
    """
    Create the noisy starting point for one training step.

    Returns:
        X0:       [B, n, n]  — clean uniform DS matrix (1/n everywhere)
        X0_noisy: [B, n, n]  — X0 + σ·ε, Sinkhorn-projected back to DS_n
        eps:      [B, n, n]  — the raw tangent noise (used for OT coupling)
    """
    X0       = torch.full((B, n, n), 1. / n, device=device, dtype=dtype)
    eps      = sample_tangent_noise(B, n, device, dtype)
    # eps is already in the DS tangent space, so X0 + σ·eps stays
    # affine-DS (row/col sums = 1) without re-projection. Re-projecting
    # the whole sum would annihilate the centroid: tangent_project(X0)=0.
    X0_noisy = X0 + sigma * eps
    return X0, X0_noisy, eps


# ══════════════════════════════════════════════════════════════════
# 4.  OPTIMAL TRANSPORT COUPLING
# ══════════════════════════════════════════════════════════════════

def ot_couple(X0_noisy: torch.Tensor,
              P_star_a: torch.Tensor,
              P_star_b: torch.Tensor) -> torch.Tensor:
    """
    OT coupling: for each sample, pick the target that X0_noisy is
    closer to in Frobenius norm.

    WHY THIS WORKS
    ──────────────
    The noise ε is a random unit vector in T.  Approximately half of
    all noise realisations will point toward the hemisphere containing
    P*_a, and half toward P*_b.

    By coupling each noisy start to its nearest target, we guarantee:
      - Every training step has a SINGLE, CONSISTENT velocity target
      - The velocity field learns to bifurcate: "if I start near P*_a,
        go to P*_a; if I start near P*_b, go to P*_b"
      - At inference: sample different ε → land on different modes

    Without OT coupling (random target selection), the two gradient
    directions cancel in expectation → the model learns v* ≈ 0.

    Args:
        X0_noisy: [B, n, n]
        P_star_a: [B, n, n]
        P_star_b: [B, n, n]

    Returns:
        P_chosen: [B, n, n]  — per-sample target, either P*_a or P*_b
        mask_a:   [B]  bool  — True where P*_a was chosen
    """
    # Frobenius distance squared to each target
    d_a = (X0_noisy - P_star_a).pow(2).sum(dim=(-2, -1))  # [B]
    d_b = (X0_noisy - P_star_b).pow(2).sum(dim=(-2, -1))  # [B]

    mask_a   = d_a <= d_b                                  # [B] bool
    mask_a_f = mask_a.float().view(-1, 1, 1)               # [B, 1, 1]

    P_chosen = mask_a_f * P_star_a + (1. - mask_a_f) * P_star_b
    return P_chosen, mask_a


def _sphere_coord_quantile(p: torch.Tensor, m: int) -> torch.Tensor:
    """
    q with Pr[u >= q] = p, where u = <ε, d> for ε uniform on the unit
    sphere of an m-dimensional space and d a fixed unit vector.
    (u+1)/2 ~ Beta((m-1)/2, (m-1)/2).
    """
    from scipy.special import betaincinv
    a = 0.5 * (m - 1)
    q = 2.0 * betaincinv(a, a, (1.0 - p).detach().double().cpu().numpy()) - 1.0
    return torch.as_tensor(q, device=p.device)


def couple(X0_noisy: torch.Tensor,
           eps: torch.Tensor,
           P_star_a: torch.Tensor,
           P_star_b: torch.Tensor,
           alpha: torch.Tensor = None,
           mode: str = "nearest") -> torch.Tensor:
    """
    Coupling of the noisy start to one of the two valid targets.

    nearest    : argmin_P ||X̃₀ − P||_F.  Since <X₀, P_a − P_b> = 0 this is
                 the half-space <ε, P_a − P_b> >= 0, of mass 1/2 for
                 isotropic ε: the sampler is unbiased over modes but cannot
                 follow α.
    calibrated : a power-diagram coupling, the same half-space shifted to
                 <ε, d> >= q(α), d = (P_a − P_b)/||P_a − P_b||, with q chosen
                 so that Pr[choose a] = α exactly.  Cells remain convex and
                 still contain their target's direction of travel.
    random     : ablation — independent of ε, a with prob. 1/2.
    """
    if mode == "nearest":
        P, _ = ot_couple(X0_noisy, P_star_a, P_star_b)
        return P
    B, n = P_star_a.shape[0], P_star_a.shape[-1]
    if mode == "random":
        pick_a = torch.rand(B, device=P_star_a.device) < 0.5
    elif mode == "calibrated":
        D     = (P_star_a - P_star_b).float()
        Dn    = D.norm(dim=(-2, -1))
        u     = (eps.float() * D).sum(dim=(-2, -1)) / Dn.clamp(min=1e-8)
        q     = _sphere_coord_quantile(alpha.float().clamp(1e-6, 1 - 1e-6),
                                       (n - 1) ** 2).float()
        pick_a = (u >= q) | (Dn < 1e-6)         # clean: P_a = P_b, either
    else:
        raise ValueError(mode)
    m = pick_a.to(P_star_a.dtype).view(-1, 1, 1)
    return m * P_star_a + (1. - m) * P_star_b


# ══════════════════════════════════════════════════════════════════
# 5.  CFM LOSS
# ══════════════════════════════════════════════════════════════════

def cfm_loss(v_pred: torch.Tensor,
             X0: torch.Tensor,
             X0_noisy: torch.Tensor,
             P_chosen: torch.Tensor,
             t: torch.Tensor) -> torch.Tensor:
    """
    Conditional Flow Matching loss.

    Straight-line path:
        X_t   = (1-t)·X0_noisy + t·P_chosen
        v*    = P_chosen - X0              (constant along path)
        loss  = ||v_pred - v*||²_F

    Note: v* uses un-noised X0 as base so the velocity magnitude is
    consistent across different noise realisations.

    Args:
        v_pred:   [B, n, n]  — velocity predicted by network (tangent-projected)
        X0:       [B, n, n]  — clean uniform start
        X0_noisy: [B, n, n]  — noisy start (used only to define the path)
        P_chosen: [B, n, n]  — OT-coupled target
        t:        [B]        — sampled time in [0, 1]

    Returns:
        scalar loss
    """
    v_star = P_chosen - X0       # [B, n, n]  target velocity
    return ((v_pred - v_star) ** 2).mean()


# ══════════════════════════════════════════════════════════════════
# 6.  FULL TRAINING STEP HELPER
# ══════════════════════════════════════════════════════════════════

def compute_cfm_loss(velocity_net_out: torch.Tensor,
                     P_star_a: torch.Tensor,
                     P_star_b: torch.Tensor,
                     sigma: float,
                     n: int) -> torch.Tensor:
    """
    Convenience wrapper that runs the full CFM pipeline:
      make_noisy_X0 → ot_couple → sample t → cfm_loss

    Args:
        velocity_net_out: [B, n, n]  — raw output of velocity network
                                        (tangent_project applied internally)
        P_star_a:         [B, n, n]
        P_star_b:         [B, n, n]
        sigma:            noise magnitude
        n:                sequence length

    Returns:
        scalar CFM loss
    """
    B      = P_star_a.shape[0]
    device = P_star_a.device
    dtype  = P_star_a.dtype

    # 1. Noisy start
    X0, X0_noisy, _ = make_noisy_X0(B, n, sigma, device, dtype)

    # 2. OT coupling
    P_chosen, _ = ot_couple(X0_noisy, P_star_a, P_star_b)

    # 3. Sample t (not used in loss formula directly but kept for consistency)
    t = torch.rand(B, device=device, dtype=dtype)

    # 4. Project velocity prediction to tangent space
    v_pred = tangent_project(velocity_net_out)

    # 5. Loss
    return cfm_loss(v_pred, X0, X0_noisy, P_chosen, t)
