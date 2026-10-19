"""
metrics.py  —  evaluation under an ambiguous target set
═══════════════════════════════════════════════════════

WHY THEIR ACCURACY AND KENDALL-τ BREAK ON THE MAIN ARM
──────────────────────────────────────────────────────
Both are scored against the single *sampled* ground-truth permutation τ.
When |𝒪| > 1, a perfect reconstruction drawn from a different orbit member
counts as wrong: the model restores the image byte-for-byte and is still
marked incorrect.  That is not an inconvenience to work around — it is the
reason the σ = 0.01 noise was added to the benchmark in the first place, and
it is the argument for the framing.  We report them anyway (flagged), because
they are meaningful on the σ > 0 arm.

Their RMSE and MAE remain valid on every arm and are kept unchanged for
comparability with their Table 1: every orbit member yields a byte-identical
reconstruction, so RMSE = MAE = 0 for all of them.

WHAT WE REPORT INSTEAD
──────────────────────
  validity_rate         fraction of the K samples landing in 𝒪.
                        Replaces accuracy.  (At K = 1 this is exactly their
                        "pixel-wise accuracy", which is already orbit-blind.)

  distinct_modes@K      number of distinct orbit members hit among K samples,
                        normalised by min(K, |𝒪|).  1.0 = the sampler covered
                        as much of the orbit as K draws possibly could.

  marginal_tv           for each ambiguous patch, the total-variation distance
                        between its empirical distribution over admissible
                        slots and uniform.  This is the generalisation of the
                        two-mode calibration error |f̂_a − α| to large orbits,
                        and it stays computable at any |𝒪|.

  rmse / mae            unchanged, for comparability with their Table 1.

CONVENTIONS
───────────
    pred_slot[i] = j   observed patch i placed at original slot j
    perm_list[j] = i   their `gt_perm_list` ordering = argsort(pred_slot)
"""

from __future__ import annotations

import numpy as np
import torch

from config import CFG
from orbit import block_multiplicities, compat_mask, is_in_orbit

try:
    from scipy.stats import kendalltau
    HAS_SCIPY_STATS = True
except ImportError:                                     # pragma: no cover
    HAS_SCIPY_STATS = False

from orbit import _hungarian_max


# ══════════════════════════════════════════════════════════════════
# 1.  ROUNDING
# ══════════════════════════════════════════════════════════════════

def hungarian_round(X: torch.Tensor) -> torch.Tensor:
    """Soft [B,n,n] → hard permutation [B,n,n] maximising ⟨X, P⟩."""
    assign = _hungarian_max(X.float())
    P = torch.zeros_like(X, dtype=torch.float32)
    P.scatter_(2, assign.unsqueeze(-1), 1.0)
    return P


def to_pred_slot(X: torch.Tensor) -> torch.Tensor:
    """Soft [B,n,n] → pred_slot [B,n] int64 via Hungarian rounding."""
    return _hungarian_max(X.float())


# ══════════════════════════════════════════════════════════════════
# 2.  RECONSTRUCTION  (their RMSE / MAE / pixel-wise accuracy)
# ══════════════════════════════════════════════════════════════════

def reconstruct(patches: torch.Tensor, pred_slot: torch.Tensor) -> torch.Tensor:
    """
    Place observed patch i at slot pred_slot[i].

    Args:
        patches:   [B, n, C, p, p]  observed (shuffled) patches
        pred_slot: [B, n]
    Returns:
        [B, n, C, p, p] in slot order
    """
    B, n = patches.shape[:2]
    out  = torch.empty_like(patches)
    idx  = pred_slot.view(B, n, 1, 1, 1).expand_as(patches)
    out.scatter_(1, idx, patches)
    return out


def recon_errors(patches: torch.Tensor,
                 pred_slot: torch.Tensor,
                 gt_slot:   torch.Tensor) -> tuple:
    """
    Per-image RMSE and MAE of the reconstruction against the original image.

    The reference is the *actual* original arrangement of these exact patches
    (reconstruct with gt_slot), so any member of 𝒪 gives error exactly 0 —
    which is the property that makes these two metrics valid on every arm.

    Returns: (rmse [B], mae [B])
    """
    pred = reconstruct(patches, pred_slot)
    ref  = reconstruct(patches, gt_slot)
    diff = (pred - ref).flatten(1)
    return diff.pow(2).mean(-1).sqrt(), diff.abs().mean(-1)


# ══════════════════════════════════════════════════════════════════
# 3.  THEIR PERMUTATION METRICS  (valid only when |𝒪| = 1)
# ══════════════════════════════════════════════════════════════════

def perm_metrics(pred_slot: torch.Tensor,
                 gt_slot:   torch.Tensor,
                 obs_blk:   torch.Tensor = None,
                 slot_blk:  torch.Tensor = None) -> dict:
    """
    Their Accuracy, Correct and Kendall-τ, against the single sampled ground
    truth, in their `gt_perm_list` representation.

    Accuracy — fraction of images reassembled perfectly.
    Correct  — every piece judged on its own: is it back at its ground-truth
               slot?  Reported as correct pieces / total pieces.  This is
               their `permutation_prop_correct_pieces`, and like Accuracy it
               is orbit-sensitive: a background piece swapped with an
               identical background piece is counted wrong even though the
               image is byte-perfect.
    Correct(content) — the same per-piece count, but a piece is judged correct
               when the *content* it carries matches the content originally at
               that slot.  This is their `pixel_wise_prop_correct_pieces`, it
               is orbit-invariant, and it is the number to quote on the main
               arm.  On σ > 0 data the two coincide exactly.
    """
    exact = pred_slot.eq(gt_slot).all(-1).float()
    piece = pred_slot.eq(gt_slot).float().mean(-1)

    if obs_blk is not None:
        landed  = torch.gather(slot_blk, 1, pred_slot)          # [B, n]
        content = landed.eq(obs_blk).float().mean(-1)           # [B]
    else:
        content = piece

    taus = []
    if HAS_SCIPY_STATS:
        p_list = torch.argsort(pred_slot, dim=-1).cpu().numpy()
        g_list = torch.argsort(gt_slot,   dim=-1).cpu().numpy()
        for a, b in zip(g_list, p_list):
            t = kendalltau(a, b)[0]
            taus.append(0.0 if np.isnan(t) else float(t))

    return dict(exact=exact.cpu().numpy(),          # Accuracy  (all-or-nothing)
                piece=piece.cpu().numpy(),          # Correct   (per-patch)
                content=content.cpu().numpy(),      # Correct   (content-based)
                tau=np.asarray(taus) if taus else np.zeros(len(exact)))


# ══════════════════════════════════════════════════════════════════
# 4.  ORBIT METRICS
# ══════════════════════════════════════════════════════════════════

def _orbit_size_capped(log_orbit: torch.Tensor, cap: float = 1e12):
    return torch.exp(log_orbit.clamp(max=np.log(cap)))


# ── finite-K floor of the marginal-TV metric ──────────────────────
#
# Even a *perfectly* uniform sampler has TV > 0 at finite K, because an
# empirical histogram of K draws cannot equal 1/m exactly.  Reporting TV
# without this floor is meaningless: at K = 5 over m = 7 slots the best
# achievable value is already ≈ 0.44.  We estimate the floor by Monte Carlo
# per (m, K) pair, cached, and report it alongside the measured TV so the two
# can be read against each other.
_TV_FLOOR_CACHE: dict = {}


def tv_floor(m: int, K: int, n_sim: int = 4000, seed: int = 0) -> float:
    """E[TV] for K i.i.d. draws from Uniform{1..m} against Uniform{1..m}."""
    key = (int(m), int(K))
    if key not in _TV_FLOOR_CACHE:
        rng    = np.random.default_rng(seed + 1000 * key[0] + key[1])
        counts = rng.multinomial(K, [1.0 / m] * m, size=n_sim) / K
        _TV_FLOOR_CACHE[key] = float(
            0.5 * np.abs(counts - 1.0 / m).sum(-1).mean())
    return _TV_FLOOR_CACHE[key]


def orbit_metrics(all_pred_slot: list,
                  obs_blk:   torch.Tensor,
                  slot_blk:  torch.Tensor,
                  log_orbit: torch.Tensor) -> dict:
    """
    The K-sample distributional metrics.

    Args:
        all_pred_slot: list of K tensors, each [B, n]
        obs_blk, slot_blk: [B, n]
        log_orbit: [B]  log|𝒪| in nats
    Returns:
        dict of per-image numpy arrays:
            validity   [B]  fraction of the K samples in 𝒪
            distinct   [B]  distinct orbit members hit / min(K, |𝒪|)
            n_distinct [B]  raw count of distinct valid members
            tv         [B]  mean per-patch marginal TV over ambiguous patches
                            (nan for images with no ambiguous patch)
            ambiguous  [B]  bool, |𝒪| > 1
    """
    K = len(all_pred_slot)
    B, n = all_pred_slot[0].shape
    device = obs_blk.device

    valid = torch.stack([is_in_orbit(ps, obs_blk, slot_blk)
                         for ps in all_pred_slot], dim=0)          # [K, B]
    stack = torch.stack(all_pred_slot, dim=0)                       # [K, B, n]

    validity = valid.float().mean(0)                                # [B]

    # ── distinct valid orbit members among the K draws ───────────
    orb_cap = _orbit_size_capped(log_orbit.to(torch.float64))
    n_distinct = np.zeros(B, dtype=np.float64)
    stack_cpu  = stack.cpu().numpy()
    valid_cpu  = valid.cpu().numpy()
    for b in range(B):
        seen = {tuple(stack_cpu[k, b]) for k in range(K) if valid_cpu[k, b]}
        n_distinct[b] = len(seen)
    denom    = np.minimum(K, orb_cap.cpu().numpy())
    distinct = n_distinct / np.maximum(denom, 1.0)

    # ── per-patch marginal TV ────────────────────────────────────
    #
    # For an ambiguous patch i (class size m ≥ 2) the K draws induce an
    # empirical distribution over the m admissible slots plus an "inadmissible"
    # catch-all.  The target is uniform on the admissible slots, so
    #
    #   TV = ½ ( Σ_{s admissible} |p_s − 1/m|  +  p_inadmissible )
    #
    # which reduces to the usual TV when every draw is valid, and correctly
    # charges a method that leaves 𝒪 altogether.  With two modes and K draws
    # this is exactly |f̂_a − α| at α = ½.
    compat = compat_mask(obs_blk, slot_blk)                         # [B,n,n]
    mult   = block_multiplicities(obs_blk).to(torch.float32)        # [B,n]

    counts = torch.zeros(B, n, n, device=device)
    onehot = torch.zeros(B, n, n, device=device)
    for k in range(K):
        onehot.zero_()
        onehot.scatter_(2, all_pred_slot[k].unsqueeze(-1), 1.0)
        counts += onehot
    p = counts / K                                                  # [B,n,n]

    p_adm   = p * compat.float()
    p_inadm = 1.0 - p_adm.sum(-1)                                   # [B,n]
    unif    = compat.float() / mult.unsqueeze(-1).clamp(min=1.0)
    tv_i    = 0.5 * ((p_adm - unif).abs() * compat.float()).sum(-1) \
              + 0.5 * p_inadm                                       # [B,n]

    amb_patch = (mult > 1)                                          # [B,n]
    n_amb     = amb_patch.sum(-1)
    tv = torch.where(n_amb > 0,
                     (tv_i * amb_patch.float()).sum(-1) / n_amb.clamp(min=1),
                     torch.full_like(tv_i[:, 0], float("nan")))

    # Floor: what an exactly-uniform sampler would score at this K, on this
    # image's own multiplicities.  tv ≈ tv_floor means "uniform to within the
    # resolution K affords".
    mult_np  = mult.cpu().numpy().astype(int)
    amb_np   = amb_patch.cpu().numpy()
    floor    = np.full(B, np.nan)
    for b in range(B):
        ms = mult_np[b][amb_np[b]]
        if ms.size:
            floor[b] = float(np.mean([tv_floor(int(m), K) for m in ms]))

    return dict(
        validity   = validity.cpu().numpy(),
        distinct   = distinct,
        n_distinct = n_distinct,
        tv         = tv.cpu().numpy(),
        tv_floor   = floor,
        ambiguous  = (log_orbit > 0).cpu().numpy(),
    )


# ══════════════════════════════════════════════════════════════════
# 5.  FULL EVALUATION
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, device: torch.device,
             k_samples: int = CFG.K_SAMPLES,
             single_sample_only: bool = False) -> dict:
    """
    Run `model.sample()` over `loader` and aggregate every metric.

    Args:
        single_sample_only: draw K = 1 and skip the distributional metrics
                            (used for the cheap during-training readout and
                            for the full-test-split accuracy/RMSE table).
    """
    if hasattr(model, "eval"):
        model.eval()
    K = 1 if single_sample_only else k_samples

    acc, agg = {}, dict(exact=[], piece=[], content=[], tau=[], rmse=[],
                        mae=[], validity=[], distinct=[], n_distinct=[],
                        tv=[], tv_floor=[], ambiguous=[], log_orbit=[])

    for patches, gt_slot, obs_blk, slot_blk, log_orbit in loader:
        patches  = patches.to(device, non_blocking=True)
        gt_slot  = gt_slot.to(device,  non_blocking=True)
        obs_blk  = obs_blk.to(device,  non_blocking=True)
        slot_blk = slot_blk.to(device, non_blocking=True)
        log_orbit = log_orbit.to(device, non_blocking=True)

        soft = model.sample(patches, k_samples=K)
        preds = [to_pred_slot(s) for s in soft]

        # single-sample metrics use the first draw
        pm = perm_metrics(preds[0], gt_slot, obs_blk, slot_blk)
        rmse, mae = recon_errors(patches, preds[0], gt_slot)
        agg["exact"].append(pm["exact"])
        agg["piece"].append(pm["piece"])
        agg["content"].append(pm["content"])
        agg["tau"].append(pm["tau"])
        agg["rmse"].append(rmse.cpu().numpy())
        agg["mae"].append(mae.cpu().numpy())
        agg["log_orbit"].append(log_orbit.cpu().numpy())

        om = orbit_metrics(preds, obs_blk, slot_blk, log_orbit)
        for key in ("validity", "distinct", "n_distinct", "tv", "tv_floor",
                    "ambiguous"):
            agg[key].append(om[key])

    cat = {k: np.concatenate(v) for k, v in agg.items() if len(v)}
    amb = cat["ambiguous"].astype(bool)

    def _mean(x, mask=None):
        v = x if mask is None else x[mask]
        v = v[~np.isnan(v)]
        return float(v.mean()) if v.size else float("nan")

    acc = dict(
        k_samples     = K,
        n_images      = int(cat["exact"].size),
        frac_ambig    = float(amb.mean()),
        mean_log_orbit= _mean(cat["log_orbit"]),
        # their metrics (valid only where |𝒪| = 1)
        accuracy      = _mean(cat["exact"]),      # Accuracy (%)  all-or-nothing
        correct       = _mean(cat["piece"]),      # Correct  (%)  per-patch
        correct_content = _mean(cat["content"]),  # Correct  (%)  orbit-invariant
        piece_acc     = _mean(cat["piece"]),      # back-compat alias
        kendall_tau   = _mean(cat["tau"]),
        rmse          = _mean(cat["rmse"]),
        mae           = _mean(cat["mae"]),
        # orbit metrics
        validity      = _mean(cat["validity"]),
        validity_amb  = _mean(cat["validity"], amb),
        distinct      = _mean(cat["distinct"], amb),
        n_distinct    = _mean(cat["n_distinct"], amb),
        marginal_tv   = _mean(cat["tv"], amb),
        marginal_tv_floor = _mean(cat["tv_floor"], amb),
        # their accuracy restricted to the unambiguous images, where it is
        # a fair number even on the main arm
        accuracy_ident= _mean(cat["exact"], ~amb) if (~amb).any() else float("nan"),
    )
    return acc


def _pct(x, width=6):
    """Percent, or 'n/a' when the metric has no images to average over."""
    return f"{'n/a':>{width}}   " if np.isnan(x) else f"{x*100:>{width}.2f}%  "


def _num(x, fmt=".4f"):
    return "n/a" if np.isnan(x) else format(x, fmt)


def format_results(r: dict, title: str = "") -> str:
    """Pretty block for logs."""
    amb_note = ("   [UNDEFINED on ambiguous images]"
                if r["frac_ambig"] > 0.01 else "")
    lines = []
    if title:
        lines += ["=" * 66, title, "=" * 66]
    lines += [
        f"  images / K                 : {r['n_images']:,} / {r['k_samples']}",
        f"  ambiguous images (|O|>1)   : {r['frac_ambig']*100:.1f}%"
        f"   mean log|O| = {r['mean_log_orbit']:.3f}",
        "",
        "  — orbit-aware (valid on every arm) —",
        f"  Validity rate              : {_pct(r['validity'])}",
        f"    on ambiguous images      : {_pct(r['validity_amb'])}",
        f"  Distinct valid modes @K    : {_pct(r['distinct'])}"
        f" (raw {_num(r['n_distinct'], '.2f')} members)",
        f"  Per-patch marginal TV      : {_num(r['marginal_tv'])}"
        f"   (uniform-sampler floor at this K: "
        f"{_num(r['marginal_tv_floor'])})",
        f"  Correct % (content-based)  : {_pct(r['correct_content'])}",
        f"  RMSE / MAE                 : {_num(r['rmse'], '.5f')} / "
        f"{_num(r['mae'], '.5f')}",
        "",
        "  — their metrics (scored against the one sampled τ) —",
        f"  Accuracy %  (whole image)  : {_pct(r['accuracy'])}" + amb_note,
        f"    on |O| = 1 images only   : {_pct(r['accuracy_ident'])}",
        f"  Correct %   (per patch)    : {_pct(r['correct'])}" + amb_note,
        f"  Kendall-τ                  : {_num(r['kendall_tau'])}" + amb_note,
    ]
    return "\n".join(lines)
