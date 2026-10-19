"""
flow_utils.py  —  tangent-space geometry, orbit coupling, CFM loss
══════════════════════════════════════════════════════════════════

The geometry is identical to the sorting task.  The flow lives on the
AFFINE SUBSPACE of matrices with unit row and column sums,

    A_n = { M ∈ R^{n×n} : M1 = 1, Mᵀ1 = 1 },

NOT on the Birkhoff polytope: there is no non-negativity constraint anywhere
in the method, and X_t routinely has negative entries.  A_n is a flat affine
space, which is exactly why straight-line conditional paths stay feasible and
no projection step is ever needed during integration.  Its direction space is
the linear subspace

    T = { V : V1 = 0, Vᵀ1 = 0 },   T(U) = C U C,  C = I − (1/n)11ᵀ

which `tangent_project` implements as U − rowmean − colmean + globalmean.
Adding any V ∈ T to a point of A_n keeps it in A_n, so both the noise and the
velocity are projected with C and the iterate never leaves A_n.  The single
map back to the permutations is the Hungarian rounding at the very end.

The ONE new piece is the coupling.  In the two earlier experiments the target
set had two elements and the coupling was "pick the nearer of P*_a, P*_b".
Here the target set is the orbit 𝒪 of the exact-content partition, whose size
is ∏_b m_b!, and the coupling is

    P* = argmin_{P ∈ 𝒪} ‖X̃₀ − P‖_F

which decomposes over the partition (see orbit.py): identified patches are
fixed and each ambiguous class contributes an independent m_b × m_b linear
assignment on the corresponding submatrix of X̃₀.  Cost O(Σ_b m_b³) —
independent of |𝒪|, so a 10⁸-element orbit costs no more than a 2-element one.

ε₀ is resampled and the coupling recomputed EVERY batch.  Caching P* would
freeze one orbit member per image and collapse the method to single-label
supervision — exactly the failure mode the experiment is about.
"""

import torch

from orbit import nearest_orbit_member, nearest_orbit_member_blockwise


# ══════════════════════════════════════════════════════════════════
# 1.  GEOMETRY OF THE UNIT-ROW/COLUMN-SUM AFFINE SPACE
# ══════════════════════════════════════════════════════════════════

def tangent_project(U: torch.Tensor) -> torch.Tensor:
    """
    Orthogonal projection onto T = {V : V1 = 0, Vᵀ1 = 0}, i.e. C U C.
    Idempotent.  Note this is a projection onto a LINEAR SUBSPACE — it does
    not clip anything to be non-negative, and nothing downstream does either.
    """
    row  = U.mean(dim=-1, keepdim=True)
    col  = U.mean(dim=-2, keepdim=True)
    glob = U.mean(dim=(-2, -1), keepdim=True)
    return U - row - col + glob


# ══════════════════════════════════════════════════════════════════
# 2.  TANGENT-SPACE NOISE AND NOISY START
# ══════════════════════════════════════════════════════════════════

def sample_tangent_noise(B: int, n: int,
                         device: torch.device,
                         dtype: torch.dtype) -> torch.Tensor:
    """Unit vector in T — doubly centred, Frobenius norm 1.  [B, n, n]."""
    eps   = torch.randn(B, n, n, device=device, dtype=dtype)
    eps   = tangent_project(eps)
    norms = eps.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    return eps / norms


def make_noisy_X0(B: int, n: int, sigma: float,
                  device: torch.device, dtype: torch.dtype):
    """
    X₀  = (1/n)·11ᵀ            the centre of A_n
    X̃₀  = X₀ + σ₀·ε           the noisy start,  ε = CεᵣC / ‖CεᵣC‖_F

    NO projection is applied to the sum, and none is needed.  ε is centred
    once, inside `sample_tangent_noise`, so it lies in T and has row and
    column sums 0; A_n is an affine space with direction space T, so adding
    σ₀·ε to X₀ ∈ A_n lands back in A_n and the sums stay exactly 1.

    Do NOT be tempted to write this as C( X₀ + σ₀ε )C.  Since C1 = 0 we have
    C·(1/n)11ᵀ·C = 0 exactly, so that expression annihilates the centroid and
    returns σ₀·CεC, whose row and column sums are 0 rather than 1.  The same
    argument covers the whole trajectory: the velocity is tangent-projected
    too, so every Euler step adds an element of T and the iterate stays in A_n
    without a single projection after initialisation.

    Entries may be negative, and that is fine: A_n is an affine space, not the
    Birkhoff polytope, and nothing downstream requires non-negativity.

    Note ‖X̃₀ − X₀‖_F = σ₀ exactly, not merely in expectation, because ε is
    normalised to unit Frobenius norm *after* centring (centring is a
    projection, so normalising first would leave ‖ε‖_F < 1).  This is what
    makes the endpoint residual of a perfectly learned field equal σ₀ exactly.
    """
    X0       = torch.full((B, n, n), 1.0 / n, device=device, dtype=dtype)
    eps      = sample_tangent_noise(B, n, device, dtype)
    X0_noisy = X0 + sigma * eps
    return X0, X0_noisy, eps


# ══════════════════════════════════════════════════════════════════
# 3.  ORBIT COUPLING
# ══════════════════════════════════════════════════════════════════

def orbit_couple(X0_noisy: torch.Tensor,
                 obs_blk:  torch.Tensor,
                 slot_blk: torch.Tensor,
                 mode:     str = "penalised") -> torch.Tensor:
    """
    P* = argmin_{P ∈ 𝒪} ‖X̃₀ − P‖_F, per sample.

    Args:
        X0_noisy: [B, n, n]
        obs_blk, slot_blk: [B, n]  content-class ids
        mode: "penalised" (one batched n×n Hungarian, GPU) or
              "blockwise"  (the O(Σ m_b³) decomposition, CPU reference)
    Returns:
        P_star: [B, n, n]

    Every noise draw is routed to its own nearest orbit member, so the K
    disjoint basins of the K orbit members each receive a consistent velocity
    target.  Without this the ∏ m_b! opposite gradients would cancel and the
    field would learn v* ≈ 0 on the ambiguous block.
    """
    X = X0_noisy.detach().float()
    if mode == "blockwise":
        P = nearest_orbit_member_blockwise(X, obs_blk, slot_blk)
    else:
        P = nearest_orbit_member(X, obs_blk, slot_blk)
    return P.to(X0_noisy.dtype)


# ══════════════════════════════════════════════════════════════════
# 4.  CFM LOSS
# ══════════════════════════════════════════════════════════════════

def cfm_loss(v_pred: torch.Tensor,
             X0: torch.Tensor,
             P_chosen: torch.Tensor) -> torch.Tensor:
    """
    Straight-line conditional path X₀ → P*, so the target velocity is the
    constant  v* = P* − X₀  and the loss is ‖T(v_pred) − v*‖²_F.
    v_pred is expected to be tangent-projected already.
    """
    return ((v_pred - (P_chosen - X0)) ** 2).mean()
