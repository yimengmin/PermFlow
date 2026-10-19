"""
orbit.py  —  the ambiguity structure of a jigsaw instance
═════════════════════════════════════════════════════════

Two patches are *exchangeable* when their pixel content is exactly equal.
Exchangeability is an equivalence relation, so it partitions the n patches
into classes B₁,…,B_r with sizes m₁,…,m_r, and the set of permutations that
perfectly reconstruct the image is the orbit

    𝒪 = { P ∈ S_n : P sends each patch to a slot whose original content
                     equals that patch },        |𝒪| = ∏_b m_b!

We never enumerate 𝒪.  The partition is a complete, compact description of
it, and every quantity we need is a closed-form function of the partition:

    nearest orbit member   argmin_{P∈𝒪} ‖X̃₀ − P‖_F     (§4)
    orbit mean             P̄ = mean over 𝒪              (§3)
    membership test        P ∈ 𝒪 ?                       (§3)
    |𝒪|                    ∏_b m_b!                      (§2)

REPRESENTATION
──────────────
Everything is expressed with two integer vectors per instance:

    slot_blk[j] = class id of the patch that originally occupied slot j
    obs_blk[i]  = class id of the i-th observed (shuffled) patch

Because the observed patches are a permutation of the original ones,
obs_blk[i] = slot_blk[τ(i)] where τ is the ground-truth assignment, and each
class occupies exactly m_b observed indices and m_b slots.

PERMUTATION CONVENTION  (matches SymmetricDiffusers' `chunk_image`)
───────────────────────────────────────────────────────────────────
    P[i, j] = 1   ⟺   observed patch i belongs at original slot j
    gt_slot[i] = j                (row-major "where does patch i go")
    perm_list[j] = i              (their `gt_perm_list`, = argsort(gt_slot))

WHY THIS IS INFORMATION-PARITY-SAFE
───────────────────────────────────
The partition is a deterministic function of the original unshuffled image,
which every method already receives — Gumbel-Sinkhorn's pixel-MSE loss is
computed against it.  So no method is handed a label the others lack, and the
ambiguity structure is *discovered from the data* rather than hand-specified.
Grouping is by exact content (a hash of the raw byte array); there is no
distance threshold and no ε to tune.
"""

from __future__ import annotations

import math

import numpy as np
import torch

try:
    from torch_linear_assignment import batch_linear_assignment
    HAS_TLA = True
except ImportError:                                     # pragma: no cover
    HAS_TLA = False

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:                                     # pragma: no cover
    HAS_SCIPY = False


# ══════════════════════════════════════════════════════════════════
# 1.  BUILDING THE PARTITION
# ══════════════════════════════════════════════════════════════════

def partition_by_content(patches: np.ndarray) -> np.ndarray:
    """
    Group patches by *exact* pixel content.

    Args:
        patches: [n, ...] array — the n patches of one image, in slot order.

    Returns:
        slot_blk: [n] int64 — class id per slot.  Ids are assigned by
                  np.unique's lexicographic order, so they are deterministic
                  and independent of the shuffling.

    Exact equality means byte equality of the stored array: two all-zero
    background patches collide, everything else essentially never does.  No
    threshold, no tolerance parameter.
    """
    flat = np.ascontiguousarray(patches).reshape(patches.shape[0], -1)
    _, slot_blk = np.unique(flat, axis=0, return_inverse=True)
    return slot_blk.astype(np.int64).reshape(-1)


def partition_from_cost(patches_obs: torch.Tensor,
                        patches_orig: torch.Tensor,
                        atol: float = 0.0) -> tuple:
    """
    The equivalent, and for the paper more transparent, construction:
    build cost[i,j] = ‖patch_i − original_patch_j‖² and read the ambiguity off
    the *exact ties* in that cost matrix.  The set of optimal assignments of
    this Hungarian problem is exactly 𝒪.

    Args:
        patches_obs :  [n, ...] observed (shuffled) patches
        patches_orig:  [n, ...] original patches, in slot order

    Returns:
        cost   : [n, n] squared-distance cost matrix
        compat : [n, n] bool — cost[i,j] == 0, i.e. i may go to slot j
    """
    a    = patches_obs.reshape(patches_obs.shape[0], -1).double()
    b    = patches_orig.reshape(patches_orig.shape[0], -1).double()
    cost = torch.cdist(a, b).pow(2)
    return cost, (cost <= atol)


# ══════════════════════════════════════════════════════════════════
# 2.  ORBIT SIZE
# ══════════════════════════════════════════════════════════════════

def orbit_log_size(slot_blk: np.ndarray) -> float:
    """log|𝒪| = Σ_b log(m_b!).  Returned in nats; use exp() only when small."""
    _, counts = np.unique(slot_blk, return_counts=True)
    return float(sum(math.lgamma(int(m) + 1) for m in counts))


def orbit_size(slot_blk: np.ndarray, cap: float = 1e18) -> float:
    """|𝒪| = ∏_b m_b!, capped so it stays representable as a float."""
    return min(math.exp(orbit_log_size(slot_blk)), cap)


def block_multiplicities(blk: torch.Tensor) -> torch.Tensor:
    """
    Per-element class size.

    Args:
        blk: [B, n] class ids
    Returns:
        mult: [B, n] — mult[b,i] = m_{blk[b,i]}
    """
    same = blk.unsqueeze(-1).eq(blk.unsqueeze(-2))        # [B, n, n]
    return same.sum(-1)


# ══════════════════════════════════════════════════════════════════
# 3.  ORBIT MEAN, MEMBERSHIP
# ══════════════════════════════════════════════════════════════════

def compat_mask(obs_blk: torch.Tensor, slot_blk: torch.Tensor) -> torch.Tensor:
    """
    [B, n, n] bool — compat[b,i,j] = True iff observed patch i may occupy
    slot j, i.e. their contents are exactly equal.
    """
    return obs_blk.unsqueeze(-1).eq(slot_blk.unsqueeze(-2))


def orbit_mean(obs_blk: torch.Tensor, slot_blk: torch.Tensor) -> torch.Tensor:
    """
    P̄ = mean over 𝒪, available in closed form: 1 at the identified positions
    and 1/m_b inside each ambiguous block.

    This is the marginal target handed to Gumbel-Sinkhorn.  It is doubly
    stochastic by construction (each class contributes an m_b × m_b block of
    1/m_b).

    Args:
        obs_blk, slot_blk: [B, n] class ids
    Returns:
        P_bar: [B, n, n]
    """
    M = compat_mask(obs_blk, slot_blk).to(torch.float32)
    return M / M.sum(-1, keepdim=True).clamp(min=1.0)


def is_in_orbit(pred_slot: torch.Tensor,
                obs_blk:   torch.Tensor,
                slot_blk:  torch.Tensor) -> torch.Tensor:
    """
    Membership test for a hard permutation.

    Args:
        pred_slot: [B, n] int — pred_slot[b,i] = slot assigned to patch i
        obs_blk, slot_blk: [B, n] class ids
    Returns:
        [B] bool — True iff the permutation reconstructs the image exactly.

    A permutation is in 𝒪 iff every patch lands in a slot of its own class;
    since the classes have equal counts on both sides, that is both necessary
    and sufficient.
    """
    landed = torch.gather(slot_blk, 1, pred_slot)          # [B, n]
    return landed.eq(obs_blk).all(dim=-1)


# ══════════════════════════════════════════════════════════════════
# 4.  NEAREST ORBIT MEMBER  (the coupling)
# ══════════════════════════════════════════════════════════════════
#
#   P* = argmin_{P ∈ 𝒪} ‖X̃₀ − P‖_F  =  argmax_{P ∈ 𝒪} ⟨X̃₀, P⟩
#
# Two equivalent implementations:
#
#   "penalised"  (default, batched, GPU)
#       one n×n Hungarian problem on  −X̃₀ + λ·(1 − compat)  with λ large
#       enough that no incompatible cell can ever be chosen.  A compatible
#       assignment always exists, so the optimum is exactly argmax over 𝒪.
#       This is the "one Hungarian problem on cost[i,j] = ‖patch_i − orig_j‖²"
#       framing: the exact ties in that cost matrix *are* the ambiguity.
#
#   "blockwise" (reference, CPU)
#       the decomposition: identified patches are fixed, and each ambiguous
#       class contributes an independent m_b × m_b linear assignment on the
#       corresponding submatrix of X̃₀.  Cost O(Σ_b m_b³), independent of |𝒪|.
#
# They return the same matrix (up to ties); `check_couplings_agree()` asserts
# it.  The blockwise version is the one quoted in the paper for its cost.


_TLA_OK = HAS_TLA   # torch_linear_assignment ships an arch-specific CUDA
                    # kernel; on a GPU it was not compiled for it raises at
                    # call time.  We probe once and fall back to scipy.


def _hungarian_max(score: torch.Tensor) -> torch.Tensor:
    """
    Batched Hungarian on a *maximisation* score.

    Args:
        score: [B, n, n]
    Returns:
        assign: [B, n] int64 — column chosen for each row.
    """
    global _TLA_OK
    B, n, _ = score.shape
    s = score.nan_to_num(0.0).float()

    if _TLA_OK:
        try:
            return batch_linear_assignment(-s).long()
        except Exception as e:                          # pragma: no cover
            _TLA_OK = False
            print(f"[orbit.py] torch_linear_assignment unusable ({e}); "
                  f"falling back to scipy.")

    if not HAS_SCIPY:
        raise RuntimeError("need scipy or torch_linear_assignment for Hungarian")

    s_np   = s.detach().cpu().numpy()
    assign = np.empty((B, n), dtype=np.int64)
    for b in range(B):
        _, col = linear_sum_assignment(-s_np[b])
        assign[b] = col
    return torch.from_numpy(assign).to(score.device)


def nearest_orbit_member(X: torch.Tensor,
                         obs_blk:  torch.Tensor,
                         slot_blk: torch.Tensor) -> torch.Tensor:
    """
    P* = argmin_{P ∈ 𝒪} ‖X − P‖_F, batched, via one penalised n×n Hungarian.

    Args:
        X:        [B, n, n] — the noisy start X̃₀ (any real matrix)
        obs_blk, slot_blk: [B, n] class ids
    Returns:
        P_star: [B, n, n] hard permutation matrices, all members of 𝒪.
    """
    B, n, _ = X.shape
    compat  = compat_mask(obs_blk, slot_blk)                      # [B,n,n]

    # λ must exceed any achievable gain from an incompatible cell.
    lam = 2.0 * X.abs().amax(dim=(-2, -1), keepdim=True) * n + 1.0
    score = X - lam * (~compat).to(X.dtype)

    assign = _hungarian_max(score)                                 # [B, n]
    P = torch.zeros_like(X)
    P.scatter_(2, assign.unsqueeze(-1), 1.0)
    return P


def nearest_orbit_member_blockwise(X: torch.Tensor,
                                   obs_blk:  torch.Tensor,
                                   slot_blk: torch.Tensor) -> torch.Tensor:
    """
    Reference implementation of the same coupling, by the decomposition over
    the partition.  Identified patches are fixed; each ambiguous class solves
    an independent m_b × m_b linear assignment.  Cost O(Σ_b m_b³).

    Slower than `nearest_orbit_member` in wall-clock (Python loop over the
    batch) but it is the version whose cost bound the paper quotes, and it is
    the one used to verify the penalised formulation.
    """
    B, n, _ = X.shape
    X_np    = X.detach().float().cpu().numpy()
    obs_np  = obs_blk.detach().cpu().numpy()
    slot_np = slot_blk.detach().cpu().numpy()

    P = np.zeros((B, n, n), dtype=np.float32)
    for b in range(B):
        for cls in np.unique(obs_np[b]):
            rows = np.flatnonzero(obs_np[b]  == cls)
            cols = np.flatnonzero(slot_np[b] == cls)
            if rows.size == 1:                      # identified — fixed
                P[b, rows[0], cols[0]] = 1.0
                continue
            sub = X_np[b][np.ix_(rows, cols)]       # m_b × m_b
            r, c = linear_sum_assignment(-sub)      # maximise ⟨X, P⟩
            P[b, rows[r], cols[c]] = 1.0
    return torch.from_numpy(P).to(X.device, X.dtype)


def check_couplings_agree(X: torch.Tensor,
                          obs_blk:  torch.Tensor,
                          slot_blk: torch.Tensor) -> bool:
    """
    True iff the penalised and blockwise couplings attain the same objective
    ⟨X, P*⟩ (the argmin itself can differ when X has exact ties).
    """
    Pa = nearest_orbit_member(X, obs_blk, slot_blk)
    Pb = nearest_orbit_member_blockwise(X, obs_blk, slot_blk)
    ok_member = (is_in_orbit(Pa.argmax(-1), obs_blk, slot_blk).all()
                 and is_in_orbit(Pb.argmax(-1), obs_blk, slot_blk).all())
    va = (X * Pa).sum((-2, -1))
    vb = (X * Pb).sum((-2, -1))
    return bool(ok_member and torch.allclose(va, vb, atol=1e-4))


# ══════════════════════════════════════════════════════════════════
# 5.  CANONICAL KEY FOR A SAMPLED ORBIT MEMBER
# ══════════════════════════════════════════════════════════════════

def orbit_member_keys(pred_slot: torch.Tensor) -> list:
    """
    Hashable identity of each sampled permutation, for counting *distinct*
    orbit members among K draws.

    Args:
        pred_slot: [B, n] int
    Returns:
        list of B tuples
    """
    return [tuple(row) for row in pred_slot.detach().cpu().tolist()]
