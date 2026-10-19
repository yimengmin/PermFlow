"""
perm_utils.py — permutation helpers shared by data, models and metrics.

Conventions
-----------
order : int array [..., n], order[r] = item placed at rank r (rank 0 = best)
P     : [..., n, n] matrix, P[i, r] = 1 iff item i is at rank r
"""
from itertools import permutations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


# ── tangent space of the affine set {X1 = 1, Xᵀ1 = 1} ─────────────────
def tangent_project(U: torch.Tensor) -> torch.Tensor:
    return U - U.mean(-1, keepdim=True) - U.mean(-2, keepdim=True) \
             + U.mean((-2, -1), keepdim=True)


def sample_tangent_noise(B: int, n: int, device, dtype=torch.float32) -> torch.Tensor:
    """Unit-Frobenius-norm noise in the tangent space (same as the paper)."""
    e = tangent_project(torch.randn(B, n, n, device=device, dtype=dtype))
    return e / e.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-8)


# ── order <-> matrix ─────────────────────────────────────────────────
def orders_to_P(orders: torch.Tensor, n: int) -> torch.Tensor:
    """[B, n] long -> [B, n, n] float, P[b, item, rank] = 1."""
    B = orders.shape[0]
    P = torch.zeros(B, n, n, device=orders.device)
    ranks = torch.arange(n, device=orders.device).expand(B, n)
    P[torch.arange(B, device=orders.device).unsqueeze(1), orders, ranks] = 1.0
    return P


def hungarian_orders(X: np.ndarray) -> np.ndarray:
    """Round [N, n, n] real matrices (item x rank) to orders [N, n] by max <X, P>."""
    N, n, _ = X.shape
    out = np.empty((N, n), dtype=np.int64)
    for b in range(N):
        items, ranks = linear_sum_assignment(-X[b])
        out[b, ranks] = items
    return out


def positions(orders: np.ndarray) -> np.ndarray:
    """pos[..., item] = rank of item."""
    return np.argsort(orders, axis=-1)


def kendall_dist(orders: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Number of discordant item pairs. orders [N, n]; center [n] or [N, n]."""
    n = orders.shape[-1]
    iu0, iu1 = np.triu_indices(n, 1)
    po = positions(orders)
    pc = positions(np.broadcast_to(center, orders.shape))
    return (((po[:, iu0] - po[:, iu1]) * (pc[:, iu0] - pc[:, iu1])) < 0).sum(-1)


# ── exact enumeration of S_n (n <= 8) ────────────────────────────────
class PermIndex:
    """All n! orders with a fast order -> index lookup."""

    def __init__(self, n: int):
        assert n <= 8, "enumeration only for n <= 8"
        self.n = n
        self.all = np.array(list(permutations(range(n))), dtype=np.int64)
        self._w = n ** np.arange(n, dtype=np.int64)
        codes = self.all @ self._w
        self._sort = np.argsort(codes)
        self._codes = codes[self._sort]

    def index(self, orders: np.ndarray) -> np.ndarray:
        c = orders.astype(np.int64) @ self._w
        return self._sort[np.searchsorted(self._codes, c)]


def mallows_logZ(n: int, theta: float) -> float:
    """log normaliser of Mallows with Kendall distance."""
    q = np.exp(-theta)
    return float(sum(np.log((1 - q ** i) / (1 - q)) for i in range(1, n + 1)))
