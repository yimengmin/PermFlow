"""
data.py — conditional Mallows mixture (synthetic, exact ground truth) and the
Sushi preference dataset (Kamishima 2003, item set A: 5000 users x 10 items).

Both expose per-example token features  feats [B, n, d_in]  (one token per
item: item features concatenated with the context) and target orders [B, n].
"""
import os

import numpy as np
import torch

from perm_utils import kendall_dist, mallows_logZ


# ══════════════════════════════════════════════════════════════════
# 1.  CONDITIONAL MALLOWS MIXTURE
# ══════════════════════════════════════════════════════════════════
class MallowsMixture:
    """
    x ~ N(0, I_d).  Component m has centre c_m(x) = argsort(-E A_m x) and
    weight w(x) = softmax(beta U x).  pi | x ~ sum_m w_m Mallows(c_m(x), theta).

    reverse=True (M = 2): A_2 = -A_1, so the two centres are exact reversals —
    the first-order marginal of the mixture is then close to uniform and any
    model whose samples are determined by one score vector / matrix plus
    independent noise (Plackett-Luce, Gumbel-Sinkhorn) cannot represent it.
    """

    def __init__(self, n=8, M=2, theta=1.0, d=8, beta=2.0, reverse=True, seed=0):
        rng = np.random.default_rng(seed)
        self.n, self.M, self.theta, self.d, self.beta = n, M, theta, d, beta
        self.E = rng.standard_normal((n, d))                     # item features
        A = rng.standard_normal((M, d, d)) / np.sqrt(d)
        if reverse:
            assert M == 2
            A[1] = -A[0]
        self.A = A
        self.U = rng.standard_normal((M, d)) / np.sqrt(d)
        self.logZ = mallows_logZ(n, theta)
        self.d_in = 2 * d

    # ── ground truth ─────────────────────────────────────────────
    def centres(self, x: np.ndarray) -> np.ndarray:
        """[B, d] -> [B, M, n] orders."""
        s = np.einsum("nd,mde,be->bmn", self.E, self.A, x)
        return np.argsort(-s, axis=-1)

    def weights(self, x: np.ndarray) -> np.ndarray:
        z = self.beta * x @ self.U.T
        z = z - z.max(-1, keepdims=True)
        w = np.exp(z)
        return w / w.sum(-1, keepdims=True)

    def logprob(self, orders: np.ndarray, x: np.ndarray) -> np.ndarray:
        """log p(orders | x) for one context x [d]; orders [N, n]."""
        C, w = self.centres(x[None])[0], self.weights(x[None])[0]
        lp = np.stack([np.log(w[m]) - self.theta * kendall_dist(orders, C[m])
                       for m in range(self.M)], 0)
        mx = lp.max(0)
        return mx + np.log(np.exp(lp - mx).sum(0)) - self.logZ

    # ── sampling (repeated insertion model) ──────────────────────
    def sample_mallows(self, centre: np.ndarray, rng) -> np.ndarray:
        """centre [B, n] -> one Mallows(centre, theta) order per row."""
        B, n = centre.shape
        cur = np.full((B, n), -1, dtype=np.int64)
        for i in range(n):
            # insert centre[:, i] at position j in 0..i, P(j) ∝ exp(-theta (i - j))
            logits = -self.theta * (i - np.arange(i + 1))
            p = np.exp(logits - logits.max()); p /= p.sum()
            pos = rng.choice(i + 1, size=B, p=p)
            idx = np.arange(i + 1)[None, :]
            prev = np.take_along_axis(cur, np.clip(idx - (idx > pos[:, None]), 0, n - 1), 1)
            new = np.where(idx == pos[:, None], centre[:, i:i + 1], prev)
            cur[:, :i + 1] = new
        return cur

    def sample(self, x: np.ndarray, rng) -> np.ndarray:
        B = x.shape[0]
        C, w = self.centres(x), self.weights(x)
        m = (rng.random((B, 1)) > np.cumsum(w, -1)).sum(-1).clip(max=self.M - 1)
        return self.sample_mallows(C[np.arange(B), m], rng)

    def contexts(self, B: int, rng) -> np.ndarray:
        return rng.standard_normal((B, self.d))

    def feats(self, x: np.ndarray) -> torch.Tensor:
        """[B, d] -> [B, n, 2d] tokens: item features ⊕ context."""
        B = x.shape[0]
        E = np.broadcast_to(self.E[None], (B, self.n, self.d))
        X = np.broadcast_to(x[:, None, :], (B, self.n, self.d))
        return torch.tensor(np.concatenate([E, X], -1), dtype=torch.float32)

    def batch(self, B: int, rng):
        x = self.contexts(B, rng)
        return self.feats(x), torch.from_numpy(self.sample(x, rng))


# ══════════════════════════════════════════════════════════════════
# 2.  SUSHI (item set A)
# ══════════════════════════════════════════════════════════════════
SUSHI_A_TO_B = [0, 1, 2, 3, 4, 6, 7, 8, 26, 29]   # set-A ids -> rows of sushi3.idata


class Sushi:
    def __init__(self, root=None, n_test=1000, seed=0):
        root = root or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "data", "sushi3-2016")
        orders = np.loadtxt(os.path.join(root, "sushi3a.5000.10.order"),
                            skiprows=1, dtype=np.int64)[:, 2:]
        U = np.loadtxt(os.path.join(root, "sushi3.udata"), dtype=np.float64)
        rows = [l.rstrip("\n").split("\t")
                for l in open(os.path.join(root, "sushi3.idata"), encoding="latin-1")]
        I = np.array([[float(v) for v in r[2:]] for r in rows])[SUSHI_A_TO_B]

        # item features: id one-hot, style, major, minor one-hot(12), 4 numeric
        n = 10
        num = (I[:, 3:7] - I[:, 3:7].mean(0)) / (I[:, 3:7].std(0) + 1e-8)
        self.item = np.concatenate([np.eye(n), I[:, 0:2], np.eye(12)[I[:, 2].astype(int)], num], 1)
        # user features: gender, age(6), region<15(12), e/w<15, region now(12), e/w now, moved
        u = U
        self.user = np.concatenate([
            u[:, 1:2], np.eye(6)[u[:, 2].astype(int)],
            np.eye(12)[u[:, 5].astype(int)], u[:, 6:7],
            np.eye(12)[u[:, 8].astype(int)], u[:, 9:10], u[:, 10:11]], 1)

        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(orders))
        self.test_idx, self.train_idx = perm[:n_test], perm[n_test:]
        self.orders, self.n = orders, n
        self.d_in = self.item.shape[1] + self.user.shape[1]

    def feats(self, idx: np.ndarray) -> torch.Tensor:
        B = len(idx)
        It = np.broadcast_to(self.item[None], (B, self.n, self.item.shape[1]))
        Ut = np.broadcast_to(self.user[idx][:, None, :], (B, self.n, self.user.shape[1]))
        return torch.tensor(np.concatenate([It, Ut], -1), dtype=torch.float32)

    def batch(self, B: int, rng):
        idx = rng.choice(self.train_idx, size=B)
        return self.feats(idx), torch.from_numpy(self.orders[idx])
