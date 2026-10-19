"""
metrics.py — distributional metrics.

Synthetic (ground truth known), per test context x with K model samples:
  tv        : TV(empirical model distribution, true p(·|x)) over all of S_n
              (n <= 8, exact enumeration)
  mode_tv   : TV between the distribution of "nearest centre" (argmin Kendall
              distance to c_m(x)) under the model and under the truth — the
              M-way analogue of MNIST's calibration error
  pair_mae  : mean |P(i ranked above j)| error over item pairs
  off_mode  : fraction of samples farther (Kendall) from every centre than the
              true 95% quantile, minus the true mass there (0.05 ideally)
Each metric is also computed for K exact samples from the truth: the
finite-sample floor ("oracle").

Sushi (held-out users, K samples per user):
  tau_mean / tau_best : Kendall-τ between a sample and the user's own order
                        (mean over samples / best of K)
  top1_tv, pos_mae, pair_mae, top3_tv : aggregate statistics of the samples vs
                        the held-out empirical orders; the "train" row compares
                        the training orders with the test orders (data floor).
"""
import numpy as np

from perm_utils import PermIndex, kendall_dist, positions


def _pair_matrix(orders):
    """orders [N, n] -> [n, n], entry (i, j) = frac with i ranked above j."""
    pos = positions(orders)
    return (pos[:, :, None] < pos[:, None, :]).mean(0)


def _iu(n):
    return np.triu_indices(n, 1)


# ══════════════════════════════════════════════════════════════════
class SynthEvaluator:
    def __init__(self, mm, n_ctx=200, K=1000, seed=123):
        self.mm, self.K = mm, K
        rng = np.random.default_rng(seed)
        self.x = mm.contexts(n_ctx, rng)
        self.exact = mm.n <= 8
        self.ref = []                                  # per-context ground-truth summaries
        if self.exact:
            self.pidx = PermIndex(mm.n)
            A = self.pidx.all
            posA = positions(A)
            pairA = (posA[:, :, None] < posA[:, None, :]).reshape(len(A), -1)
        for x in self.x:
            C = mm.centres(x[None])[0]
            if self.exact:
                p = np.exp(mm.logprob(A, x)); p /= p.sum()
                D = np.stack([kendall_dist(A, C[m]) for m in range(mm.M)], 1)
                mode = D.argmin(1); dmin = D.min(1)
                mode_p = np.bincount(mode, weights=p, minlength=mm.M)
                order = np.argsort(dmin); cdf = np.cumsum(p[order])
                thr = dmin[order][np.searchsorted(cdf, 0.95)]
                tail = p[dmin > thr].sum()
                pair = (p @ pairA).reshape(mm.n, mm.n)
                self.ref.append(dict(p=p, C=C, mode_p=mode_p, thr=thr, tail=tail, pair=pair))
            else:
                S = mm.sample(np.repeat(x[None], 20000, 0), rng)
                D = np.stack([kendall_dist(S, C[m]) for m in range(mm.M)], 1)
                dmin = D.min(1)
                thr = np.quantile(dmin, 0.95)
                self.ref.append(dict(C=C, mode_p=np.bincount(D.argmin(1), minlength=mm.M) / len(S),
                                     thr=thr, tail=(dmin > thr).mean(), pair=_pair_matrix(S)))

    def _score(self, samples):
        """samples [n_ctx, K, n] -> dict of means."""
        mm, iu = self.mm, _iu(self.mm.n)
        out = {k: [] for k in ["tv", "mode_tv", "pair_mae", "off_mode"]}
        for c, S in enumerate(samples):
            r = self.ref[c]
            D = np.stack([kendall_dist(S, r["C"][m]) for m in range(mm.M)], 1)
            mode_q = np.bincount(D.argmin(1), minlength=mm.M) / len(S)
            out["mode_tv"].append(0.5 * np.abs(mode_q - r["mode_p"]).sum())
            out["off_mode"].append((D.min(1) > r["thr"]).mean() - r["tail"])
            out["pair_mae"].append(np.abs(_pair_matrix(S) - r["pair"])[iu].mean())
            if self.exact:
                q = np.bincount(self.pidx.index(S), minlength=len(r["p"])) / len(S)
                out["tv"].append(0.5 * np.abs(q - r["p"]).sum())
        return {k: float(np.mean(v)) for k, v in out.items() if v}

    def oracle(self, seed=7):
        rng = np.random.default_rng(seed)
        S = np.stack([self.mm.sample(np.repeat(x[None], self.K, 0), rng) for x in self.x])
        return self._score(S)

    def evaluate(self, model, device, **sample_kw):
        import torch
        out = []
        for s0 in range(0, len(self.x), 25):
            f = self.mm.feats(self.x[s0:s0 + 25]).to(device)
            out.append(model.sample(f, self.K, **sample_kw))
        return self._score(np.concatenate(out))


# ══════════════════════════════════════════════════════════════════
def _top3_codes(orders, n):
    return orders[:, 0] * n * n + orders[:, 1] * n + orders[:, 2]


def _agg_stats(orders, n):
    pos = positions(orders)
    return dict(top1=np.bincount(orders[:, 0], minlength=n) / len(orders),
                pos=np.stack([np.bincount(pos[:, i], minlength=n) for i in range(n)]) / len(orders),
                pair=_pair_matrix(orders),
                top3=np.bincount(_top3_codes(orders, n), minlength=n ** 3) / len(orders))


def _agg_compare(a, b, n):
    iu = _iu(n)
    return dict(top1_tv=0.5 * np.abs(a["top1"] - b["top1"]).sum(),
                pos_mae=np.abs(a["pos"] - b["pos"]).mean(),
                pair_mae=np.abs(a["pair"] - b["pair"])[iu].mean(),
                top3_tv=0.5 * np.abs(a["top3"] - b["top3"]).sum())


def _kendall_tau(S, y, n):
    """S [N, K, n], y [N, n] -> tau [N, K]."""
    N, K, _ = S.shape
    d = kendall_dist(S.reshape(N * K, n), np.repeat(y, K, 0)).reshape(N, K)
    return 1 - 4 * d / (n * (n - 1))


class SushiEvaluator:
    def __init__(self, ds, K=100):
        self.ds, self.K, self.n = ds, K, ds.n
        self.y = ds.orders[ds.test_idx]
        self.test_stats = _agg_stats(self.y, self.n)

    def data_floor(self):
        r = _agg_compare(_agg_stats(self.ds.orders[self.ds.train_idx], self.n),
                         self.test_stats, self.n)
        return {k: float(v) for k, v in r.items()}

    def evaluate(self, model, device, **sample_kw):
        out = []
        idx = self.ds.test_idx
        for s0 in range(0, len(idx), 100):
            f = self.ds.feats(idx[s0:s0 + 100]).to(device)
            out.append(model.sample(f, self.K, **sample_kw))
        S = np.concatenate(out)                                   # [N, K, n]
        tau = _kendall_tau(S, self.y, self.n)
        # aggregate: one sample per user, averaged over the K draws
        agg = [ _agg_compare(_agg_stats(S[:, k], self.n), self.test_stats, self.n)
                for k in range(min(self.K, 20)) ]
        r = {k: float(np.mean([a[k] for a in agg])) for k in agg[0]}
        r.update(tau_mean=float(tau.mean()), tau_best=float(tau.max(1).mean()))
        return r
