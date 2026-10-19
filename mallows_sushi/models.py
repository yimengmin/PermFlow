"""
models.py — PermFlow and the baselines, all on the same transformer backbone.

  PermFlow   : paper's flow (path X_t = (1-t)X̃₀ + tP, target P − X̃₀,
               velocity sees X_t and t, tangent projection C(·)C).  With one
               observed order per example the coupling is the standard
               independent CFM coupling.
  GS         : Gumbel-Sinkhorn, MSE between Sinkhorn((S+G)/τ) and the one-hot
               observed P; samples = Hungarian(S + G).
  PL         : Plackett-Luce, exact likelihood; samples = argsort(s + G).
  PLMix      : mixture of M Plackett-Luce components, exact likelihood.

Every model exposes  loss(feats, orders)  and  sample(feats, K, ...) -> [B, K, n].
"""
import numpy as np
import torch
import torch.nn as nn

from perm_utils import (tangent_project, sample_tangent_noise, orders_to_P,
                        hungarian_orders)


def _gumbel(shape, device):
    u = torch.rand(shape, device=device).clamp(1e-9, 1 - 1e-9)
    return -torch.log(-torch.log(u))


class Backbone(nn.Module):
    """Item tokens -> pre-LN transformer -> per-item embeddings [B, n, d].

    xt_mode (flow only): "token" = X_t flattened into one extra token (paper);
                         "rows"  = additionally add row X_t[i, :] to item token i.
    """

    def __init__(self, d_in, n, d_model=128, nlayers=4, nheads=4, dropout=0.1,
                 use_xt=False, use_t=False, xt_mode="token"):
        super().__init__()
        self.n, self.use_xt, self.use_t, self.xt_mode = n, use_xt, use_t, xt_mode
        self.inp = nn.Sequential(nn.Linear(d_in, d_model), nn.GELU(),
                                 nn.Linear(d_model, d_model))
        if use_xt:
            self.xt = nn.Sequential(nn.Linear(n * n, d_model), nn.LayerNorm(d_model))
            if xt_mode == "rows":
                self.xrow = nn.Linear(n, d_model)
        if use_t:
            self.tp = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(),
                                    nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nheads, 4 * d_model, dropout,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.tf = nn.TransformerEncoder(layer, nlayers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, feats, X=None, t=None):
        B = feats.shape[0]
        h = self.inp(feats)
        if self.use_xt:
            if self.xt_mode == "rows":
                h = h + self.xrow(X)
            tok = self.xt(X.reshape(B, -1))
            if self.use_t:
                tok = tok + self.tp(t.reshape(B, 1))
            h = torch.cat([tok.unsqueeze(1), h], 1)
        h = self.ln(self.tf(h))
        return h[:, 1:] if self.use_xt else h


class _MatrixHead(nn.Module):
    """per-item embeddings -> [B, n, n] item x rank matrix (bilinear)."""

    def __init__(self, n, d_model):
        super().__init__()
        self.rank = nn.Embedding(n, d_model)
        self.scale = d_model ** -0.5

    def forward(self, h):
        return h @ self.rank.weight.T * self.scale


# ══════════════════════════════════════════════════════════════════
class PermFlow(nn.Module):
    def __init__(self, d_in, n, sigma0=1.0, xt_mode="token", **kw):
        super().__init__()
        self.n, self.sigma0 = n, sigma0
        self.net = Backbone(d_in, n, use_xt=True, use_t=True, xt_mode=xt_mode, **kw)
        self.head = _MatrixHead(n, self.net.ln.normalized_shape[0])

    def velocity(self, feats, X, t):
        return tangent_project(self.head(self.net(feats, X, t)))

    def _start(self, B, device):
        X0 = torch.full((B, self.n, self.n), 1.0 / self.n, device=device)
        return X0 + self.sigma0 * sample_tangent_noise(B, self.n, device)

    def loss(self, feats, orders):
        B, dev = feats.shape[0], feats.device
        P = orders_to_P(orders, self.n)
        Xn = self._start(B, dev)
        t = torch.rand(B, device=dev).view(B, 1, 1)
        Xt = (1 - t) * Xn + t * P
        v = self.velocity(feats, Xt, t.view(B))
        return ((v - (P - Xn)) ** 2).mean()

    @torch.no_grad()
    def sample(self, feats, K, steps=10, chunk=8192):
        B, dev = feats.shape[0], feats.device
        f = feats.repeat_interleave(K, 0)                 # [B*K, n, d]
        out = []
        for s0 in range(0, f.shape[0], chunk):
            fc = f[s0:s0 + chunk]
            X = self._start(fc.shape[0], dev)
            dt = 1.0 / steps
            for s in range(steps):
                t = torch.full((fc.shape[0],), s * dt, device=dev)
                X = X + dt * self.velocity(fc, X, t)
            out.append(X.float().cpu().numpy())
        return hungarian_orders(np.concatenate(out)).reshape(B, K, self.n)


# ══════════════════════════════════════════════════════════════════
class GS(nn.Module):
    def __init__(self, d_in, n, tau=1.0, sk_iters=20, **kw):
        super().__init__()
        self.n, self.tau, self.sk_iters = n, tau, sk_iters
        self.net = Backbone(d_in, n, **kw)
        self.head = _MatrixHead(n, self.net.ln.normalized_shape[0])

    def scores(self, feats):
        return self.head(self.net(feats))

    def _sinkhorn(self, L):
        for _ in range(self.sk_iters):
            L = L - L.logsumexp(-1, keepdim=True)
            L = L - L.logsumexp(-2, keepdim=True)
        return L.exp()

    def loss(self, feats, orders):
        S = self.scores(feats)
        soft = self._sinkhorn((S + _gumbel(S.shape, S.device)) / self.tau)
        return ((soft - orders_to_P(orders, self.n)) ** 2).mean()

    @torch.no_grad()
    def sample(self, feats, K, noise=True, decoder="gm", **_):
        """decoder="gm": Hungarian on S+G (Gumbel-Matching);
        decoder="soft": Hungarian on Sinkhorn((S+G)/tau), which is not the same."""
        S = self.scores(feats)                                    # [B, n, n]
        Sk = S.unsqueeze(1).expand(-1, K, -1, -1)
        if noise:
            Sk = Sk + _gumbel(Sk.shape, S.device)
        if decoder == "soft":
            Sk = self._sinkhorn(Sk / self.tau)
        B = S.shape[0]
        return hungarian_orders(Sk.reshape(B * K, self.n, self.n).float().cpu().numpy()
                                ).reshape(B, K, self.n)


# ══════════════════════════════════════════════════════════════════
def _pl_loglik(s, orders):
    """s [..., n] item scores; orders [..., n] -> log P_PL(order)."""
    so = s.gather(-1, orders)
    lse = torch.logcumsumexp(so.flip(-1), -1).flip(-1)       # logsumexp of remaining items
    return (so - lse).sum(-1)


class PL(nn.Module):
    def __init__(self, d_in, n, **kw):
        super().__init__()
        self.n = n
        self.net = Backbone(d_in, n, **kw)
        self.out = nn.Linear(self.net.ln.normalized_shape[0], 1)

    def scores(self, feats):
        return self.out(self.net(feats)).squeeze(-1)             # [B, n]

    def loss(self, feats, orders):
        return -_pl_loglik(self.scores(feats), orders).mean()

    @torch.no_grad()
    def sample(self, feats, K, **_):
        s = self.scores(feats).unsqueeze(1)                       # [B, 1, n]
        g = s + _gumbel((s.shape[0], K, self.n), s.device)
        return torch.argsort(g, -1, descending=True).cpu().numpy()


class PLMix(nn.Module):
    def __init__(self, d_in, n, M=4, **kw):
        super().__init__()
        self.n, self.M = n, M
        self.net = Backbone(d_in, n, **kw)
        d = self.net.ln.normalized_shape[0]
        self.out = nn.Linear(d, M)
        self.mix = nn.Linear(d, M)

    def _params(self, feats):
        h = self.net(feats)
        s = self.out(h).transpose(1, 2)                           # [B, M, n]
        logw = self.mix(h.mean(1)).log_softmax(-1)                # [B, M]
        return s, logw

    def loss(self, feats, orders):
        s, logw = self._params(feats)
        ll = _pl_loglik(s, orders.unsqueeze(1).expand(-1, self.M, -1))
        return -(logw + ll).logsumexp(-1).mean()

    @torch.no_grad()
    def sample(self, feats, K, **_):
        s, logw = self._params(feats)
        B = s.shape[0]
        m = torch.multinomial(logw.exp(), K, replacement=True)    # [B, K]
        sk = s.gather(1, m.unsqueeze(-1).expand(-1, -1, self.n))
        g = sk + _gumbel(sk.shape, s.device)
        return torch.argsort(g, -1, descending=True).cpu().numpy()


def build_model(name, d_in, n, args):
    kw = dict(d_model=args.d_model, nlayers=args.nlayers, nheads=args.nheads,
              dropout=args.dropout)
    if name == "permflow":
        return PermFlow(d_in, n, sigma0=args.sigma0, xt_mode=args.xt_mode, **kw)
    if name == "gs":
        return GS(d_in, n, tau=args.gs_tau, sk_iters=args.gs_iters, **kw)
    if name == "pl":
        return PL(d_in, n, **kw)
    if name == "plmix":
        return PLMix(d_in, n, M=args.plmix_M, **kw)
    raise ValueError(name)
