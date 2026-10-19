"""
gumbel_sinkhorn_baseline.py -- Gumbel-Sinkhorn (GS) baseline for visual sorting (Section 5.1).

GS uses the same DigitEncoder and transformer as PermFlow (without the state
token) and reads the bilinear head as a score matrix S.  It is trained with
the squared error between Sinkhorn((S + G)/tau) (tau = 1, 20 log-domain
iterations, G i.i.d. Gumbel) and a one-hot valid order per example:

    --target a        always P_a
    --target b        always P_b
    --target random   P_a or P_b with probability 1/2, redrawn every step
    --target sample   P_a with probability alpha, else P_b, redrawn every step

Clean sequences have P_a = P_b.  Samples are decoded with

    --decoder soft    Hungarian on Sinkhorn((S + G)/tau)   (Table 1)
    --decoder gm      Hungarian on S + G (Gumbel-Matching)  (Appendix decoder table)

    python gumbel_sinkhorn_baseline.py --target random --seed 42
    python gumbel_sinkhorn_baseline.py --target random --seed 42 --skip_train --decoder gm --k 10 100
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import time
import torch
import torch.nn as nn
import numpy as np

from config  import CFG
from data    import build_loaders
from model   import DigitEncoder, VelocityTransformer
from metrics import full_eval


def _gumbel(shape, device, dtype=torch.float32) -> torch.Tensor:
    """i.i.d. Gumbel(0, 1) noise."""
    u = torch.rand(shape, device=device, dtype=dtype).clamp(1e-9, 1 - 1e-9)
    return -torch.log(-torch.log(u))


def _log_sinkhorn(log_S: torch.Tensor, n_iters: int) -> torch.Tensor:
    """Log-domain Sinkhorn normalisation."""
    for _ in range(n_iters):
        log_S = log_S - log_S.logsumexp(dim=-1, keepdim=True)
        log_S = log_S - log_S.logsumexp(dim=-2, keepdim=True)
    return torch.exp(log_S)


class GumbelSinkhornModel(nn.Module):
    """DigitEncoder + transformer of PermFlow's architecture, read as a score matrix."""

    def __init__(self, n=CFG.N, d_enc=CFG.D_ENC, tau=1.0, sk_iters=20,
                 target="random", decoder="soft"):
        super().__init__()
        self.n, self.tau, self.sk_iters = n, tau, sk_iters
        self.target, self.decoder = target, decoder
        self.encoder = DigitEncoder(d_enc=d_enc)
        self.score   = VelocityTransformer(n=n, d_enc=d_enc)

    def _scores(self, images):
        return self.score(self.encoder(images))                   # [B, N, N]

    def training_loss(self, images, P_a, P_b, alpha) -> torch.Tensor:
        S    = self._scores(images)
        soft = _log_sinkhorn((S + _gumbel(S.shape, S.device, S.dtype)) / self.tau,
                             self.sk_iters)
        a = alpha.view(-1, 1, 1)
        if self.target == "a":
            w = torch.ones_like(a)
        elif self.target == "b":
            w = torch.zeros_like(a)
        elif self.target == "random":
            w = (torch.rand_like(a) < 0.5).to(a.dtype)
        else:                                   # "sample": P_a with probability alpha
            w = (torch.rand_like(a) < a).to(a.dtype)
        P = w * P_a + (1.0 - w) * P_b
        return ((soft - P) ** 2).mean()

    @torch.no_grad()
    def sample(self, images, k_samples: int = 1) -> list:
        """K perturbed matrices; Hungarian rounding is applied by metrics.full_eval."""
        S = self._scores(images)
        out = []
        for _ in range(k_samples):
            Z = S + _gumbel(S.shape, S.device, S.dtype)
            out.append(Z if self.decoder == "gm" else _log_sinkhorn(Z / self.tau, self.sk_iters))
        return out


def train_gs(model, train_loader, device, epochs, lr, weight_decay, grad_clip, use_amp):
    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=CFG.LR_MIN)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot, n_ex = 0.0, 0
        for images, P_a, P_b, alpha, _ in train_loader:
            images, P_a, P_b, alpha = (x.to(device, non_blocking=True) for x in (images, P_a, P_b, alpha))
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                loss = model.training_loss(images, P_a, P_b, alpha)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt); scaler.update()
            tot += loss.item() * images.size(0); n_ex += images.size(0)
        sched.step()
        if ep % 10 == 0 or ep == epochs:
            print(f"epoch {ep:>4}  loss {tot / max(n_ex, 1):.4f}  [{time.time() - t0:.0f}s]", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Gumbel-Sinkhorn baseline for visual sorting.")
    p.add_argument("--target", default="random", choices=["a", "b", "random", "sample"])
    p.add_argument("--decoder", default="soft", choices=["soft", "gm"])
    p.add_argument("--seed", type=int, default=CFG.SEED,
                   help="training seed; data are always built with seed 42")
    p.add_argument("--epochs", type=int, default=CFG.N_EPOCHS)
    p.add_argument("--lr", type=float, default=CFG.LR)
    p.add_argument("--weight_decay", type=float, default=CFG.WEIGHT_DECAY)
    p.add_argument("--grad_clip", type=float, default=CFG.GRAD_CLIP)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--sinkhorn_iters", type=int, default=20)
    p.add_argument("--k", nargs="+", type=int, default=[5, 10, 20, 40, 60, 80, 100])
    p.add_argument("--skip_train", action="store_true", help="evaluate a saved checkpoint")
    args = p.parse_args()
    CFG.SEED = args.seed
    ckpt = f"{CFG.OUT_DIR}/gs_{args.target}_seed{args.seed}.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(CFG.SEED); np.random.seed(CFG.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CFG.SEED)
    os.makedirs(CFG.OUT_DIR, exist_ok=True)
    print(f"GS  target={args.target}  decoder={args.decoder}  seed={args.seed}")

    train_loader, _, amb_loader, cln_loader = build_loaders(42)
    model = GumbelSinkhornModel(n=CFG.N, d_enc=CFG.D_ENC, tau=args.tau,
                                sk_iters=args.sinkhorn_iters, target=args.target,
                                decoder=args.decoder).to(device)
    print(f"Parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    if args.skip_train:
        model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    else:
        train_gs(model, train_loader, device, args.epochs, args.lr,
                 args.weight_decay, args.grad_clip, CFG.USE_AMP)
        torch.save({"model_state": model.state_dict()}, ckpt)
        print(f"Checkpoint saved: {ckpt}")

    model.eval()
    print(f"\n{'K':>5}  {'CleanAcc':>9}  {'Coverage':>9}  {'AnyCor':>8}  {'CalErr':>8}")
    for k in args.k:
        r = full_eval(model, cln_loader, amb_loader, device, k_samples=k)
        print(f"{k:>5}  {r['clean_acc']*100:>8.1f}%  {r['coverage']*100:>8.1f}%  "
              f"{r['any_correct']*100:>7.1f}%  {r['calibration']:>8.4f}")


if __name__ == "__main__":
    main()
