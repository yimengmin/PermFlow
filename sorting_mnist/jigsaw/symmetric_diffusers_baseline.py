"""
symmetric_diffusers_baseline.py  —  head-to-head against the released code
══════════════════════════════════════════════════════════════════════════

This drives the *authors' own* implementation (Zhang, Yang & Liao, ICLR 2025,
https://github.com/DSL-Lab/SymmetricDiffusers) on our jigsaw loaders:
`models.ReverseDiffusion` and `diffusion.DiffusionUtils` are imported
unmodified from a checkout of their repo.  Run

    bash setup_symmetric_diffusers.sh

once to place that checkout in ./third_party/SymmetricDiffusers.

SUPERVISION — the same target set, for free
───────────────────────────────────────────
Their training signal is `x_start`, the patch tensor in its *original slot
order*, which the forward riffle shuffle then scrambles.  Every member of 𝒪
produces a byte-identical ordered patch tensor — that is what being in the
orbit means — so training on x_start supervises against the whole target set
and never against one distinguished τ.  No modification is needed to put
SymmetricDiffusers on the same footing as PermFlow and Gumbel-Sinkhorn here;
we only have to *decode* it correctly.

DECODING — sampling, not beam search
────────────────────────────────────
Their released eval decodes with `p_sample_beam_search`, which returns the
modes of p_θ.  Using that here would make a genuine distribution learner look
artificially unimodal — the same error this whole experiment exists to avoid.
We therefore decode with `p_sample_loop(..., deterministic=False)`, drawing K
independent reverse trajectories, exactly the sampler their own code exposes.
`--decode beam` is available for reference and is flagged in the output as
mode-seeking; do not use it for the distributional metrics.

HYPERPARAMETERS
───────────────
Taken verbatim from their released configs
(configs/unscramble-*/config_*_{g}x{g}.json): riffle-shuffle forward process,
generalised-PL reverse, per-g diffusion length and denoising schedule, the
same 7-layer/8-head/d-128/FFN-512 transformer and the same CNN, AdamW
(lr 1e-5, β=(0.9,0.98), wd 1e-2, eps 1e-9) with their Noam warm-up schedule.

Usage
-----
    python -u symmetric_diffusers_baseline.py --grid 4 --sigma_pix 0
    python -u symmetric_diffusers_baseline.py --grid 4 --sigma_pix 0.01
    python -u symmetric_diffusers_baseline.py --grid 4 --skip_train \
        --ckpt outputs/sd_mnist_g4_sig0.pt --k 5 10 20
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import time

import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler

from config import CFG
from data import build_loaders
from metrics import evaluate, format_results, reconstruct


# ══════════════════════════════════════════════════════════════════
# 1.  THEIR RELEASED CODE
# ══════════════════════════════════════════════════════════════════

DEFAULT_SD_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "third_party", "SymmetricDiffusers")


def import_symmetric_diffusers(repo: str):
    """Import ReverseDiffusion / DiffusionUtils from a checkout of their repo."""
    if not os.path.isdir(repo):
        raise SystemExit(
            f"SymmetricDiffusers checkout not found at {repo}\n"
            f"Run:  bash setup_symmetric_diffusers.sh")
    sys.path.insert(0, repo)
    import models as sd_models              # noqa: E402
    from diffusion import DiffusionUtils    # noqa: E402
    return sd_models, DiffusionUtils


# Diffusion length and denoising schedule, per their released configs.
SD_SCHEDULE = {
    ("mnist", 4): (10, [0, 4, 6, 10]),
    ("mnist", 5): (11, [0, 5, 7, 11]),
    ("mnist", 6): (12, [0, 6, 8, 12]),
}
SD_WARMUP = {"mnist": 51_600}


class NoamScheduler(_LRScheduler):
    """Their `TransformerScheduler`, reimplemented so we do not need to pull in
    the `transformers` package for a five-line formula."""

    def __init__(self, optimizer, dim_embed, warmup_steps, last_epoch=-1):
        self.dim_embed        = dim_embed
        self.warmup_steps     = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        lr = self.dim_embed ** (-0.5) * min(
            self._step_count ** (-0.5),
            self._step_count * self.warmup_steps ** (-1.5))
        return [lr] * self.num_param_groups


def build_sd(repo: str, dataset: str, g: int, device, beam_pl: int = 200,
             beam_time: int = 20):
    """Instantiate their reverse model + diffusion utils for this task."""
    sd_models, DiffusionUtils = import_symmetric_diffusers(repo)
    T, reverse_steps = SD_SCHEDULE[(dataset, g)]

    model = sd_models.ReverseDiffusion(
        "unscramble-noisy-MNIST",
        1,                                # in_channels
        g,                                # num_pieces (per side)
        28,                               # image_size
        32, 3, 1, 1,                      # CNN block 1  (hidden, k, s, p)
        64, 5, 1, 1,                      # CNN block 2  (unused for jigsaw)
        4,                                # num_digits   (unused for jigsaw)
        CFG.D_MODEL, CFG.NHEADS, CFG.D_FFN, CFG.NLAYERS, CFG.DROPOUT,
        "square",                         # d_out_adjust for generalized_PL
        True,                             # use_pos_enc
    ).to(device)

    diffusion = DiffusionUtils(
        num_timesteps      = T,
        sample_N           = 3,
        transition         = "riffle",
        latent             = True,
        reinforce_N        = 10,
        reinforce_ema_rate = 0.995,
        entropy_reg_rate   = 0.05,
        reverse            = "generalized_PL",
        reverse_steps      = reverse_steps,
        loss               = "log_likelihood",
        beam_size          = {"PL": beam_pl, "time": beam_time},
        perm_fix_first     = False,
    ).to(device)

    return model, diffusion, T, reverse_steps


# ══════════════════════════════════════════════════════════════════
# 2.  SAMPLING WRAPPER  (matches FlowModel.sample so metrics.evaluate reuses)
# ══════════════════════════════════════════════════════════════════

class SDSampler:
    """
    Exposes `.sample(patches, k_samples) -> list of K [B, n, n]` on top of
    their diffusion object.

    decode="sample" : K independent draws of the reverse process
                      (`p_sample_loop(deterministic=False)`).  This is the
                      correct decoder for any distributional claim.
    decode="beam"   : their released beam-search decoder.  Mode-seeking and
                      deterministic — all K draws coincide.
    """

    def __init__(self, model, diffusion, decode: str = "sample"):
        self.model, self.diffusion, self.decode = model, diffusion, decode

    def eval(self):
        self.model.eval()
        return self

    @torch.no_grad()
    def sample(self, patches: torch.Tensor, k_samples: int = 1) -> list:
        B, n = patches.shape[:2]
        out  = []
        for _ in range(k_samples):
            if self.decode == "beam":
                _, perm = self.diffusion.p_sample_beam_search(patches, self.model)
            else:
                _, perm = self.diffusion.p_sample_loop(
                    patches, self.model, deterministic=False)
            # perm[j] = index of the patch placed at slot j → pred_slot = argsort
            pred_slot = torch.argsort(perm, dim=-1)
            P = torch.zeros(B, n, n, device=patches.device)
            P.scatter_(2, pred_slot.unsqueeze(-1), 1.0)
            out.append(P)
        return out


# ══════════════════════════════════════════════════════════════════
# 3.  TRAINING  (their loop, single-GPU)
# ══════════════════════════════════════════════════════════════════

def train_sd(model, diffusion, train_loader, dist_loader, device,
             epochs: int, lr: float, warmup: int, eval_every: int,
             k_samples: int, grad_clip: float = 1.0) -> dict:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98),
                            weight_decay=1e-2, eps=1e-9)
    sched = NoamScheduler(opt, dim_embed=CFG.D_MODEL, warmup_steps=warmup)

    history = dict(epoch=[], train_loss=[], validity=[], distinct=[],
                   marginal_tv=[], rmse=[])
    hdr = (f"{'Ep':>5}  {'Loss':>11}  {'Valid':>8}  {'Dist@K':>8}  "
           f"{'margTV':>8}  {'RMSE':>8}  {'Time':>7}")
    print(f"\n{hdr}")
    print("─" * len(hdr))
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        diffusion.train()
        tot, nb = 0.0, 0
        for patches, gt_slot, obs_blk, slot_blk, log_orbit in train_loader:
            patches = patches.to(device, non_blocking=True)
            gt_slot = gt_slot.to(device, non_blocking=True)

            # x_start = the patches in original slot order.  Identical for
            # every member of 𝒪 ⇒ supervision is against the target set.
            x_start = reconstruct(patches, gt_slot)

            loss = diffusion.training_losses(x_start, model)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            sched.step()
            tot += loss.item(); nb += 1

        if ep % eval_every == 0 or ep == 1 or ep == epochs:
            sampler = SDSampler(model, diffusion, "sample").eval()
            r = evaluate(sampler, dist_loader, device, k_samples=k_samples)
            history["epoch"].append(ep)
            history["train_loss"].append(tot / max(nb, 1))
            history["validity"].append(r["validity"])
            history["distinct"].append(r["distinct"])
            history["marginal_tv"].append(r["marginal_tv"])
            history["rmse"].append(r["rmse"])
            print(f"{ep:>5}  {tot/max(nb,1):>11.5f}  {r['validity']*100:>7.2f}%  "
                  f"{r['distinct']*100:>7.2f}%  {r['marginal_tv']:>8.4f}  "
                  f"{r['rmse']:>8.5f}  {time.time()-t0:>6.0f}s")

    print("─" * len(hdr))
    return history


# ══════════════════════════════════════════════════════════════════
# 4.  MAIN
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="SymmetricDiffusers baseline (released code, sampled decode).")
    p.add_argument("--grid",      type=int,   default=CFG.GRID)
    p.add_argument("--sigma_pix", type=float, default=CFG.SIGMA_PIX)
    p.add_argument("--epochs",    type=int,   default=120)   # their setting
    p.add_argument("--lr",        type=float, default=1e-5)  # their setting
    p.add_argument("--batch_size", type=int,  default=64)    # their setting
    p.add_argument("--warmup",    type=int,   default=None)
    p.add_argument("--decode",    default="sample", choices=["sample", "beam"])
    p.add_argument("--k",         nargs="+", type=int, default=[10])
    p.add_argument("--eval_every", type=int,  default=10)
    p.add_argument("--seed",      type=int,   default=CFG.SEED)
    p.add_argument("--sd_repo",   default=DEFAULT_SD_REPO)
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--ckpt",      default=None)
    p.add_argument("--out_dir",   default=CFG.OUT_DIR)
    p.add_argument("--no_cache",  action="store_true")
    args = p.parse_args()

    CFG.GRID, CFG.SIGMA_PIX = args.grid, args.sigma_pix
    CFG.BATCH_SIZE, CFG.SEED, CFG.OUT_DIR = args.batch_size, args.seed, args.out_dir
    CFG.K_SAMPLES = max(args.k)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(CFG.SEED)
    np.random.seed(CFG.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CFG.SEED)
    os.makedirs(CFG.OUT_DIR, exist_ok=True)

    tag  = CFG.tag()
    ckpt = args.ckpt or f"{CFG.OUT_DIR}/sd_{tag}.pt"
    warmup = args.warmup or SD_WARMUP[CFG.DATASET]

    model, diffusion, T, steps = build_sd(args.sd_repo, CFG.DATASET, CFG.GRID,
                                          device)
    n_par = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)

    print(f"Device  : {device}")
    print(f"Task    : {CFG.DATASET}  g={CFG.GRID}  n={CFG.n()}  "
          f"σ_pix={CFG.SIGMA_PIX:g}")
    print(f"SD      : riffle, T={T}, denoising schedule {steps}, "
          f"generalized_PL reverse")
    print(f"Decode  : {args.decode}"
          + ("   [sampled reverse process]" if args.decode == "sample"
             else "   [BEAM SEARCH — mode-seeking, not a distributional decoder]"))
    print(f"Params  : {n_par:,}\n")

    train_loader, test_loader, dist_loader = build_loaders(
        batch_size=CFG.BATCH_SIZE, use_cache=not args.no_cache)

    history = None
    if args.skip_train:
        model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
        print(f"Loaded checkpoint {ckpt}\n")
    else:
        print("=" * 66)
        print("TRAINING  SymmetricDiffusers")
        print("=" * 66)
        history = train_sd(model, diffusion, train_loader, dist_loader, device,
                           epochs=args.epochs, lr=args.lr, warmup=warmup,
                           eval_every=args.eval_every, k_samples=max(args.k))
        torch.save({"model_state": model.state_dict(), "history": history,
                    "cfg_tag": tag, "T": T, "reverse_steps": steps}, ckpt)
        print(f"Checkpoint saved: {ckpt}\n")

    # ── evaluation ───────────────────────────────────────────────
    print("=" * 66)
    print(f"FINAL EVALUATION  (decode = {args.decode})")
    print("=" * 66)
    sampler = SDSampler(model, diffusion, args.decode).eval()

    full = evaluate(sampler, test_loader, device, single_sample_only=True)
    print(format_results(full, "full test split, single sample"))

    results = {"decode": args.decode, "full_test_K1": full, "by_k": {}}
    for k in args.k:
        r = evaluate(sampler, dist_loader, device, k_samples=k)
        results["by_k"][k] = r
        print()
        print(format_results(r, f"distributional prefix, K = {k}"))

    with open(f"{CFG.OUT_DIR}/sd_{tag}_{args.decode}.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Results : {CFG.OUT_DIR}/sd_{tag}_{args.decode}.json")


if __name__ == "__main__":
    main()
