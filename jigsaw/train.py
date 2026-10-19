"""
train.py -- PermFlow on the jigsaw benchmark (Section 5.2).

    python -u train.py --grid 4 --sigma_pix 0.01 --seed 42 --k_eval 5 10 20 50 100   # published setting
    python -u train.py --grid 4 --sigma_pix 0    --seed 42 --k_eval 5 10 20 50 100   # orbit arm
    python -u train.py --grid 4 --sigma_pix 0 --state_independent                    # ablation
    python -u train.py --grid 4 --sigma_pix 0 --coupling label                       # ablation
    python -u train.py --grid 4 --sigma_pix 0 --no_project                           # ablation

The final-epoch model is evaluated (no checkpoint selection); results are
written to <out_dir>/permflow_<tag>.json.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json

import numpy as np
import torch

from config import CFG
from data import build_loaders
from metrics import evaluate, format_results
from model import FlowModel, StateIndependentFlowModel
from trainer import train


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PermFlow on jigsaw puzzles.")
    p.add_argument("--grid",      type=int,   default=CFG.GRID,
                   help="g — pieces per side; n = g² patches.")
    p.add_argument("--sigma_pix", type=float, default=CFG.SIGMA_PIX,
                   help="data-space pixel noise σ. 0 = ambiguous arm, "
                        "0.01 = their NoisyMNIST control.")
    p.add_argument("--sigma_0",   type=float, default=CFG.SIGMA_0,
                   help="flow-space noise on X₀.")
    p.add_argument("--steps",     type=int,   default=CFG.N_STEPS)
    p.add_argument("--k",         type=int,   default=CFG.K_SAMPLES)
    p.add_argument("--epochs",    type=int,   default=CFG.N_EPOCHS)
    p.add_argument("--lr",        type=float, default=CFG.LR)
    p.add_argument("--batch_size", type=int,  default=CFG.BATCH_SIZE)
    p.add_argument("--seed",      type=int,   default=CFG.SEED)
    p.add_argument("--coupling",  default="penalised",
                   choices=["penalised", "blockwise", "label"],
                   help="label = ablation: single sampled τ, no orbit coupling.")
    p.add_argument("--no_project", action="store_true",
                   help="ablation: use raw f_θ as velocity, no tangent projection.")
    p.add_argument("--state_independent", action="store_true",
                   help="ablation: drop the X_t state token.")
    p.add_argument("--k_eval",    nargs="+", type=int, default=None,
                   help="extra K values for the final table, e.g. 5 10 20 50 100")
    p.add_argument("--out_dir",   default=CFG.OUT_DIR)
    p.add_argument("--no_cache",  action="store_true")
    p.add_argument("--compile",   action="store_true")
    return p


def apply_args(args) -> None:
    CFG.GRID       = args.grid
    CFG.SIGMA_PIX  = args.sigma_pix
    CFG.SIGMA_0    = args.sigma_0
    CFG.N_STEPS    = args.steps
    CFG.K_SAMPLES  = args.k
    CFG.N_EPOCHS   = args.epochs
    CFG.LR         = args.lr
    CFG.BATCH_SIZE = args.batch_size
    CFG.SEED       = args.seed
    CFG.OUT_DIR    = args.out_dir


@torch.no_grad()
def constraint_violation(model, loader, device, max_batches: int = 8) -> dict:
    """Row/column-sum residual of the continuous endpoints X_1 (K = 1)."""
    model.eval()
    mx, tot, cnt = 0.0, 0.0, 0
    for b, (patches, *_rest) in enumerate(loader):
        if b >= max_batches:
            break
        X = model.sample(patches.to(device), k_samples=1)[0].double()
        r = torch.cat([(X.sum(-1) - 1).abs(), (X.sum(-2) - 1).abs()], dim=-1)
        mx = max(mx, r.max().item()); tot += r.sum().item(); cnt += r.numel()
    return {"max": mx, "mean": tot / max(cnt, 1)}


def main() -> None:
    args = build_argparser().parse_args()
    apply_args(args)

    torch.manual_seed(CFG.SEED)
    np.random.seed(CFG.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CFG.SEED)
    os.makedirs(CFG.OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag    = (CFG.tag() + ("_noXt" if args.state_independent else "")
              + ("_label" if args.coupling == "label" else "")
              + ("_blockwise" if args.coupling == "blockwise" else "")
              + ("_noproj" if args.no_project else ""))

    print(f"Device  : {device}")
    print(f"Task    : {CFG.DATASET}  g={CFG.GRID}  n={CFG.n()}  "
          f"patch={CFG.patch_size()}×{CFG.patch_size()}  σ_pix={CFG.SIGMA_PIX:g}")
    print(f"Flow    : σ₀={CFG.SIGMA_0}  steps={CFG.N_STEPS}  K={CFG.K_SAMPLES}  "
          f"coupling={args.coupling}"
          + ("  [X_t-conditioned]" if not args.state_independent
             else "  [STATE-INDEPENDENT ABLATION]"))
    print(f"Optim   : AdamW lr={CFG.LR} wd={CFG.WEIGHT_DECAY} "
          f"cosine→{CFG.LR_MIN} epochs={CFG.N_EPOCHS} bs={CFG.BATCH_SIZE}\n")

    # ── Data ─────────────────────────────────────────────────────
    train_loader, test_loader, dist_loader = build_loaders(
        use_cache=not args.no_cache)

    # ── Model ────────────────────────────────────────────────────
    Model = StateIndependentFlowModel if args.state_independent else FlowModel
    model = Model(
        n           = CFG.n(),
        in_channels = CFG.in_channels(),
        patch_size  = CFG.patch_size(),
        d_model     = CFG.D_MODEL,
        n_steps     = CFG.N_STEPS,
        sigma_0     = CFG.SIGMA_0,
        coupling    = args.coupling,
        project     = not args.no_project,
    ).to(device)

    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters : {n_par:,}")
    print(f"  Encoder  : {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"  Velocity : {sum(p.numel() for p in model.velocity.parameters()):,}\n")

    if args.compile:
        try:
            model = torch.compile(model)
            print("torch.compile: enabled\n")
        except Exception as e:                          # pragma: no cover
            print(f"torch.compile: skipped ({e})\n")

    # ── Train ────────────────────────────────────────────────────
    print("=" * 66)
    print("TRAINING")
    print("=" * 66)
    history = train(model, train_loader, dist_loader, device,
                    epochs=CFG.N_EPOCHS, k_samples=CFG.K_SAMPLES)

    raw = getattr(model, "_orig_mod", model)

    # ── Final evaluation ─────────────────────────────────────────
    full = evaluate(raw, test_loader, device, single_sample_only=True)
    print(format_results(full, "FINAL — full test split, single sample"))

    feas = constraint_violation(raw, test_loader, device)
    print(f"  Endpoint row/col-sum violation  max|X1 1 - 1| = {feas['max']:.3e}"
          f"   mean = {feas['mean']:.3e}")

    results = {"full_test_K1": full, "by_k": {}, "feasibility": feas,
               "args": vars(args)}
    k_list  = args.k_eval or [CFG.K_SAMPLES]
    for k in k_list:
        r = evaluate(raw, dist_loader, device, k_samples=k)
        results["by_k"][k] = r
        print()
        print(format_results(r, f"FINAL — distributional prefix, K = {k}"))

    if len(k_list) > 1:
        hdr = (f"\n{'K':>5}  {'Valid%':>8}  {'Dist@K%':>9}  {'margTV':>8}  "
               f"{'Corr%(c)':>9}  {'Corr%':>7}  {'Acc%':>7}  "
               f"{'RMSE':>9}  {'MAE':>9}")
        print(hdr)
        print("─" * len(hdr))
        for k in k_list:
            r = results["by_k"][k]
            print(f"{k:>5}  {r['validity']*100:>7.2f}%  "
                  f"{r['distinct']*100:>8.2f}%  {r['marginal_tv']:>8.4f}  "
                  f"{r['correct_content']*100:>8.2f}%  "
                  f"{r['correct']*100:>6.2f}%  {r['accuracy']*100:>6.2f}%  "
                  f"{r['rmse']:>9.6f}  {r['mae']:>9.6f}")
        print("─" * len(hdr))

    # ── Save ─────────────────────────────────────────────────────
    ckpt = f"{CFG.OUT_DIR}/permflow_{tag}.pt"
    torch.save({"model_state": raw.state_dict(),
                "history": history, "results": results,
                "cfg": {k: v for k, v in vars(CFG).items()
                        if not k.startswith("_") and not callable(v)
                        and not isinstance(v, classmethod)}},
               ckpt)
    with open(f"{CFG.OUT_DIR}/permflow_{tag}.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Checkpoint : {ckpt}")
    print(f"  Results    : {CFG.OUT_DIR}/permflow_{tag}.json")


if __name__ == "__main__":
    main()
