"""
train.py -- PermFlow on visual sorting with two weighted modes (Section 5.1).

    python train.py --coupling nearest      --seed 42   # PermFlow, nearest coupling
    python train.py --coupling calibrated   --seed 42   # PermFlow, calibrated coupling
    python train.py --state_independent     --seed 42   # ablation: v = f(c)
    python train.py --coupling random       --seed 42   # ablation: random coupling
    python train.py --no_project            --seed 42   # ablation: no tangent projection

PermFlow uses the straight path X_t = (1-t) X~0 + t P*, regresses the velocity
onto P* - X~0, and conditions the network on (X_t, t).  The state-independent
ablation has no state or time input and regresses onto the mean direction
P* - X0.  Results (all K) are written to outputs/flow_<tag>.json.
"""

import os
import json
import argparse
import torch
import numpy as np

from config  import CFG
from data    import build_loaders
from model   import FlowModel
from trainer import train
from metrics import full_eval


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=CFG.SEED,
                   help="training seed; data are always built with seed 42")
    p.add_argument("--coupling", default="nearest",
                   choices=["nearest", "calibrated", "random"])
    p.add_argument("--state_independent", action="store_true",
                   help="ablation: velocity v = f(c), no state or time input")
    p.add_argument("--no_project", action="store_true",
                   help="ablation: no tangent projection of the velocity")
    p.add_argument("--epochs", type=int, default=CFG.N_EPOCHS)
    p.add_argument("--k_eval", nargs="+", type=int,
                   default=[5, 10, 20, 40, 60, 80, 100])
    p.add_argument("--no_compile", action="store_true")
    return p.parse_args()


@torch.no_grad()
def constraint_violation(model, loader, device, max_batches=8):
    """Row/column-sum residual of the continuous endpoints."""
    model.eval()
    mx, tot, cnt = 0.0, 0.0, 0
    for b, (images, *_r) in enumerate(loader):
        if b >= max_batches:
            break
        X = model.sample(images.to(device), k_samples=1)[0].double()
        r = torch.cat([(X.sum(-1) - 1).abs(), (X.sum(-2) - 1).abs()], -1)
        mx = max(mx, r.max().item()); tot += r.sum().item(); cnt += r.numel()
    return {"max": mx, "mean": tot / max(cnt, 1)}


def main():
    args = parse_args()
    CFG.SEED, CFG.N_EPOCHS = args.seed, args.epochs
    use_xt = not args.state_independent
    tag = ("xt" if use_xt else "stateindep") + f"_{args.coupling}" \
          + ("_noproj" if args.no_project else "") + f"_seed{args.seed}"
    CFG.CKPT_PATH = f"{CFG.OUT_DIR}/flow_{tag}.pt"

    torch.manual_seed(CFG.SEED)
    np.random.seed(CFG.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CFG.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CFG.OUT_DIR, exist_ok=True)
    print(f"Device : {device}")
    print(f"Run    : {tag}  (N={CFG.N}, sigma0={CFG.SIGMA_0}, epochs={CFG.N_EPOCHS})\n")

    train_loader, val_loader, amb_loader, cln_loader = build_loaders(42)

    model = FlowModel(
        n        = CFG.N,
        d_enc    = CFG.D_ENC,
        n_steps  = CFG.N_STEPS,
        sigma_0  = CFG.SIGMA_0,
        use_xt   = use_xt,
        coupling = args.coupling,
        project  = not args.no_project,
    ).to(device)
    print(f"Parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    if hasattr(torch, "compile") and not args.no_compile:
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile: skipped ({e})")

    history = train(model, train_loader, cln_loader, amb_loader, device)

    raw_model = getattr(model, "_orig_mod", model)
    torch.save({"model_state": raw_model.state_dict(), "history": history,
                "cfg": {k: v for k, v in vars(CFG).items() if not k.startswith("_")}},
               CFG.CKPT_PATH)
    print(f"\nCheckpoint saved: {CFG.CKPT_PATH}")

    sweep = {}
    print(f"\n{'K':>5}  {'CleanAcc':>9}  {'Coverage':>9}  {'AnyCor':>8}  {'CalErr':>8}")
    for k in args.k_eval:
        r = full_eval(raw_model, cln_loader, amb_loader, device, k_samples=k)
        sweep[k] = r
        print(f"{k:>5}  {r['clean_acc']*100:>8.1f}%  {r['coverage']*100:>8.1f}%  "
              f"{r['any_correct']*100:>7.1f}%  {r['calibration']:>8.4f}")
    feas = constraint_violation(raw_model, cln_loader, device)
    print(f"Endpoint row/col-sum violation: max {feas['max']:.3e}  mean {feas['mean']:.3e}")
    with open(CFG.CKPT_PATH.replace(".pt", ".json"), "w") as f:
        json.dump({"tag": tag, "args": vars(args),
                   "n_params": sum(p.numel() for p in raw_model.parameters()),
                   "sweep": sweep, "feasibility": feas}, f, indent=2)


if __name__ == "__main__":
    main()
