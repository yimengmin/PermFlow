"""
eval_steps.py — re-evaluate a trained X_t-conditioned checkpoint with a
different number of Euler steps at inference (no retraining).

    python eval_steps.py --ckpt outputs/flow_xt_nearest_seed42.pt
"""
import argparse, json, os, re
import numpy as np
import torch

from config  import CFG
from data    import build_loaders
from model   import FlowModel
from metrics import full_eval


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--steps", nargs="+", type=int, default=[1, 2, 5, 10, 20, 50, 100])
    p.add_argument("--k", nargs="+", type=int, default=[10, 100])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    name = os.path.basename(args.ckpt)
    m = re.match(r"flow_(xt|stateindep)_(nearest|calibrated|random)(_noproj)?_seed\d+\.pt", name)
    assert m, name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlowModel(n=CFG.N, d_enc=CFG.D_ENC, n_steps=CFG.N_STEPS, sigma_0=CFG.SIGMA_0,
                      use_xt=(m[1] == "xt"), coupling=m[2], project=(m[3] is None)).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model_state"])
    model.eval()
    _, _, amb_loader, cln_loader = build_loaders(42)

    print(f"Checkpoint : {args.ckpt}")
    print(f"{'steps':>5} {'K':>4}  {'CleanAcc':>8}  {'Coverage':>8}  {'AnyCor':>7}  {'CalErr':>7}")
    out = {}
    for S in args.steps:
        model.n_steps = S
        for k in args.k:
            torch.manual_seed(args.seed); np.random.seed(args.seed)
            r = full_eval(model, cln_loader, amb_loader, device, k_samples=k)
            out[f"{S}_{k}"] = r
            print(f"{S:>5} {k:>4}  {r['clean_acc']*100:>7.1f}%  {r['coverage']*100:>7.1f}%  "
                  f"{r['any_correct']*100:>6.1f}%  {r['calibration']:>7.4f}", flush=True)
    with open(args.ckpt.replace(".pt", "_steps.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)


if __name__ == "__main__":
    main()
