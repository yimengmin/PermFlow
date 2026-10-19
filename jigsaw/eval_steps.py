"""
eval_steps.py — re-evaluate a trained PermFlow jigsaw checkpoint with a
different number of Euler steps at inference (no retraining).
Takes the same flags as train.py, so the checkpoint tag resolves the same way.

    python eval_steps.py --grid 4 --sigma_pix 0 --eval_steps 1 2 5 10 20 50
"""
import sys
import numpy as np
import torch

from config  import CFG
from data    import build_loaders
from model   import FlowModel
from metrics import evaluate
from train   import build_argparser, apply_args


def main():
    p = build_argparser()
    p.add_argument("--eval_steps", nargs="+", type=int, default=[1, 2, 5, 10, 20, 50])
    p.add_argument("--eval_k",     type=int, default=10)
    args = p.parse_args()
    apply_args(args)
    tag = (CFG.tag() + ("_label" if args.coupling == "label" else "")
              + ("_blockwise" if args.coupling == "blockwise" else "")
           + ("_noproj" if args.no_project else "")
           )
    ckpt = f"{CFG.OUT_DIR}/permflow_{tag}.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader, dist_loader = build_loaders(use_cache=not args.no_cache)
    model = FlowModel(n=CFG.n(), in_channels=CFG.in_channels(), patch_size=CFG.patch_size(),
                      d_model=CFG.D_MODEL, n_steps=CFG.N_STEPS, sigma_0=CFG.SIGMA_0,
                      coupling=args.coupling, project=not args.no_project).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    model.eval()

    print(f"Checkpoint : {ckpt}")
    K = args.eval_k
    hdr = (f"{'steps':>5}  {'Valid@1':>8}  {'Acc@1':>7}  {'Corr@1':>7}  {'Corr(c)@1':>9}  "
           f"{'tau@1':>7}  {'Valid@K':>8}  {'Dist@K':>7}  {'margTV@K':>8}   (K={K})")
    print(hdr); print("─" * len(hdr))
    for S in args.eval_steps:
        model.n_steps = S
        torch.manual_seed(42); np.random.seed(42)
        f = evaluate(model, test_loader, device, single_sample_only=True)
        torch.manual_seed(42); np.random.seed(42)
        d = evaluate(model, dist_loader, device, k_samples=K)
        print(f"{S:>5}  {f['validity']*100:>7.2f}%  {f['accuracy']*100:>6.2f}%  "
              f"{f['correct']*100:>6.2f}%  {f['correct_content']*100:>8.2f}%  "
              f"{f['kendall_tau']:>7.4f}  {d['validity']*100:>7.2f}%  "
              f"{d['distinct']*100:>6.2f}%  {d['marginal_tv']:>8.4f}", flush=True)


if __name__ == "__main__":
    main()
