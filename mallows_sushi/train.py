"""
train.py — train one model on one dataset and evaluate.

  python train.py --data synth --n 8 --M 2 --model permflow
  python train.py --data sushi --model gs
  python train.py --data synth --n 8 --M 2 --model permflow --eval_only   # evaluate saved .pt
"""
import argparse, json, os, time

import numpy as np
import torch

from data import MallowsMixture, Sushi
from metrics import SynthEvaluator, SushiEvaluator
from models import build_model


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="synth", choices=["synth", "sushi"])
    p.add_argument("--model", default="permflow", choices=["permflow", "gs", "pl", "plmix"])
    # synthetic
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--M", type=int, default=2)
    p.add_argument("--theta", type=float, default=1.0)
    p.add_argument("--no_reverse", action="store_true")
    p.add_argument("--data_seed", type=int, default=0)
    # model
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--nlayers", type=int, default=4)
    p.add_argument("--nheads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--sigma0", type=float, default=1.0)
    p.add_argument("--xt_mode", default="token", choices=["token", "rows"])
    p.add_argument("--gs_tau", type=float, default=1.0)
    p.add_argument("--gs_iters", type=int, default=20)
    p.add_argument("--plmix_M", type=int, default=4)
    # optimisation
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--bs", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    # evaluation
    p.add_argument("--K", type=int, default=None)
    p.add_argument("--n_ctx", type=int, default=200)
    p.add_argument("--eval_steps", nargs="+", type=int, default=[1, 2, 5, 10, 20, 50])
    p.add_argument("--eval_every", type=int, default=5000)
    p.add_argument("--out_dir", default="./outputs")
    p.add_argument("--eval_only", action="store_true",
                   help="load <out_dir>/<tag>.pt (same flags as training) and only evaluate")
    return p.parse_args()


def main():
    a = parse()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if a.data == "synth":
        ds = MallowsMixture(n=a.n, M=a.M, theta=a.theta, reverse=not a.no_reverse,
                            seed=a.data_seed)
        n = a.n
        ev = SynthEvaluator(ds, n_ctx=a.n_ctx, K=a.K or 1000)
        tag = f"synth_n{a.n}_M{a.M}{'' if not a.no_reverse else '_rand'}_th{a.theta:g}"
    else:
        ds = Sushi()
        n = ds.n
        ev = SushiEvaluator(ds, K=a.K or 100)
        tag = "sushi"
    name = a.model + ("_rows" if a.model == "permflow" and a.xt_mode == "rows" else "") \
           + (f"_s0{a.sigma0:g}" if a.model == "permflow" and a.sigma0 != 1.0 else "")
    tag = f"{tag}_{name}_seed{a.seed}"
    os.makedirs(a.out_dir, exist_ok=True)

    model = build_model(a.model, ds.d_in, n, a).to(dev)
    print(f"Run     : {tag}")
    print(f"Params  : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Args    : {vars(a)}\n")

    ref = ev.oracle() if a.data == "synth" else ev.data_floor()
    print(f"Reference ({'oracle, K exact samples' if a.data == 'synth' else 'train vs test'}): "
          + "  ".join(f"{k}={v:.4f}" for k, v in ref.items()) + "\n", flush=True)

    is_flow = a.model == "permflow"
    ckpt = f"{a.out_dir}/{tag}.pt"
    if a.eval_only:
        model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
        print(f"Loaded  : {ckpt}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=1e-5)
    t0, run = time.time(), 0.0
    for step in range(1, 0 if a.eval_only else a.steps + 1):
        model.train()
        f, o = ds.batch(a.bs, rng)
        loss = model.loss(f.to(dev), o.to(dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % a.eval_every == 0 or step == a.steps:
            model.eval()
            r = ev.evaluate(model, dev, **({"steps": 10} if is_flow else {}))
            print(f"step {step:>6}  loss {run:.5f}  "
                  + "  ".join(f"{k}={v:.4f}" for k, v in r.items())
                  + f"  [{time.time() - t0:.0f}s]", flush=True)

    # ── final evaluation ─────────────────────────────────────────
    model.eval()
    res = {"tag": tag, "args": vars(a), "reference": ref, "final": {}}
    print("\nFINAL")
    if is_flow:
        for S in a.eval_steps:
            r = ev.evaluate(model, dev, steps=S)
            res["final"][f"steps{S}"] = r
            print(f"  steps={S:<3d} " + "  ".join(f"{k}={v:.4f}" for k, v in r.items()), flush=True)
    else:
        r = ev.evaluate(model, dev)
        res["final"]["default"] = r
        print("  " + "  ".join(f"{k}={v:.4f}" for k, v in r.items()))
        if a.model == "gs":
            r = ev.evaluate(model, dev, noise=False)
            res["final"]["no_noise"] = r
            print("  no-noise " + "  ".join(f"{k}={v:.4f}" for k, v in r.items()))
    if not a.eval_only:
        torch.save(model.state_dict(), ckpt)
    out = f"{a.out_dir}/{tag}{'_eval' if a.eval_only else ''}.json"
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
