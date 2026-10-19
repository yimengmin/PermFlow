"""Re-evaluate trained GS checkpoints with both decoders:
  gm   : Hungarian on S+G            (Gumbel-Matching; used in the original runs)
  soft : Hungarian on Sinkhorn((S+G)/tau)
with and without noise.  Writes outputs/<tag>_decoders.json."""
import argparse, json, sys
import numpy as np, torch
sys.argv = [sys.argv[0]] + sys.argv[1:]
from data import MallowsMixture, Sushi
from metrics import SynthEvaluator, SushiEvaluator
from models import build_model
import train as T

def main():
    a = T.parse()
    a.model = "gs"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.data == "synth":
        ds = MallowsMixture(n=a.n, M=a.M, theta=a.theta, reverse=not a.no_reverse, seed=a.data_seed)
        ev = SynthEvaluator(ds, n_ctx=a.n_ctx, K=a.K or 1000); n = a.n
        tag = f"synth_n{a.n}_M{a.M}{'' if not a.no_reverse else '_rand'}_th{a.theta:g}_gs_seed{a.seed}"
    else:
        ds = Sushi(); ev = SushiEvaluator(ds, K=a.K or 100); n = ds.n
        tag = f"sushi_gs_seed{a.seed}"
    m = build_model("gs", ds.d_in, n, a).to(dev)
    m.load_state_dict(torch.load(f"{a.out_dir}/{tag}.pt", map_location=dev)); m.eval()
    res = {}
    for dec in ("gm", "soft"):
        for noise in (True, False):
            torch.manual_seed(a.seed)
            r = ev.evaluate(m, dev, noise=noise, decoder=dec)
            k = f"{dec}{'' if noise else '_nonoise'}"; res[k] = r
            print(f"{tag} {k:13s} " + "  ".join(f"{kk}={v:.4f}" for kk, v in r.items()), flush=True)
    json.dump(res, open(f"{a.out_dir}/{tag}_decoders.json", "w"), indent=2)

if __name__ == "__main__":
    main()
