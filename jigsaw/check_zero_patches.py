"""
check_zero_patches.py  —  how ambiguous is each grid size, really?
══════════════════════════════════════════════════════════════════

The pre-flight check the experiment design calls for: count exactly-zero
patches per image at each g.  If the count commonly exceeds ~10, then
|𝒪| ≥ 10! and "distinct valid modes @ K" is uninformative at K = 100 — in
that regime the headline table should use a smaller g and the larger grids
should be reported on validity rate and per-patch marginal TV only.

It also reports the full exact-content partition (not just the all-zero
class), since two identical *non*-zero patches count exactly the same way,
and the σ sweep, so you can see |𝒪| collapse to 1 as the noise rises.

Usage
-----
    python -u check_zero_patches.py                       # MNIST, g = 4..6
    python -u check_zero_patches.py --sigmas 0 1e-4 1e-3 1e-2
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import math

import numpy as np
import torch

from config import CFG
from data import add_pixel_noise, chunk_images, load_images
from orbit import orbit_log_size, partition_by_content


def analyse(images: torch.Tensor, g: int, sigma: float, rng) -> dict:
    images  = add_pixel_noise(images, sigma, rng)
    patches = chunk_images(images, g).numpy()          # [N, n, C, p, p]
    N, n = patches.shape[0], patches.shape[1]

    zeros   = np.zeros(N, dtype=np.int64)
    dupes   = np.zeros(N, dtype=np.int64)      # patches in a class of size >1
    biggest = np.zeros(N, dtype=np.int64)      # largest class size m_max
    log_orb = np.zeros(N, dtype=np.float64)

    flat = patches.reshape(N, n, -1)
    zero_mask = (flat == 0).all(-1)                    # exact zeros
    zeros = zero_mask.sum(-1)

    for k in range(N):
        blk = partition_by_content(patches[k])
        _, counts = np.unique(blk, return_counts=True)
        biggest[k] = counts.max()
        dupes[k]   = counts[counts > 1].sum()
        log_orb[k] = orbit_log_size(blk)

    return dict(
        n            = n,
        zeros_mean   = zeros.mean(),
        zeros_med    = float(np.median(zeros)),
        zeros_p90    = float(np.percentile(zeros, 90)),
        zeros_max    = int(zeros.max()),
        frac_gt10    = float((zeros > 10).mean()),
        m_max_mean   = biggest.mean(),
        m_max_max    = int(biggest.max()),
        dupes_mean   = dupes.mean(),
        frac_ambig   = float((log_orb > 0).mean()),
        log_orb_mean = log_orb.mean(),
        log_orb_med  = float(np.median(log_orb)),
        orbit_med    = math.exp(float(np.median(log_orb))),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--grids",   nargs="+", type=int, default=None)
    p.add_argument("--sigmas",  nargs="+", type=float, default=[0.0, 0.01])
    p.add_argument("--split",   default="test", choices=["train", "test"])
    p.add_argument("--n_images", type=int, default=10_000)
    p.add_argument("--seed",    type=int, default=CFG.SEED)
    p.add_argument("--no_quantise", action="store_true",
                   help="keep the noised image in float32 — shows the "
                        "degenerate step-at-0⁺ sweep.")
    args = p.parse_args()
    if args.no_quantise:
        CFG.QUANTISE = False
    print(f"Pixel domain : {'8-bit (re-quantised)' if CFG.QUANTISE else 'float32'}")

    grids = args.grids or [4, 5, 6]

    images = load_images("mnist", train=(args.split == "train"))
    if args.n_images > 0:
        images = images[: args.n_images]
    print(f"Dataset : mnist  split={args.split}  "
          f"images={images.shape[0]:,}  size={tuple(images.shape[1:])}")
    print()

    hdr = (f"{'σ':>7}  {'g':>2}  {'n':>3}  {'patch':>6}  "
           f"{'zeros(mean)':>11}  {'med':>4}  {'p90':>4}  {'max':>4}  "
           f"{'P(zeros>10)':>11}  {'m_max':>6}  {'ambig%':>7}  "
           f"{'med |O|':>12}")
    print(hdr)
    print("─" * len(hdr))

    for sigma in args.sigmas:
        rng = np.random.default_rng(args.seed)
        for g in grids:
            r = analyse(images, g, sigma, rng)
            ps = images.shape[-1] // g
            print(f"{sigma:>7g}  {g:>2d}  {r['n']:>3d}  {ps:>3d}×{ps:<2d}  "
                  f"{r['zeros_mean']:>11.2f}  {r['zeros_med']:>4.0f}  "
                  f"{r['zeros_p90']:>4.0f}  {r['zeros_max']:>4d}  "
                  f"{r['frac_gt10']*100:>10.1f}%  {r['m_max_mean']:>6.2f}  "
                  f"{r['frac_ambig']*100:>6.1f}%  {r['orbit_med']:>12.4g}")
        print("─" * len(hdr))

    print()
    print("Reading:")
    print("  zeros       — exactly-zero (all-black) patches per image")
    print("  m_max       — size of the largest exact-content class (mean)")
    print("  ambig%      — images with |O| > 1")
    print("  med |O|     — median orbit size = median of ∏_b m_b!")
    print()
    print("If P(zeros>10) is large at some g, |O| ≥ 10! there and")
    print("distinct-modes@K is uninformative at K = 100: put the smaller g in")
    print("the headline table and report that g on validity rate and")
    print("per-patch marginal TV only.")


if __name__ == "__main__":
    main()
