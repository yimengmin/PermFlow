#!/bin/bash
# Section 5.1 (visual sorting): every run reported in Table 1 and Appendix G.
# Each line is independent; run them in parallel on a cluster if available.
set -e
cd "$(dirname "$0")"
for SEED in 42 0 1; do
  python -u train.py --coupling nearest    --seed $SEED      # PermFlow, nearest coupling
  python -u train.py --coupling calibrated --seed $SEED      # PermFlow, calibrated coupling
  python -u train.py --state_independent   --seed $SEED      # ablation: v = f(c)
  for T in a b random sample; do                             # GS with one valid order per example
    python -u gumbel_sinkhorn_baseline.py --target $T --seed $SEED
    python -u gumbel_sinkhorn_baseline.py --target $T --seed $SEED --skip_train --decoder gm --k 10 100
  done
done
python -u train.py --coupling random --seed 42               # ablation: random coupling
python -u train.py --no_project      --seed 42               # ablation: no tangent projection
for SEED in 42 0 1; do                                       # Euler-step sweep (Section 5.4)
  for C in nearest calibrated; do
    python -u eval_steps.py --ckpt outputs/flow_xt_${C}_seed${SEED}.pt
  done
done
