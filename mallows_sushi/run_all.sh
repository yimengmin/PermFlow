#!/bin/bash
# Section 5.3 (Mallows mixtures and Sushi): every run reported in Tables 4-5 and Appendix I.
set -e
cd "$(dirname "$0")"
[ -d data/sushi3-2016 ] || bash download_sushi.sh
DATA=("--data synth --n 6 --M 2" "--data synth --n 8 --M 2" "--data synth --n 8 --M 3 --no_reverse"
      "--data synth --n 10 --M 2" "--data sushi --steps 5000 --bs 256 --eval_every 1000")
MODELS=("--model permflow" "--model permflow --xt_mode rows" "--model gs" "--model pl" "--model plmix")
for SEED in 42 0 1; do
  for D in "${DATA[@]}"; do
    for M in "${MODELS[@]}"; do
      python -u train.py $D $M --seed $SEED
    done
  done
done
# GS with both decoders (Appendix I)
for SEED in 42 0 1; do
  for D in "--data synth --n 6 --M 2" "--data synth --n 8 --M 2" "--data synth --n 8 --M 3 --no_reverse" \
           "--data synth --n 10 --M 2" "--data sushi"; do
    python -u eval_gs_decoders.py $D --seed $SEED
  done
done
