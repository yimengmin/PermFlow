#!/bin/bash
# Section 5.2 (jigsaw puzzles): every run reported in Tables 2-3 and Appendix H.
set -e
cd "$(dirname "$0")"
KE="--k_eval 5 10 20 50 100"
for SIG in 0.01 0; do
  for G in 4 5 6; do
    for SEED in 42 0 1; do
      python -u train.py --grid $G --sigma_pix $SIG --seed $SEED $KE
    done
  done
done
python -u train.py --grid 4 --sigma_pix 0 --state_independent $KE   # ablations (Table 3)
python -u train.py --grid 4 --sigma_pix 0 --coupling label     $KE
python -u train.py --grid 4 --sigma_pix 0 --no_project         $KE
for SIG in 0.01 0; do for G in 4 5 6; do                            # Euler-step sweep (seed 42)
  python -u eval_steps.py --grid $G --sigma_pix $SIG --eval_steps 1 2 5 10 20 50
done; done
python -u check_zero_patches.py --grids 4 5 6 --sigmas 0 1e-4 1e-3 3e-3 1e-2   # orbit statistics
# SymmetricDiffusers on our data (released code): beam search at sigma=0.01, sampling at sigma=0
bash setup_symmetric_diffusers.sh
for G in 4 5; do
  python -u symmetric_diffusers_baseline.py --grid $G --sigma_pix 0.01 --epochs 120 --decode beam --k 1
done
python -u symmetric_diffusers_baseline.py --grid 4 --sigma_pix 0 --epochs 120 --decode sample --k 5 10 20
