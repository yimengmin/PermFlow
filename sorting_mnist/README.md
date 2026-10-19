# Learning Unbiased Permutations via Flow Matching — code

Code for the three experiments of the paper. Each folder is self-contained
and has a `run_all.sh` that reproduces every run reported for that experiment.

| Folder | Paper section | Contents |
|---|---|---|
| `sorting_mnist/` | 5.1, Appendix G | PermFlow (nearest / calibrated coupling; ablations: state-independent, random coupling, no projection); Gumbel-Sinkhorn trained on one valid order per example (only `P_a`, only `P_b`, random, `P_a` w.p. α), soft and Gumbel-Matching decoders; Euler-step sweep |
| `jigsaw/` | 5.2, Appendix H | PermFlow on the jigsaw benchmark (σ = 0.01 and σ = 0), orbit coupling; ablations (state-independent, single-label coupling, no projection); Euler-step sweep; orbit statistics; adapter for the released SymmetricDiffusers code |
| `mallows_sushi/` | 5.3, Appendix I | Conditional Mallows mixtures (exact ground truth for n ≤ 8) and Sushi rankings; PermFlow (state token / rows), Gumbel-Sinkhorn, Plackett–Luce, PL mixture; GS decoder comparison |

The cross-task Euler-step analysis of Section 5.4 uses the step sweeps produced
in each folder (`eval_steps.py` in `sorting_mnist/` and `jigsaw/`; the
`--eval_steps` option of `mallows_sushi/train.py`).

Trained checkpoints can be re-evaluated without retraining:
`sorting_mnist/eval_steps.py --ckpt ...` (PermFlow) and
`gumbel_sinkhorn_baseline.py --skip_train` (GS); `jigsaw/eval_steps.py` (same
flags as `train.py`); `mallows_sushi/train.py --eval_only` (same flags as
training; writes `<tag>_eval.json`) and `mallows_sushi/eval_gs_decoders.py`.

## Environment

```bash
bash setup_env.sh          # creates the conda env `permflow` and builds torch-linear-assignment
conda activate permflow
```

`environment.yml` pins Python 3.9.21, PyTorch 2.5.1 (CUDA 12.4), torchvision
0.20.1, NumPy 2.0.2, SciPy 1.13.1 and the CUDA 12.4 toolchain needed to compile
`torch-linear-assignment==0.0.5`, which performs the batched Hungarian rounding
on the GPU at inference (sorting and jigsaw). `setup_env.sh` installs it with
`--no-build-isolation` and checks it against SciPy; run it on a GPU node. If
the package is unavailable, the code falls back to the exact (slower)
`scipy.optimize.linear_sum_assignment`; the Mallows/Sushi task (n <= 10) always
uses SciPy. `environment_full.yml` is the complete export of the environment
used for the paper (same name, `permflow`), for exact replication. MNIST is
downloaded by torchvision on first use; the Sushi data are downloaded by
`mallows_sushi/download_sushi.sh`. The SymmetricDiffusers baseline needs its
released code, fetched by `jigsaw/setup_symmetric_diffusers.sh` (the parts we
import depend only on PyTorch).

## Common settings

PermFlow uses the straight path `X_t = (1-t) X~0 + t P*`, regresses the
tangent-projected velocity onto `P* - X~0`, conditions the network on
`(X_t, t)` through one extra state token, starts from
`X~0 = J/n + sigma0 * C(G)/||C(G)||_F` with `sigma0 = 1`, and integrates with
`S = 10` Euler steps followed by Hungarian rounding unless stated otherwise.
All methods within a task share the encoder, transformer, optimiser (AdamW,
lr 3e-4, weight decay 1e-4, cosine decay to 1e-5) and data, and the
final-epoch model is reported (no checkpoint selection). Results in the paper
are means over seeds 42, 0 and 1; the dataset is fixed across seeds.

## Quick start

```bash
cd sorting_mnist && python train.py --coupling calibrated --seed 42
cd jigsaw        && python train.py --grid 4 --sigma_pix 0 --seed 42 --k_eval 5 10 20 50 100
cd mallows_sushi && bash download_sushi.sh && python train.py --data synth --n 8 --M 2 --model permflow
```
