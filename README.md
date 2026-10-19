# Learning Unbiased Permutations via Flow Matching — code

One folder per experiment in the paper:

- `sorting_mnist/`: sorting MNIST digits when one image is a blend of two
  digits, so that two orders are valid with given weights.
- `jigsaw/`: reassembling MNIST jigsaw puzzles, with and without the pixel
  noise of the published benchmark.
- `mallows_sushi/`: learning ranking distributions, on conditional Mallows
  mixtures and on the Sushi preference data.

Each folder is self-contained. Its `run_all.sh` reproduces every run reported
for that experiment, including the baselines, the ablations and the sweep over
the number of Euler steps.

## Environment

```bash
bash setup_env.sh          # creates the conda env `permflow` and builds torch-linear-assignment
conda activate permflow
```

`setup_env.sh` builds the environment we used (Python 3.9, PyTorch 2.5.1 with
CUDA 12.4, torchvision, NumPy, SciPy) and compiles
[`torch-linear-assignment`](https://pypi.org/project/torch-linear-assignment/),
which does the Hungarian rounding on the GPU at inference; run it on a GPU
node. `environment_full.yml` is the complete export of that environment.

You probably do not need the full environment. The code only uses PyTorch
(with torchvision, to download MNIST), NumPy, SciPy and
`torch-linear-assignment`, so an existing PyTorch installation should work
after

```bash
pip install --no-build-isolation torch-linear-assignment
```

We have not tested this setup. Without `torch-linear-assignment` the code falls
back to SciPy's `linear_sum_assignment`, which gives the same result but is
slower.

MNIST is downloaded by torchvision on first use, and the Sushi data by
`mallows_sushi/download_sushi.sh`. The SymmetricDiffusers baseline needs its
released code, fetched by `jigsaw/setup_symmetric_diffusers.sh`.

## Running

```bash
cd sorting_mnist && python train.py --coupling calibrated --seed 42
cd jigsaw        && python train.py --grid 4 --sigma_pix 0 --seed 42 --k_eval 5 10 20 50 100
cd mallows_sushi && bash download_sushi.sh && python train.py --data synth --n 8 --M 2 --model permflow
```

A trained model can be evaluated again without retraining, for example with a
different number of Euler steps: `sorting_mnist/eval_steps.py --ckpt ...`,
`sorting_mnist/gumbel_sinkhorn_baseline.py --skip_train`,
`jigsaw/eval_steps.py`, and `mallows_sushi/train.py --eval_only` (the last two
take the same flags as training).

All methods within a task share the network, optimiser and data. Results in
the paper are means over seeds 42, 0 and 1, using the model after the last
epoch.
