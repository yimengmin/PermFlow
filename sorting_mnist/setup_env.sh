#!/bin/bash
# Create the `permflow` environment and build torch-linear-assignment 0.0.5,
# the batched GPU Hungarian solver used to round samples at inference.
# Run on a machine with an NVIDIA GPU: the CUDA kernel is compiled only when
# torch.cuda.is_available() at build time.
set -e
cd "$(dirname "$0")"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env create -f environment.yml
conda activate permflow
export CUDA_HOME="$CONDA_PREFIX"
# --no-build-isolation: compile against the installed PyTorch 2.5.1, not a fresh one
pip install --no-build-isolation --no-cache-dir torch-linear-assignment==0.0.5
python - <<'PY'
import torch
from torch_linear_assignment import batch_linear_assignment
from scipy.optimize import linear_sum_assignment
C = torch.randn(64, 9, 9, device="cuda")
a = batch_linear_assignment(-C).cpu()
b = torch.stack([torch.as_tensor(linear_sum_assignment(-C[i].cpu().numpy())[1]) for i in range(64)])
assert torch.equal(a, b), "GPU and SciPy assignments differ"
print("torch-linear-assignment OK (CUDA kernel matches SciPy)")
PY
