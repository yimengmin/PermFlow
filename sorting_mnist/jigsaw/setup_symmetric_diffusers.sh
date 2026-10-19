#!/bin/bash
# Fetch the released SymmetricDiffusers implementation (ICLR 2025) so the
# head-to-head baseline can import ReverseDiffusion / DiffusionUtils
# unmodified.  Only their model + diffusion code is used; their data
# pipeline, DDP training script and beam-search evaluation are not.
#
#   bash setup_symmetric_diffusers.sh
#
# The checkout is git-ignored — it is a third-party dependency, not part of
# this repository.
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/third_party"
mkdir -p "${DIR}"

if [ -d "${DIR}/SymmetricDiffusers/.git" ]; then
    echo "Already present: ${DIR}/SymmetricDiffusers"
    exit 0
fi

git clone --depth 1 https://github.com/DSL-Lab/SymmetricDiffusers \
    "${DIR}/SymmetricDiffusers"

echo
echo "Done. ${DIR}/SymmetricDiffusers"
echo "The baseline needs only torch from their requirements; their"
echo "lr_schedulers.py (which imports 'transformers') is not used —"
echo "symmetric_diffusers_baseline.py reimplements the Noam schedule."
