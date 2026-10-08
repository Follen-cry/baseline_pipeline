#!/bin/bash
# Run the whole vae_cond experiment on ONE GPU of the current node: wiring check -> 5 datapoints x 3 conditions -> comparison.
# usage: bash run.sh [gpu_index]      (default 0)    ~15 generations, a few minutes on an A40
set -e
D=$(cd "$(dirname "$0")" && pwd)
cd "$D"
PY=$(python3 common.py internvlu_python)
export CUDA_VISIBLE_DEVICES=${1:-0}
CUDA_VISIBLE_DEVICES= $PY check_inputs.py      # CPU only; aborts on a wiring mismatch
$PY gen.py                                      # resumable: skips outputs that already exist
$PY make_grid.py
