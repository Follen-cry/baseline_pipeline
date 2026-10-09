#!/bin/bash
# Cropping/misalignment check: regenerate with output size == the VAE-branch condition size (688x400) and native size (832x480).
# The default run asks for 672x384 while the gen processor feeds the VAE a 688x400 condition (25x43 vs 24x42 patch grid).
# usage: bash run_size_test.sh [gpu_index]
set -e
D=$(cd "$(dirname "$0")" && pwd)
cd "$D"
PY=$(python3 common.py internvlu_python)
export CUDA_VISIBLE_DEVICES=${1:-0}
$PY gen.py --size 688x400 --tag _688x400
$PY gen.py --size 832x480 --tag _832x480
