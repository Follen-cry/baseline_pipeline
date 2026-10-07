#!/bin/bash
# Start one Qwen3-VL-30B-A3B-Instruct-FP8 vLLM server (weights, served name, vllm binary from ../config.json).
# usage (on the GPU node): ./serve_judge.sh <gpu> <port>      then  export EP=http://<node>:<port>/v1[,...]
# 3 images per prompt are required (PhyEditBench sends input, GT and prediction). First start compiles for a few minutes.
set -e
D=$(cd "$(dirname "$0")" && pwd)
W=$(python3 "$D/common.py" judge.weights); S=$(python3 "$D/common.py" judge.served_name)
V=$(python3 "$D/common.py" envs.vllm_bin)
CUDA_VISIBLE_DEVICES=$1 exec "$V" serve "$W" --served-model-name "$S" --host 0.0.0.0 --port "$2" \
  --max-model-len 20480 --max-num-seqs 32 --limit-mm-per-prompt '{"image": 3}' --gpu-memory-utilization 0.90 --seed 0
