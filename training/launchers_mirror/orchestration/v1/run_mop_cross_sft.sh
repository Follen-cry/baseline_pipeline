#!/bin/bash
# multi_object_placement_Composed_intermediate_cross: 3000-row SFT, one model
# per GPU, run in parallel. Usage: bash run_mop_cross_sft.sh base:0 s2:1 s4:3
set -uo pipefail
REPO_ROOT="/scratch/network/ssd2/junlin/ssl_mllm/Model_Related/InternVLU/InternVL/internvl_chat"
SFT_SCRIPT="shell/internvlu/sft/run_target_pred_mop_sft.sh"
META="/scratch/network/ssd2/junlin/ssl_mllm/baseline_pipeline/data/v1/meta/multi_object_placement_Composed_intermediate_cross_train_meta.json"
OUT_ROOT="/scratch/network/ssd/junlin/models"
LOG_ROOT="/scratch/network/ssd2/junlin/ssl_mllm/results/mop_cross_sft_logs"
mkdir -p "$LOG_ROOT"
declare -A CKPTS=(
  [base]=""
  [s2]="/scratch/network/ssd2/junlin/models/internvlu-s2-gen-merged"
  [s4]="/scratch/network/ssd2/junlin/models/internvlu-s4-gen-merged"
)
cd "$REPO_ROOT"
for job in "$@"; do
  name="${job%%:*}"; gpu="${job##*:}"; ckpt="${CKPTS[$name]}"
  (
    export MASTER_PORT=$((34300 + gpu)) CUDA_VISIBLE_DEVICES="$gpu" GPUS=1 GEN_IMAGE_SIZE=1024 META_PATH="$META" OUTPUT_DIR="$OUT_ROOT/internvlu-${name}-mopcross3000sft"
    [ -n "$ckpt" ] && export INTERNVLU_CKPT="$ckpt"
    bash "$SFT_SCRIPT" > "$LOG_ROOT/${name}.log" 2>&1
    echo "[queue] $name exit=$?"
  ) &
done
wait
