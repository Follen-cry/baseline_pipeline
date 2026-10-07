#!/bin/bash
# SFT sweep for 3 newly-adopted composed-intermediate-grid tasks
# (multi_object_placement, rotation_puzzle, shape_color_then_move), each
# 500 train rows, same pipeline as 2d_geo_trans_Composed_intermediate (see
# that dataset's README.md) -- composed single-image input, real-GT
# complete-grid output, VAE-conditioned on the single input image.
#
# Unlike the first (2d_geo) sweep, which trained 6 models, this one only
# trains 3 per task (base, S2, S4) per explicit instruction -- 9 runs total.
# Reuses run_target_pred_mop_sft.sh UNCHANGED, only META_PATH/INTERNVLU_CKPT/
# OUTPUT_DIR overridden. GEN_IMAGE_SIZE stays at the script's own default
# (1024), matching every task's partial_grid_input.png/gt_grid.png resolution.
#
# GPU: queued SEQUENTIALLY on whichever single GPU is confirmed idle when
# launched -- edit CUDA_VISIBLE_DEVICES below (checked via nvidia-smi
# immediately before launch, since this is a shared multi-tenant node).
#
#   bash run_composed_intermediate_sft_queue_3tasks.sh
#   bash run_composed_intermediate_sft_queue_3tasks.sh multi_object_placement:s2 rotation_puzzle:base
set -uo pipefail

REPO_ROOT="/scratch/network/ssd2/junlin/ssl_mllm/Model_Related/InternVLU/InternVL/internvl_chat"
SFT_SCRIPT="shell/internvlu/sft/run_target_pred_mop_sft.sh"
META_ROOT="/scratch/network/ssd2/junlin/ssl_mllm/baseline_pipeline/data/v1/meta"
OUT_ROOT="/scratch/network/ssd/junlin/models"
LOG_ROOT="/scratch/network/ssd2/junlin/ssl_mllm/results/composed_intermediate_sft_logs"
mkdir -p "$LOG_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export GPUS="${GPUS:-1}"

declare -A CKPTS=(
  [base]=""  # unset -> run_target_pred_mop_sft.sh's own BASE_SNAPSHOT default
  [s2]="/scratch/network/ssd2/junlin/models/internvlu-s2-gen-merged"
  [s4]="/scratch/network/ssd2/junlin/models/internvlu-s4-gen-merged"
)

TASKS=(multi_object_placement rotation_puzzle shape_color_then_move)
MODELS=(base s2 s4)

# Build the default job list "task:model" x9, or take an override list from argv.
DEFAULT_JOBS=()
for t in "${TASKS[@]}"; do
  for m in "${MODELS[@]}"; do
    DEFAULT_JOBS+=("${t}:${m}")
  done
done
JOBS=("${@:-${DEFAULT_JOBS[@]}}")

cd "$REPO_ROOT"

for job in "${JOBS[@]}"; do
    task="${job%%:*}"
    name="${job##*:}"
    ckpt="${CKPTS[$name]}"
    meta_path="${META_ROOT}/${task}_Composed_intermediate_train_meta.json"
    out_dir="${OUT_ROOT}/internvlu-${name}-${task}500sft"
    log_file="${LOG_ROOT}/${task}_${name}.log"

    echo "=============================================================="
    echo " [queue] ${task}/${name}  ckpt=${ckpt:-<script default: BASE_SNAPSHOT>}"
    echo " [queue] meta=$meta_path"
    echo " [queue] out=$out_dir  log=$log_file"
    echo " [queue] started: $(date)"
    echo "=============================================================="

    if [ ! -f "$meta_path" ]; then
        echo " [queue] ${task}/${name} SKIPPED -- meta not found: $meta_path"
        continue
    fi

    if [ -n "$ckpt" ]; then
        META_PATH="$meta_path" OUTPUT_DIR="$out_dir" INTERNVLU_CKPT="$ckpt" \
            bash "$SFT_SCRIPT" > "$log_file" 2>&1
    else
        META_PATH="$meta_path" OUTPUT_DIR="$out_dir" \
            bash "$SFT_SCRIPT" > "$log_file" 2>&1
    fi

    status=$?
    if [ $status -eq 0 ]; then
        echo " [queue] ${task}/${name} DONE ($(date)) -> ${out_dir}-merged"
    else
        echo " [queue] ${task}/${name} FAILED (exit $status, $(date)) -- see $log_file"
    fi
done

echo "=============================================================="
echo " [queue] all requested jobs processed: ${JOBS[*]}"
echo "=============================================================="
