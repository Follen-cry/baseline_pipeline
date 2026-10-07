#!/bin/bash
# SFT sweep for the 2d_geo_trans_Composed_intermediate task (500 train rows,
# composed single-image input -> real-GT complete-grid output, VAE-conditioned
# on the single input image -- see data/v1/datasets/2d_geo_trans_Composed_intermediate/README.md).
#
# Trains 6 models FROM the same starting-checkpoint family used everywhere
# else in this repo (InternVL-U base + the S0-S4 stage-1 gen-SFT checkpoints),
# reusing run_target_pred_mop_sft.sh UNCHANGED (same script that produced
# internvlu-{base,s0,s1,s2,s3,s4}-4task500sft) with only META_PATH/
# INTERNVLU_CKPT/OUTPUT_DIR overridden. GEN_IMAGE_SIZE is left at the script's
# own default (1024), which matches this task's actual image resolution
# exactly (partial_grid_input.png / gt_grid.png are both 1024x1024) --
# verified before this sweep was launched, no override needed.
#
# IMPORTANT: only GPU 7 was confirmed idle on this node when this was written
# (GPUs 0-6 all had other users' active processes -- see `nvidia-smi`). All 6
# runs are therefore queued SEQUENTIALLY on GPU 7 alone (GPUS=1), not run in
# parallel, to avoid contending for shared infrastructure. Re-check
# `nvidia-smi` and edit CUDA_VISIBLE_DEVICES/GPUS below if more GPUs are free
# when this is re-run.
#
# A single run's failure does not abort the queue -- each is best-effort so
# one bad config doesn't block the other 5; check the per-model log for
# failures afterward.
#
#   bash run_2d_geo_trans_composed_intermediate_sft_queue.sh
#   bash run_2d_geo_trans_composed_intermediate_sft_queue.sh s2 s3   # only these
set -uo pipefail

# The baseline_pipeline submodule has its own copy of this script, but that
# copy's OWN internals still hardcode REPO_ROOT/INTERNVLU_PKG_PATH back to
# the original Model_Related/InternVLU tree (not yet repointed as part of the
# repo migration) -- so we invoke the original directly here rather than
# creating a false impression of running "from" the submodule.
REPO_ROOT="/scratch/network/ssd2/junlin/ssl_mllm/Model_Related/InternVLU/InternVL/internvl_chat"
SFT_SCRIPT="shell/internvlu/sft/run_target_pred_mop_sft.sh"
META_PATH="/scratch/network/ssd2/junlin/ssl_mllm/baseline_pipeline/data/v1/meta/2d_geo_trans_Composed_intermediate_train_meta.json"
OUT_ROOT="/scratch/network/ssd/junlin/models"
LOG_ROOT="/scratch/network/ssd2/junlin/ssl_mllm/results/2d_geo_trans_composed_intermediate_sft_logs"
mkdir -p "$LOG_ROOT"

export CUDA_VISIBLE_DEVICES="7"
export GPUS="1"

declare -A CKPTS=(
  [base]=""  # unset -> run_target_pred_mop_sft.sh's own BASE_SNAPSHOT default
  [s0]="/scratch/network/ssd2/junlin/models/internvlu-s0-gen-merged"
  [s1]="/scratch/network/ssd2/junlin/models/internvlu-s1-gen-merged"
  [s2]="/scratch/network/ssd2/junlin/models/internvlu-s2-gen-merged"
  [s3]="/scratch/network/ssd2/junlin/models/internvlu-s3-gen-merged"
  [s4]="/scratch/network/ssd2/junlin/models/internvlu-s4-gen-merged"
)
ORDER=(base s0 s1 s2 s3 s4)

MODELS=("${@:-${ORDER[@]}}")

cd "$REPO_ROOT"

for name in "${MODELS[@]}"; do
    ckpt="${CKPTS[$name]}"
    out_dir="${OUT_ROOT}/internvlu-${name}-2dgeo500sft"
    log_file="${LOG_ROOT}/${name}.log"

    echo "=============================================================="
    echo " [queue] $name  ckpt=${ckpt:-<script default: BASE_SNAPSHOT>}"
    echo " [queue] out=$out_dir  log=$log_file"
    echo " [queue] started: $(date)"
    echo "=============================================================="

    if [ -n "$ckpt" ]; then
        META_PATH="$META_PATH" OUTPUT_DIR="$out_dir" INTERNVLU_CKPT="$ckpt" \
            bash "$SFT_SCRIPT" > "$log_file" 2>&1
    else
        META_PATH="$META_PATH" OUTPUT_DIR="$out_dir" \
            bash "$SFT_SCRIPT" > "$log_file" 2>&1
    fi

    status=$?
    if [ $status -eq 0 ]; then
        echo " [queue] $name DONE ($(date)) -> ${out_dir}-merged"
    else
        echo " [queue] $name FAILED (exit $status, $(date)) -- see $log_file"
    fi
done

echo "=============================================================="
echo " [queue] all requested models processed: ${MODELS[*]}"
echo "=============================================================="
