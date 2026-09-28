#!/bin/bash
# edit_sft_probe DIAGNOSTIC probe (not a training-setting change): fine-tune one checkpoint with a
# small instruction-editing set, using exactly the v2 T-series recipe (sft/v2/run_t_gen_sft.sh:
# full-unified, LLM-LoRA r32, lr 1e-5, gen_loss_weight 0.5 / warmup 20, imgen_ratio 1.0, area-512
# targets, global batch 16, lm_loss_weight 0), 1 epoch over build_probe_data.py's ~800 rows = ~50
# steps. No in-training validation. Identical for base / T0 / T2 / T3; only the starting checkpoint
# differs.
#
# This is the lightweight variant paired with v2_suite_1_sub (200-item subset): same recipe as
# ../../v2_suite_1/diagnosis/edit_sft/run_edit_sft_probe.sh, only the data (800 rows, disjoint from
# the 200-item subset manifest instead of the full 1,210-item suite) and epoch count (1 instead of
# 2) differ. Usually launched via train_{base,T0,T2,T3}.sh rather than directly.
#
#   MODEL=T0 CKPT=/scratch/network/ssd/junlin/models/internvlu-v2-t0-gen-merged GPUS=4 bash run_probe_sft.sh
# Output: /scratch/network/ssd/junlin/models/edit_sft_probe_sub/<MODEL>(-merged)   (env: internvlu)
set -euo pipefail
: "${MODEL:?}"; : "${CKPT:?}"
GPUS="${GPUS:-4}"
LR="${LR:-1e-5}"
export GEN_DECODER_LR="${GEN_DECODER_LR:-5e-5}"
BASE_SNAPSHOT=/scratch/network/ssd2/junlin/huggingface/hub/models--InternVL-U--InternVL-U/snapshots/f012d760e69712bb47f7d3d09a24280f346cee01
CHAT=/scratch/network/ssd2/junlin/ssl_mllm/baseline_pipeline/training/models/internvl-u/internvl_chat
OUTPUT_DIR=/scratch/network/ssd/junlin/models/edit_sft_probe_sub/$MODEL
MERGED_DIR=${OUTPUT_DIR}-merged
mkdir -p "$OUTPUT_DIR"
cd "$CHAT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True INTERNVLU_VAE_COND=1 REPORT_TO=tensorboard
unset VAL_EVAL_JSONL
GPUS=$GPUS BATCH_SIZE=16 PER_DEVICE_BATCH_SIZE=1 IMGEN_RATIO=1.0 GEN_IMAGE_SIZE=512 \
GEN_LOSS_WEIGHT=0.5 GEN_LOSS_WARMUP=20 \
META_PATH=/scratch/network/ssd/junlin/ssl_eval/v2_eval_suite_1_sub/edit_sft_probe/data/meta.json \
OUTPUT_DIR="$OUTPUT_DIR" INTERNVLU_CKPT="$CKPT" \
INTERNVLU_PKG_PATH=/scratch/network/ssd2/junlin/ssl_mllm/Model_Related/InternVLU/InternVL-U \
MASTER_PORT="${MASTER_PORT:-34261}" \
bash shell/internvlu/engine/internvlu_4b_sft_full_unified.sh \
    --num_train_epochs 1 --learning_rate "$LR" --use_llm_lora 32 --lm_loss_weight 0.0 --gen_resize_mode area ${EXTRA:-}
grep -q train_runtime "$OUTPUT_DIR/training_log.txt" || { echo "training did not finish" >&2; exit 1; }
# same post-processing as the v2 launcher: restore the processor, merge LoRA
[ -d "$OUTPUT_DIR/processor" ] && [ ! -d "$OUTPUT_DIR/processor_broken_bak" ] && mv "$OUTPUT_DIR/processor" "$OUTPUT_DIR/processor_broken_bak"
cp -rL "$BASE_SNAPSHOT/processor" "$OUTPUT_DIR/processor"
python tools/merge_lora_u_full.py "$OUTPUT_DIR" "$MERGED_DIR" --copy --base-snapshot "$BASE_SNAPSHOT"
echo "DONE $MERGED_DIR"
