#!/bin/bash
# InternVL-U discriminative SFT on WorldPrediction-WM training data
# (baseline_pipeline/data/v1/sources/worldprediction/ -- CrossTask + EPIC-KITCHENS-100
# + IKEAASM, COIN held out as eval; see that dir's README for full provenance).
#
# This is a TEXT-ONLY LoRA SFT (4-way MCQ: "{"action": "X"}"), not an image-
# generation run -- unlike every run_s{0..4}_gen_sft.sh, which all call the
# "_full"/"_full_unified" engine (trains generation_decoder + LLM-LoRA + mlp1
# on a mixed gen+lm loss). WM has zero imgen rows, so this deliberately reuses
# the plain understanding-only engine instead:
#   internvl/train/internvl_chat_finetune_u.py (via engine/internvlu_4b_sft_lora.sh's
#   pattern), NOT internvl_chat_finetune_u_full.py.
# This is the same engine legacy/train_ffs_v2_1ep.sh used for CLEVRER FFS (another
# 4-way MCQ "understanding" task) -- that script is this one's closest real
# precedent, not the S0-S4 template.
#
# IMPORTANT path gotcha: the plain understanding engine loads the InternVLUChatModel
# directly from the snapshot's `vlm/` component subfolder (NOT the snapshot root
# InternVLUPipeline.from_pretrained() expects for the imgen scripts) -- INTERNVLU_CKPT
# below has `/vlm` appended for exactly this reason. Get this wrong and the load
# will either fail or silently pick up the wrong config.
#
# Small-dataset hyperparameters: use_llm_lora is 16 (not S0-S4's 32) specifically
# because these train sets (few hundred rows) are small enough that a higher-
# capacity adapter risks memorizing rather than generalizing -- matches the
# FFS precedent, not arbitrary.
#
# EPOCHS default is 1 (set 2026-09-13 per explicit user direction, standing
# default for this experiment series going forward). Note this is NOT what
# the original epoch-count comparison found best: on the first
# worldprediction/COIN-eval-split experiment, epoch=1 (~14-16 steps) was
# empirically undertrained and noisy (checkpoint ranking scrambled vs.
# epoch=15, several results fell below the zero-shot/majority-baseline
# reference) while epoch=15 (~200-240 steps) gave a much cleaner signal --
# see project memory (project_worldprediction_wm_sft_6checkpoint_comparison).
# Override with EPOCHS=15 (or any value) via env if you want that behavior
# back; this default only controls what happens when EPOCHS is unset.
#
# No in-loop eval: eval_strategy stays "no", matching every run_*_sft.sh in
# this repo (none of them wire HF Trainer's in-loop eval, even S3 which has a
# real held-out split sitting unused). Evaluate this run for real by pointing
# eval/suites/worldprediction/run.py at the merged checkpoint and running the
# COIN-only WM benchmark -- that's the actual scoring path already used for
# base/S0-S4 (results/internvlu_s{0..4}_4task500sft_4f/), not our wm_eval.jsonl
# (that jsonl exists for train/eval format-parity bookkeeping, not as the
# scoring mechanism).
#
# GPU note: this is a shared multi-tenant node -- CUDA_VISIBLE_DEVICES below
# defaults to whatever was free when this script was written (2026-09-11);
# RE-CHECK `nvidia-smi` before launching, it will have changed.
#
# Usage:
#   conda activate internvlu
#   cd baseline_pipeline/training/models/internvl-u/internvl_chat
#   bash shell/internvlu/sft/v1/run_worldprediction_wm_sft.sh
# Override anything via env, e.g.:
#   GPUS=2 CUDA_VISIBLE_DEVICES=6,7 EPOCHS=20 bash shell/internvlu/sft/v1/run_worldprediction_wm_sft.sh
set -euo pipefail

REPO_ROOT="/scratch/network/ssd2/junlin/ssl_mllm/baseline_pipeline/training/models/internvl-u/internvl_chat"
cd "$REPO_ROOT"

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton-cache-$USER}"
mkdir -p "$TRITON_CACHE_DIR"

# ---- config (override via env) ----
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,6,7}"
GPUS="${GPUS:-4}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-4e-5}"
LORA_RANK="${LORA_RANK:-16}"
BATCH_SIZE="${BATCH_SIZE:-32}"          # effective (global) batch size
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-12288}"  # 18 images/row (2 state + 4 cands x 4 frames);
                                            # verify with a short smoke test before a full run

META_PATH="${META_PATH:-/scratch/network/ssd2/junlin/ssl_mllm/baseline_pipeline/data/v1/meta/worldprediction_wm_train_meta.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/network/ssd2/junlin/models/internvlu-wm-sft}"
MERGED_DIR="${MERGED_DIR:-${OUTPUT_DIR}-merged}"
BASE_SNAPSHOT="${BASE_SNAPSHOT:-/scratch/network/ssd2/junlin/huggingface/hub/models--InternVL-U--InternVL-U/snapshots/f012d760e69712bb47f7d3d09a24280f346cee01}"
INTERNVLU_CKPT="${INTERNVLU_CKPT:-$BASE_SNAPSHOT/vlm}"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export MASTER_PORT="${MASTER_PORT:-34231}"
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

echo "=============================================================="
echo " WorldPrediction-WM SFT (understanding-only, no imgen)"
echo " GPUs=$GPUS (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) epochs=$EPOCHS lr=$LR lora=r$LORA_RANK"
echo " effective batch=$BATCH_SIZE (grad_acc=$GRADIENT_ACC, per_device=$PER_DEVICE_BATCH_SIZE)"
echo " meta   : $META_PATH"
echo " ckpt   : $INTERNVLU_CKPT"
echo " output : $OUTPUT_DIR"
echo " merged : $MERGED_DIR"
echo "=============================================================="
mkdir -p "$OUTPUT_DIR"

# ---- 1) train ----
torchrun \
  --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 \
  --nproc_per_node="$GPUS" --master_port="$MASTER_PORT" \
  internvl/train/internvl_chat_finetune_u.py \
  --model_name_or_path "$INTERNVLU_CKPT" \
  --conv_style "qwen2_5-chat-v3" \
  --use_fast_tokenizer False \
  --output_dir "$OUTPUT_DIR" \
  --meta_path "$META_PATH" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp False \
  --freeze_backbone True \
  --use_llm_lora "$LORA_RANK" \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs "$EPOCHS" \
  --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACC" \
  --eval_strategy "no" \
  --save_strategy "no" \
  --save_total_limit 1 \
  --learning_rate "$LR" \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --report_to "tensorboard" \
  ${EXTRA_TRAIN_ARGS:-} \
  2>&1 | tee -a "$OUTPUT_DIR/training_log.txt"

# ---- 2) merge LoRA -> plain InternVLUChatModel checkpoint ----
# Uses merge_lora_u.py (the plain understanding-model merge tool), NOT
# merge_lora_u_full.py (that's for the imgen pipeline's multi-component
# InternVLUPipeline checkpoints, which this run never produces).
python tools/merge_lora_u.py "$OUTPUT_DIR" "$MERGED_DIR"

echo "DONE. Merged checkpoint: $MERGED_DIR"
echo "Next: point eval/suites/worldprediction/configs/vlm/internvlu/*.json's"
echo "model_name at $MERGED_DIR and run the COIN-only WM benchmark to score it."
