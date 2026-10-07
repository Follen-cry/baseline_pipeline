#!/bin/bash
# Reproducible end-to-end: train the InternVL-U next-frame-generation SFT on
# the S3 setting (VC2IA-F) of the final S0-S3 merged dataset -- 3 context
# frames + caption + 4 MCQ candidate frames -> fixed next-frame target PLUS
# the answer letter (A/B/C/D) -- then fix the saved processor and merge the
# LoRA into an inference-ready pipeline.
#
# Data: data/v1/datasets/final_s0s3/S3_train.jsonl, 47,770 rows (imgen_form: multimodal),
# interleaving VBVR (25k, same-task-different-window distractors -- flagged
# as an easier v1 scheme, kept as-is per 2026-08-31 decision, revisit only if
# the trained model saturates VBVR's MCQ trivially) with 4 real-world sources
# whose distractors are same-clip/same-video temporal-distance negatives
# (near/mid/far) mined first, cross-clip random only as fallback. See
# data/specs/S0_S3_DATASET_SPEC.md for full provenance. S0/S1/S2/S3 are trained as
# 4 independent models (no cross-setting mixing), so there is one of these
# scripts per setting and no repeat_time tuning across them. The on-disk row
# order is source-blocked, not shuffled -- fine as-is, since
# internvl_chat_finetune_u_full.py's TaskTypeBatchSampler already reshuffles
# indices every epoch via set_epoch (verified, no config needed).
#
# There is a clean 2,075-row held-out eval split
# (data/v1/datasets/final_s0s3/S3_eval.jsonl / data/v1/meta/final_S3_eval_meta.json,
# eval_holdout: true, verified zero image-level overlap with train) but it is
# NOT wired into this launcher -- internvlu_4b_sft_full_unified.sh hardcodes
# --eval_strategy "no" (2026-08-31 decision: train only for now, revisit if
# in-training MCQ-accuracy tracking becomes useful).
#
# max_dynamic_patch is 7 for S3 (3 context + 4 MCQ options) vs. 3 for S0-S2 --
# already set per-entry in final_S3_meta.json and takes priority over the
# launcher's own --max_dynamic_patch default (see
# internvl_chat_finetune_u_full.py: `entry.get("max_dynamic_patch",
# data_args.max_dynamic_patch)`), so no extra flag needed here.
#
# is_option (per-image 0/1, marks which of the 7 images are MCQ candidates)
# is only consumed to build option_context_mask if --gen_no_leak is passed;
# omitted here to keep this project's existing "dual-leak" convention (option
# images stay inside the generation-conditioning span).
#
# Recipe (shared template across S0-S3 -- adjust per-setting via env once you
# have a reason to): full-unified (trains generation_decoder + LLM-LoRA r32 +
# mlp1), 1 epoch, lr 1e-5, imgen_ratio 1.0, gen_loss_weight 0.5 (warmup 20),
# gen 512, effective batch 16. ~2,986 total steps at this row count. Same
# forward pass also runs text cross-entropy on the answer letter, weighted
# 0.5 (lm_loss_weight) -- gen:lm ratio is 1:1, matching every other
# run_*_gen_sft.sh's project-wide 0.5 template default (briefly tried 1.0/1.0
# on 2026-08-31, reverted same day: a uniform scale on both weights together
# is nearly invariant under AdamW's per-parameter normalization as long as
# the ratio between them is unchanged, so there was no measurable benefit to
# deviating from the established default).
#
# Every row's cond_image is direction-aware (always the context frame nearest
# the target), so INTERNVLU_VAE_COND=1 is exported to make explicit that the
# code uses this field rather than a fallback.
#
# IMPORTANT: distributed (multi-GPU) works in the `internvlu` env (torch 2.6 /
# NCCL 2.21.5). It does NOT work in the vLLM env (NCCL 2.27.5 segfaults on the
# 530.30.02 driver). Run this with the internvlu env, on a node with free GPUs
# (check `nvidia-smi` first -- shared multi-tenant cluster).
#
# Usage:
#   conda activate internvlu
#   cd Model_Related/InternVLU/InternVL/internvl_chat
#   bash shell/internvlu/sft/v1/run_s3_gen_sft.sh
# Override anything via env, e.g.:
#   GPUS=4 EPOCHS=1 OUTPUT_DIR=/path bash shell/internvlu/sft/v1/run_s3_gen_sft.sh
set -euo pipefail

REPO_ROOT="/scratch/network/ssd2/junlin/ssl_mllm/Model_Related/InternVLU/InternVL/internvl_chat"
cd "$REPO_ROOT"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---- config (override via env) ----
GPUS="${GPUS:-8}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1e-5}"
LORA_RANK="${LORA_RANK:-32}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GEN_IMAGE_SIZE="${GEN_IMAGE_SIZE:-512}"
GEN_LOSS_WEIGHT="${GEN_LOSS_WEIGHT:-0.5}"
GEN_LOSS_WARMUP="${GEN_LOSS_WARMUP:-20}"
IMGEN_RATIO="${IMGEN_RATIO:-1.0}"
# Ratio between the two loss terms is GEN_LOSS_WEIGHT : LM_LOSS_WEIGHT -- both
# set to 0.5 (1:1 ratio, matching the project-wide template default -- see
# header for the 2026-08-31 back-and-forth). See modeling_internvlu_unified.py
# configure_lm_loss_weight/forward(); a 3rd "ratio" knob would be redundant
# with these two independent weights.
LM_LOSS_WEIGHT="${LM_LOSS_WEIGHT:-0.5}"

# Hybrid ViT+VAE conditioning: all context frames go through the ViT (standard
# multi-image path); the VAE-encoded pixel condition uses each row's explicit
# cond_image (direction-aware -- always the context frame nearest the
# target). See dataset_unified.py's MultimodalImgenLazyDataset.__getitem__.
export INTERNVLU_VAE_COND="${INTERNVLU_VAE_COND:-1}"

META_PATH="${META_PATH:-/scratch/network/ssd2/junlin/ssl_mllm/data/meta/final_S3_meta.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/network/ssd2/junlin/models/internvlu-s3-gen}"
MERGED_DIR="${MERGED_DIR:-${OUTPUT_DIR}-merged}"
BASE_SNAPSHOT="${BASE_SNAPSHOT:-/scratch/network/ssd2/junlin/huggingface/hub/models--InternVL-U--InternVL-U/snapshots/f012d760e69712bb47f7d3d09a24280f346cee01}"
INTERNVLU_CKPT="${INTERNVLU_CKPT:-$BASE_SNAPSHOT}"
INTERNVLU_PKG_PATH="${INTERNVLU_PKG_PATH:-/scratch/network/ssd2/junlin/ssl_mllm/Model_Related/InternVLU/InternVL-U}"

echo "=============================================================="
echo " S3 (VC2IA-F) GEN SFT  |  GPUs=$GPUS epochs=$EPOCHS lr=$LR lora=r$LORA_RANK"
echo " gen_loss_weight=$GEN_LOSS_WEIGHT (warmup $GEN_LOSS_WARMUP steps)  lm_loss_weight=$LM_LOSS_WEIGHT"
echo " meta   : $META_PATH"
echo " output : $OUTPUT_DIR"
echo " merged : $MERGED_DIR"
echo "=============================================================="
mkdir -p "$OUTPUT_DIR"

# ---- 1) train (full-unified launcher; trailing args override its hardcoded defaults) ----
GPUS="$GPUS" \
BATCH_SIZE="$BATCH_SIZE" \
PER_DEVICE_BATCH_SIZE=1 \
IMGEN_RATIO="$IMGEN_RATIO" \
GEN_IMAGE_SIZE="$GEN_IMAGE_SIZE" \
GEN_LOSS_WEIGHT="$GEN_LOSS_WEIGHT" \
GEN_LOSS_WARMUP="$GEN_LOSS_WARMUP" \
META_PATH="$META_PATH" \
OUTPUT_DIR="$OUTPUT_DIR" \
INTERNVLU_CKPT="$INTERNVLU_CKPT" \
INTERNVLU_PKG_PATH="$INTERNVLU_PKG_PATH" \
bash shell/internvlu/engine/internvlu_4b_sft_full_unified.sh \
    --num_train_epochs "$EPOCHS" \
    --learning_rate "$LR" \
    --use_llm_lora "$LORA_RANK" \
    --lm_loss_weight "$LM_LOSS_WEIGHT" \
    ${EXTRA_TRAIN_ARGS:-}

# ---- 2) fix the degenerate processor the training save writes ----
# (training's save_pretrained drops image_processor_kwargs -> InternVLUPipeline load
#  fails with KeyError: 'image_processor_kwargs'. Replace with the base snapshot's.)
if [ -d "$OUTPUT_DIR/processor" ] && [ ! -d "$OUTPUT_DIR/processor_broken_bak" ]; then
    mv "$OUTPUT_DIR/processor" "$OUTPUT_DIR/processor_broken_bak"
fi
cp -rL "$BASE_SNAPSHOT/processor" "$OUTPUT_DIR/processor"
echo "[processor] replaced with base snapshot's (image_processor_kwargs restored)"

# ---- 3) merge LoRA -> inference-ready pipeline ----
python tools/merge_lora_u_full.py "$OUTPUT_DIR" "$MERGED_DIR" --copy --base-snapshot "$BASE_SNAPSHOT"

# ---- 4) verify merged VLM has no LoRA tensors left ----
python - "$MERGED_DIR" <<'PY'
import glob, sys
from safetensors import safe_open
f = glob.glob(f"{sys.argv[1]}/vlm/*.safetensors")[0]
ks = list(safe_open(f, "pt").keys())
n_lora = sum("lora" in k for k in ks)
print(f"[verify] merged vlm: {n_lora} lora keys / {len(ks)} total  ->",
      "OK" if n_lora == 0 else "STILL HAS LORA")
PY

echo "DONE. Inference-ready pipeline: $MERGED_DIR"
