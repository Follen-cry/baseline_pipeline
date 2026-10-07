#!/bin/bash
# Reproducible end-to-end: train the InternVL-U next-frame-generation SFT on
# the S2 setting (VC2I-F) of the final S0-S3 merged dataset -- 3 context
# frames + caption (scene description / narration / pose-derived action
# text) -> fixed next-frame target -- then fix the saved processor and merge
# the LoRA into an inference-ready pipeline.
#
# Data: data/v1/datasets/final_s0s3/S2_train.jsonl, 49,113 rows, interleaving VBVR (25k,
# per-task scene_description()) with 4 real-world sources whose captions are
# real narration/pose/Panda-70M text, not synthesized (EPIC-Kitchen, NWM/
# RECON, panda70m_epic_ssl, panda70m_ssl v1). See data/specs/S0_S3_DATASET_SPEC.md
# for full provenance. S0/S1/S2/S3 are trained as 4 independent models (no
# cross-setting mixing), so there is one of these scripts per setting and no
# repeat_time tuning across them. The on-disk row order is source-blocked,
# not shuffled -- fine as-is, since internvl_chat_finetune_u_full.py's
# TaskTypeBatchSampler already reshuffles indices every epoch via set_epoch
# (verified, no config needed).
#
# Recipe (shared template across S0-S3 -- adjust per-setting via env once you
# have a reason to): full-unified (trains generation_decoder + LLM-LoRA r32 +
# mlp1), 1 epoch, lr 1e-5, imgen_ratio 1.0, gen_loss_weight 0.5 (warmup 20),
# gen 512, effective batch 16. ~3,070 total steps at this row count.
#
# Every row's cond_image is direction-aware (always the context frame
# nearest the target), so INTERNVLU_VAE_COND=1 is exported to make explicit
# that the code uses this field rather than a fallback.
#
# LM_LOSS_WEIGHT defaults to 0.0 here (2026-08-31 decision): even though S2
# rows carry a real caption, that caption sits in the (masked, -100) human
# turn -- the GPT turn being supervised is still the same fixed phrase ("The
# next frame should look like this:") on every row, so its text-CE signal
# carries ~zero information regardless of caption content. Zeroing
# lm_loss_weight is the code-level no-ce switch (model.configure_lm_loss_
# weight), replacing the old data-side bare-"<img>" hack. S3 keeps
# LM_LOSS_WEIGHT=1.0 since its answer-letter CE is real supervision.
#
# No held-out eval wired in for S2 (none exists for S0-S2; only S3 has one,
# and it isn't plumbed into this launcher yet -- eval_strategy stays "no").
#
# IMPORTANT: distributed (multi-GPU) works in the `internvlu` env (torch 2.6 /
# NCCL 2.21.5). It does NOT work in the vLLM env (NCCL 2.27.5 segfaults on the
# 530.30.02 driver). Run this with the internvlu env, on a node with free GPUs
# (check `nvidia-smi` first -- shared multi-tenant cluster).
#
# Usage:
#   conda activate internvlu
#   cd Model_Related/InternVLU/InternVL/internvl_chat
#   bash shell/internvlu/sft/v1/run_s2_gen_sft.sh
# Override anything via env, e.g.:
#   GPUS=4 EPOCHS=1 OUTPUT_DIR=/path bash shell/internvlu/sft/v1/run_s2_gen_sft.sh
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
# Ratio between the two loss terms is GEN_LOSS_WEIGHT : LM_LOSS_WEIGHT -- see
# modeling_internvlu_unified.py configure_lm_loss_weight/forward(); a 3rd
# "ratio" knob would be redundant with these two independent weights
# (2026-08-31 decision). LM_LOSS_WEIGHT defaults to 0.0 for S2 (see header).
# GEN_LOSS_WEIGHT reverted to the project-wide template default of 0.5
# (briefly tried 1.0 across S0-S3, reverted same day -- with LM_LOSS_WEIGHT=
# 0.0 here gen_loss is the only active term, and per the earlier Adam
# discussion a constant scale on a single loss is nearly invariant to
# AdamW's per-parameter normalization at steady state, so this matches
# every other run_*_gen_sft.sh in the repo instead of deviating for no
# measurable benefit).
LM_LOSS_WEIGHT="${LM_LOSS_WEIGHT:-0.0}"

# Hybrid ViT+VAE conditioning: all context frames go through the ViT (standard
# multi-image path); the VAE-encoded pixel condition uses each row's explicit
# cond_image (direction-aware -- always the context frame nearest the
# target). See dataset_unified.py's MultimodalImgenLazyDataset.__getitem__.
export INTERNVLU_VAE_COND="${INTERNVLU_VAE_COND:-1}"

META_PATH="${META_PATH:-/scratch/network/ssd2/junlin/ssl_mllm/data/meta/final_S2_meta.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/network/ssd2/junlin/models/internvlu-s2-gen}"
MERGED_DIR="${MERGED_DIR:-${OUTPUT_DIR}-merged}"
BASE_SNAPSHOT="${BASE_SNAPSHOT:-/scratch/network/ssd2/junlin/huggingface/hub/models--InternVL-U--InternVL-U/snapshots/f012d760e69712bb47f7d3d09a24280f346cee01}"
INTERNVLU_CKPT="${INTERNVLU_CKPT:-$BASE_SNAPSHOT}"
INTERNVLU_PKG_PATH="${INTERNVLU_PKG_PATH:-/scratch/network/ssd2/junlin/ssl_mllm/Model_Related/InternVLU/InternVL-U}"

echo "=============================================================="
echo " S2 (VC2I-F) GEN SFT  |  GPUs=$GPUS epochs=$EPOCHS lr=$LR lora=r$LORA_RANK"
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
