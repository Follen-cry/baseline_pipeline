#!/bin/bash
# Reproducible end-to-end: train the InternVL-U next-frame-generation SFT on
# the S4 setting (progression-description + next-frame text-and-image
# prediction) -- 3 context frames + caption -> a GPT turn that (1) describes
# how the scene changes across the 3 frames, (2) predicts in free text what
# the next frame will look like ("Next frame prediction: ..."), and (3)
# triggers image generation via the same "<img>" token S0-S3 use -- then fix
# the saved processor and merge the LoRA into an inference-ready pipeline.
#
# Data: baseline_pipeline/data/v1/datasets/s4_progression/S4_train.jsonl, 49,112 rows --
# S2's exact window set (not S3's MCQ-filtered subset -- S4 drops the MCQ
# entirely, so it doesn't need S3's distractor-availability constraints; see
# baseline_pipeline/data/v1/recipes/s4_progression/README.md for the full
# design/decision log and why S2's superset was used instead of S3's).
# Captions for both new text segments are VLM-labeled (Qwen3-VL-30B-A3B-
# Instruct-FP8), not synthesized placeholders -- see that same README for
# the labeling run's details (model, prompt, infra fixes, final counts).
#
# Recipe (shared template across S0-S3 -- adjust per-setting via env once you
# have a reason to): full-unified (trains generation_decoder + LLM-LoRA r32 +
# mlp1), 1 epoch, lr 1e-5, imgen_ratio 1.0, gen_loss_weight 0.5 (warmup 20),
# gen 512, effective batch 16. ~3,070 total steps at this row count.
#
# max_dynamic_patch is 3 here (3 context frames, no MCQ candidates -- unlike
# S3's 7) -- already set per-entry in final_S4_meta.json, no extra flag
# needed (see internvl_chat_finetune_u_full.py: `entry.get("max_dynamic_patch",
# data_args.max_dynamic_patch)`).
#
# LM_LOSS_WEIGHT defaults to 0.5 here, matching S3's actual (not its stale
# header-comment claim of 1.0 -- see PROVENANCE.md for that discrepancy)
# 1:1 gen:lm ratio, since S4's GPT turn carries real, substantial text
# supervision (arguably richer than S3's single answer-letter). Whether this
# ratio should shift now that the text target is a multi-sentence caption
# instead of a single token is an OPEN QUESTION -- deferred to whoever
# reviews this run's results; 0.5/0.5 is the safe "don't deviate from the
# established project template without evidence" default (see S2's header
# for the fuller Adam-invariance argument against changing absolute scale).
#
# Every row's cond_image is direction-aware (always the last context frame,
# forward-only for S4 -- no direction mixing), so INTERNVLU_VAE_COND=1 is
# exported to make explicit that the code uses this field rather than a
# fallback.
#
# No held-out eval split exists for S4 (built from S2, which itself has none
# -- only S3 does), so eval_strategy stays "no" like S0-S2.
#
# IMPORTANT: distributed (multi-GPU) works in the `internvlu` env (torch 2.6 /
# NCCL 2.21.5). It does NOT work in the vLLM env (NCCL 2.27.5 segfaults on the
# 530.30.02 driver). Run this with the internvlu env, on a node with free GPUs
# (check `nvidia-smi` first -- shared multi-tenant cluster).
#
# Usage:
#   conda activate internvlu
#   cd Model_Related/InternVLU/InternVL/internvl_chat
#   bash shell/internvlu/sft/v1/run_s4_gen_sft.sh
# Override anything via env, e.g.:
#   GPUS=4 EPOCHS=1 OUTPUT_DIR=/path bash shell/internvlu/sft/v1/run_s4_gen_sft.sh
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
# modeling_internvlu_unified.py configure_lm_loss_weight/forward(). 0.5/0.5
# matches the project-wide template default (see header for why this wasn't
# changed despite S4's richer text target).
LM_LOSS_WEIGHT="${LM_LOSS_WEIGHT:-0.5}"

# Hybrid ViT+VAE conditioning: all context frames go through the ViT (standard
# multi-image path); the VAE-encoded pixel condition uses each row's explicit
# cond_image (direction-aware -- always the last context frame here, since S4
# is forward-only). See dataset_unified.py's MultimodalImgenLazyDataset.__getitem__.
export INTERNVLU_VAE_COND="${INTERNVLU_VAE_COND:-1}"

META_PATH="${META_PATH:-/scratch/network/ssd2/junlin/ssl_mllm/baseline_pipeline/data/v1/meta/final_S4_meta.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/network/ssd2/junlin/models/internvlu-s4-gen}"
MERGED_DIR="${MERGED_DIR:-${OUTPUT_DIR}-merged}"
BASE_SNAPSHOT="${BASE_SNAPSHOT:-/scratch/network/ssd2/junlin/huggingface/hub/models--InternVL-U--InternVL-U/snapshots/f012d760e69712bb47f7d3d09a24280f346cee01}"
INTERNVLU_CKPT="${INTERNVLU_CKPT:-$BASE_SNAPSHOT}"
INTERNVLU_PKG_PATH="${INTERNVLU_PKG_PATH:-/scratch/network/ssd2/junlin/ssl_mllm/Model_Related/InternVLU/InternVL-U}"

echo "=============================================================="
echo " S4 (progression + next-frame text+image) GEN SFT  |  GPUs=$GPUS epochs=$EPOCHS lr=$LR lora=r$LORA_RANK"
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
