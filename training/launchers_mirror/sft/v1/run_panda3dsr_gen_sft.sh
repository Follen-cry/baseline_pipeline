#!/bin/bash
# Reproducible end-to-end: train the InternVL-U next-frame-generation SFT on
# the Panda-70M panda3dsr_gen_noaction split (1500 train), then fix the
# saved processor and merge the LoRA into an inference-ready pipeline.
# "noaction" = no instruction text in the prompt, pure visual next-frame
# prediction from 4 context frames. Source: Panda-70M clips caption-filtered
# to match 3DSRBench's visual distribution (data/scripts/build_panda70m_3dsrbench_filter.py),
# built into SSL data via data/scripts/build_panda70m_ssl.py.
#
# 1500/150 train/eval (not EPIC's 4000/300): a 5000-clip video2dataset download
# was attempted but YouTube bot-blocked this cluster's egress IP partway through
# (only 1073/5000 clips, 21.5%, actually downloaded -- see
# data/scripts/build_panda70m_ssl.py module docstring). Targets rescaled to what that
# pool honestly supports.
#
# Recipe (identical to run_epic_gen_sft.sh, for fair comparison): full-unified
#   (trains generation_decoder + LLM-LoRA r32 + mlp1), 2 epochs, lr 1e-5,
#   imgen_ratio 1.0, gen_loss_weight 0.5 (warmup 20), gen 512, effective batch 16.
#   Single-node DDP, GPUS GPUs (default 8).
#
# IMPORTANT: distributed (multi-GPU) works in the `internvlu` env (torch 2.6 /
# NCCL 2.21.5). It does NOT work in the vLLM env (NCCL 2.27.5 segfaults on the
# 530.30.02 driver). Run this with the internvlu env, on a node with free GPUs
# -- this recipe's 8-image dual variant peaks near the A40 (46GB) memory
# ceiling, so prefer an A40 node, not a 24GB-class node.
#
# Usage:
#   conda activate internvlu
#   cd InternVL/internvl_chat
#   bash shell/internvlu/sft/v1/run_panda3dsr_gen_sft.sh
# Override anything via env, e.g.:
#   GPUS=4 EPOCHS=1 OUTPUT_DIR=/path bash shell/internvlu/sft/v1/run_panda3dsr_gen_sft.sh
set -euo pipefail

REPO_ROOT="/scratch/network/ssd2/junlin/ssl_mllm/Model_Related/InternVLU/InternVL/internvl_chat"
cd "$REPO_ROOT"

# ---- config (override via env) ----
GPUS="${GPUS:-8}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-1e-5}"
LORA_RANK="${LORA_RANK:-32}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GEN_IMAGE_SIZE="${GEN_IMAGE_SIZE:-512}"
GEN_LOSS_WEIGHT="${GEN_LOSS_WEIGHT:-0.5}"
GEN_LOSS_WARMUP="${GEN_LOSS_WARMUP:-20}"
IMGEN_RATIO="${IMGEN_RATIO:-1.0}"

META_PATH="${META_PATH:-/scratch/network/ssd2/junlin/ssl_mllm/data/meta/panda3dsr_gen_noaction_meta.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/network/ssd2/junlin/models/internvlu-panda3dsr-gen-noaction}"
MERGED_DIR="${MERGED_DIR:-${OUTPUT_DIR}-merged}"
# See run_epic_gen_sft.sh: the /homes/ HF cache copy is broken and stale post-
# reorg; use the repaired ssd2 copy and pass it through explicitly.
BASE_SNAPSHOT="${BASE_SNAPSHOT:-/scratch/network/ssd2/junlin/huggingface/hub/models--InternVL-U--InternVL-U/snapshots/f012d760e69712bb47f7d3d09a24280f346cee01}"
INTERNVLU_CKPT="${INTERNVLU_CKPT:-$BASE_SNAPSHOT}"
INTERNVLU_PKG_PATH="${INTERNVLU_PKG_PATH:-/scratch/network/ssd2/junlin/ssl_mllm/Model_Related/InternVLU/InternVL-U}"

echo "=============================================================="
echo " PANDA-70M/3DSRBench GEN SFT (noaction)  |  GPUs=$GPUS epochs=$EPOCHS lr=$LR lora=r$LORA_RANK"
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
python tools/merge_lora_u_full.py "$OUTPUT_DIR" "$MERGED_DIR" --copy

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
