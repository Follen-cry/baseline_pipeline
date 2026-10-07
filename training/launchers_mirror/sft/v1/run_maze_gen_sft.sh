#!/bin/bash
# Reproducible end-to-end: train the InternVL-U next-frame-generation SFT on the
# 5x5 maze dataset (data/datasets/maze_dataset, 9600 train clips), then fix the saved
# processor and merge the LoRA into an inference-ready pipeline.
#
# Task: 3 context frames -> generate the 4th. The agent advances 2 maze cells
# per frame along a fixed GT path; unlike the video-derived SSL sets the target
# frame is provably correct, and the only thing that changes between frames is
# the agent's position.
#
# Recipe: identical to run_epic_gen_sft.sh (full-unified: generation_decoder +
#   LLM-LoRA r32 + mlp1, lr 1e-5, imgen_ratio 1.0, gen_loss_weight 0.5
#   (warmup 20), gen 512, effective batch 16) except EPOCHS defaults to 1.
#   9600 train rows / batch 16 = 600 optimizer steps.
#
# Two deliberate differences from the EPIC runs:
#   - the meta uses max_dynamic_patch=1: maze frames are rendered at exactly
#     448x448, one native tile, so tiling would only upsample crisp line art.
#   - OUTPUT_DIR lives on /scratch/network/ssd (4.4T free), not ssd2, which was
#     down to ~29G. Both are network-visible from any node.
#
# IMPORTANT: distributed (multi-GPU) works in the `internvlu` env (torch 2.6 /
# NCCL 2.21.5). It does NOT work in the vLLM env (NCCL 2.27.5 segfaults on the
# 530.30.02 driver). Run this with the internvlu env, on a node with free GPUs.
#
# Usage:
#   conda activate internvlu
#   cd InternVL/internvl_chat
#   bash shell/internvlu/sft/v1/run_maze_gen_sft.sh
# Override anything via env, e.g.:
#   GPUS=4 EPOCHS=2 OUTPUT_DIR=/path bash shell/internvlu/sft/v1/run_maze_gen_sft.sh
set -euo pipefail

REPO_ROOT="/scratch/network/ssd2/junlin/ssl_mllm/Model_Related/InternVLU/InternVL/internvl_chat"
cd "$REPO_ROOT"

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

META_PATH="${META_PATH:-/scratch/network/ssd2/junlin/ssl_mllm/data/meta/maze_gen_meta.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/network/ssd/junlin/models/internvlu-maze-gen}"
MERGED_DIR="${MERGED_DIR:-${OUTPUT_DIR}-merged}"
# NOTE: the /homes/ HF cache copy of the base snapshot is BROKEN (missing custom
# modeling/tokenizer files, see memory internvlu_homes_cache_corruption). Use the
# repaired /scratch/network/ssd2 copy and pass it through as INTERNVLU_CKPT --
# the core launcher's own default still points at the broken /homes/ path.
BASE_SNAPSHOT="${BASE_SNAPSHOT:-/scratch/network/ssd2/junlin/huggingface/hub/models--InternVL-U--InternVL-U/snapshots/f012d760e69712bb47f7d3d09a24280f346cee01}"
INTERNVLU_CKPT="${INTERNVLU_CKPT:-$BASE_SNAPSHOT}"
INTERNVLU_PKG_PATH="${INTERNVLU_PKG_PATH:-/scratch/network/ssd2/junlin/ssl_mllm/Model_Related/InternVLU/InternVL-U}"

echo "=============================================================="
echo " MAZE GEN SFT  |  GPUs=$GPUS epochs=$EPOCHS lr=$LR lora=r$LORA_RANK"
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
