#!/bin/bash
# Reproducible end-to-end: train the InternVL-U next-frame-generation SFT on
# the VBVR next-frame split (10 in-domain tasks x 10,000 windows = 100,000
# train rows, 3-in -> 1-out), then fix the saved processor and merge the LoRA
# into an inference-ready pipeline.
#
# Data: no real text GT exists for VBVR (the "prompt" is a synthesized
# instruction, not an annotated caption), so this defaults to the --no-ce
# jsonl produced by VBVR-DataGeneration/next_frame/vbvr_next_frame_sample.py
# --no-ce (data/datasets/vbvr_next_frame/next_frame_train_no_ce.jsonl): the GPT turn is
# a bare "<img>" marker instead of "The next frame should look like this:
# <img>", so the text-CE span collapses to a single always-identical token --
# as close to image-loss-only as the data side gets without a trainer-side
# labels-masking change (see that script's module docstring for why that
# wasn't added). LLM stays on LoRA (never fully unfrozen) specifically so a
# huge, narrow-domain, zero-real-text dataset like this can't erode language/
# instruction-following capability; only the LoRA delta + gen_decoder + mlp1
# actually train here.
#
# GPUs: written for torrnode8, which had 4 fully idle A40s (idx 2,3,6,7) as of
# the 2026-08-26 check -- confirm with `nvidia-smi` on torrnode8 before
# running, this is a shared multi-tenant cluster and free GPUs can be claimed
# by someone else at any time. Override CUDA_VISIBLE_DEVICES/GPUS for a
# different node or GPU set.
#
# IMPORTANT: distributed (multi-GPU) works in the `internvlu` env (torch 2.6 /
# NCCL 2.21.5). It does NOT work in the vLLM env (NCCL 2.27.5 segfaults on the
# 530.30.02 driver). Run this with the internvlu env.
#
# Usage:
#   ssh torrnode8
#   conda activate internvlu
#   cd /scratch/network/ssd2/junlin/ssl_mllm/Model_Related/InternVLU/InternVL/internvl_chat
#   bash shell/internvlu/sft/v1/run_vbvr_gen_sft.sh
# Override anything via env, e.g.:
#   GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 EPOCHS=2 bash shell/internvlu/sft/v1/run_vbvr_gen_sft.sh
# To train on the old CE-supervised jsonl instead (comparison run):
#   META_PATH=/scratch/network/ssd2/junlin/ssl_mllm/data/meta/vbvr_next_frame_meta.json \
#   OUTPUT_DIR=/scratch/network/ssd/junlin/models/internvlu-vbvr-gen-ce \
#   bash shell/internvlu/sft/v1/run_vbvr_gen_sft.sh
set -euo pipefail

REPO_ROOT="/scratch/network/ssd2/junlin/ssl_mllm/Model_Related/InternVLU/InternVL/internvl_chat"
cd "$REPO_ROOT"

# expandable_segments reclaims allocator fragmentation slack that has tipped
# marginal-memory GPUs into OOM on this shared cluster before (see the other
# run_*_sft.sh scripts in this dir).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---- GPU selection (torrnode8, 4 idle A40s as of 2026-08-26 -- re-check!) ----
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,6,7}"
GPUS="${GPUS:-4}"

# ---- config (override via env) ----
EPOCHS="${EPOCHS:-1}"                 # 100k rows is already large; 1 epoch = ~6,250 steps at batch 16
LR="${LR:-1e-5}"
LORA_RANK="${LORA_RANK:-32}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GEN_IMAGE_SIZE="${GEN_IMAGE_SIZE:-512}"
GEN_LOSS_WEIGHT="${GEN_LOSS_WEIGHT:-0.5}"
GEN_LOSS_WARMUP="${GEN_LOSS_WARMUP:-20}"
IMGEN_RATIO="${IMGEN_RATIO:-1.0}"

# Hybrid ViT+VAE conditioning: all 3 input frames go through the ViT (standard
# multi-image path); the VAE-encoded pixel condition defaults to the LAST of
# them (frame_2, "ctx3") since VBVR rows carry no explicit cond_image -- see
# dataset_unified.py's MultimodalImgenLazyDataset.__getitem__. This is already
# the code default (INTERNVLU_VAE_COND defaults to "1"); exported explicitly
# here to match run_epic_gen_sft.sh / run_maze_route_sft.sh's convention of
# not relying on the implicit default.
export INTERNVLU_VAE_COND="${INTERNVLU_VAE_COND:-1}"

META_PATH="${META_PATH:-/scratch/network/ssd2/junlin/ssl_mllm/data/meta/vbvr_next_frame_no_ce_meta.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/network/ssd/junlin/models/internvlu-vbvr-gen-noce}"
MERGED_DIR="${MERGED_DIR:-${OUTPUT_DIR}-merged}"
BASE_SNAPSHOT="${BASE_SNAPSHOT:-/scratch/network/ssd2/junlin/huggingface/hub/models--InternVL-U--InternVL-U/snapshots/f012d760e69712bb47f7d3d09a24280f346cee01}"
INTERNVLU_CKPT="${INTERNVLU_CKPT:-$BASE_SNAPSHOT}"
INTERNVLU_PKG_PATH="${INTERNVLU_PKG_PATH:-/scratch/network/ssd2/junlin/ssl_mllm/Model_Related/InternVLU/InternVL-U}"

echo "=============================================================="
echo " VBVR NEXT-FRAME GEN SFT (no-ce)  |  GPUs=$GPUS ($CUDA_VISIBLE_DEVICES) epochs=$EPOCHS lr=$LR lora=r$LORA_RANK"
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
