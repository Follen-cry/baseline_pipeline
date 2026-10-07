#!/bin/bash
# Route-drawing SFT: edit-mode (VAE-conditioned) fine-tune on data/datasets/maze_dataset/
# maze_route/maze_route_train.jsonl (3000 pairs: plain maze -> maze with GT
# path drawn). Modeled on run_maze_gen_sft.sh, with two changes:
#   - INTERNVLU_VAE_COND=1 always exported (this task needs the pixel-level
#     VAE condition channel; the next-frame maze_gen task deliberately didn't).
#   - INTERNVLU_CKPT is meant to be overridden per run: point it at the base
#     HF snapshot for the "base" arm, or at an already-merged maze_gen
#     checkpoint (e.g. internvlu-maze-gen-merged) to continue-SFT the
#     fixed-stride next-frame checkpoint onto the route-drawing task.
#
# Usage:
#   conda activate internvlu
#   cd Model_Related/InternVLU/InternVL/internvl_chat
#   RUN_TAG=base INTERNVLU_CKPT=<base snapshot> \
#     bash shell/internvlu/sft/v1/run_maze_route_sft.sh
#   RUN_TAG=fixedstride INTERNVLU_CKPT=/scratch/network/ssd/junlin/models/internvlu-maze-gen-merged \
#     bash shell/internvlu/sft/v1/run_maze_route_sft.sh
set -euo pipefail

REPO_ROOT="/scratch/network/ssd2/junlin/ssl_mllm/Model_Related/InternVLU/InternVL/internvl_chat"
cd "$REPO_ROOT"

RUN_TAG="${RUN_TAG:?set RUN_TAG=base|fixedstride (or any short tag) to name the output dir}"

# ---- config (override via env) ----
GPUS="${GPUS:-4}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1e-5}"
LORA_RANK="${LORA_RANK:-32}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GEN_IMAGE_SIZE="${GEN_IMAGE_SIZE:-512}"
GEN_LOSS_WEIGHT="${GEN_LOSS_WEIGHT:-0.5}"
GEN_LOSS_WARMUP="${GEN_LOSS_WARMUP:-20}"
IMGEN_RATIO="${IMGEN_RATIO:-1.0}"

META_PATH="${META_PATH:-/scratch/network/ssd2/junlin/ssl_mllm/data/meta/maze_route_meta.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/network/ssd/junlin/models/internvlu-maze-route-${RUN_TAG}}"
MERGED_DIR="${MERGED_DIR:-${OUTPUT_DIR}-merged}"

BASE_SNAPSHOT="/scratch/network/ssd2/junlin/huggingface/hub/models--InternVL-U--InternVL-U/snapshots/f012d760e69712bb47f7d3d09a24280f346cee01"
INTERNVLU_CKPT="${INTERNVLU_CKPT:?set INTERNVLU_CKPT to the base snapshot or a merged maze_gen checkpoint}"
INTERNVLU_PKG_PATH="${INTERNVLU_PKG_PATH:-/scratch/network/ssd2/junlin/ssl_mllm/Model_Related/InternVLU/InternVL-U}"

echo "=============================================================="
echo " MAZE ROUTE SFT  |  tag=$RUN_TAG  GPUs=$GPUS epochs=$EPOCHS lr=$LR lora=r$LORA_RANK"
echo " start_ckpt : $INTERNVLU_CKPT"
echo " meta   : $META_PATH"
echo " output : $OUTPUT_DIR"
echo " merged : $MERGED_DIR"
echo " INTERNVLU_VAE_COND=1 (edit-mode pixel condition, forced on for this task)"
echo "=============================================================="
mkdir -p "$OUTPUT_DIR"

# ---- 1) train ----
# expandable_segments reclaims the "reserved but unallocated" fragmentation
# slack (~2GB observed) that tipped several marginal-memory GPUs into OOM on
# this shared, multi-tenant cluster.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
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
INTERNVLU_VAE_COND=1 \
bash shell/internvlu/engine/internvlu_4b_sft_full_unified.sh \
    --num_train_epochs "$EPOCHS" \
    --learning_rate "$LR" \
    --use_llm_lora "$LORA_RANK" \
    ${EXTRA_TRAIN_ARGS:-}

# ---- 2) fix the degenerate processor the training save writes ----
if [ -d "$OUTPUT_DIR/processor" ] && [ ! -d "$OUTPUT_DIR/processor_broken_bak" ]; then
    mv "$OUTPUT_DIR/processor" "$OUTPUT_DIR/processor_broken_bak"
fi
cp -rL "$BASE_SNAPSHOT/processor" "$OUTPUT_DIR/processor"
echo "[processor] replaced with base snapshot's (image_processor_kwargs restored)"

# ---- 3) merge LoRA -> inference-ready pipeline ----
# --base-snapshot backfills the custom modeling .py files + tokenizer files
# training's raw output never has (they're not HF "weights", so save_pretrained
# doesn't write them) -- without this, the merged vlm/ dir fails to load at all
# under trust_remote_code=True (hit this exactly when using an earlier merge,
# internvlu-maze-gen-merged, as a further-SFT starting checkpoint).
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
