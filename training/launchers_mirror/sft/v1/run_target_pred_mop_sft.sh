#!/bin/bash
# Target-frame-prediction SFT on multi_object_placement (VBVR target_pred ID
# task, 1-in -> 1-out: first_frame -> final_frame), 2000-row train split.
#
# No-CE: uses the new trainer-level --lm_loss_weight flag (see
# modeling_internvlu_unified.py configure_lm_loss_weight / forward()) set to
# 0.0, NOT the data-side "<img>"-only-caption trick used by the next_frame
# pipeline. target_pred rows keep their fixed GPT-turn caption ("The final
# image should look like this: <img>"); the trainer just zeroes that CE term's
# contribution to the gradient instead of relying on the data collapsing it.
#
# SAFETY: INTERNVLU_CKPT is loaded read-only (--model_name_or_path); nothing
# is ever written back into it. OUTPUT_DIR/MERGED_DIR must always be new,
# distinct directories -- this is what lets us continue-train from the
# existing internvlu-vbvr-gen-noce-merged checkpoint without touching it.
#
# Usage (base-checkpoint start, on torrnode11 GPU3+GPU5):
#   conda activate internvlu
#   cd /scratch/network/ssd2/junlin/ssl_mllm/Model_Related/InternVLU/InternVL/internvl_chat
#   CUDA_VISIBLE_DEVICES=3,5 GPUS=2 \
#   OUTPUT_DIR=/scratch/network/ssd/junlin/models/internvlu-tp-mop-base-1ep-noce \
#   bash shell/internvlu/sft/v1/run_target_pred_mop_sft.sh
#
# Usage (continue from the 100k next_frame-SFT checkpoint, on torrnode9 GPU0+GPU1):
#   CUDA_VISIBLE_DEVICES=0,1 GPUS=2 \
#   INTERNVLU_CKPT=/scratch/network/ssd/junlin/models/internvlu-vbvr-gen-noce-merged \
#   OUTPUT_DIR=/scratch/network/ssd/junlin/models/internvlu-tp-mop-sftckpt-1ep-noce \
#   bash shell/internvlu/sft/v1/run_target_pred_mop_sft.sh
set -euo pipefail

REPO_ROOT="/scratch/network/ssd2/junlin/ssl_mllm/Model_Related/InternVLU/InternVL/internvl_chat"
cd "$REPO_ROOT"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---- GPU selection (override per node/run -- see usage above) ----
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?must set CUDA_VISIBLE_DEVICES explicitly, e.g. 3,5}"
GPUS="${GPUS:-2}"

# ---- config (override via env) ----
EPOCHS="${EPOCHS:-1}"                 # sanity pass; 2000 rows / batch 16 = 125 steps/epoch
LR="${LR:-1e-5}"
LORA_RANK="${LORA_RANK:-32}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GEN_IMAGE_SIZE="${GEN_IMAGE_SIZE:-512}"
GEN_LOSS_WEIGHT="${GEN_LOSS_WEIGHT:-0.5}"
GEN_LOSS_WARMUP="${GEN_LOSS_WARMUP:-20}"
IMGEN_RATIO="${IMGEN_RATIO:-1.0}"      # every row here is a target_pred (imgen) row
LM_LOSS_WEIGHT="${LM_LOSS_WEIGHT:-0.0}"  # No-CE: zero text-CE gradient contribution

# target_pred rows carry a single input frame; VAE-cond defaults to that frame
# via dataset_unified.py's MultimodalImgenLazyDataset (same convention as the
# next_frame no-ce script).
export INTERNVLU_VAE_COND="${INTERNVLU_VAE_COND:-1}"

META_PATH="${META_PATH:-/scratch/network/ssd2/junlin/ssl_mllm/data/meta/vbvr_target_pred_id_mop_meta.json}"
OUTPUT_DIR="${OUTPUT_DIR:?must set OUTPUT_DIR explicitly -- never point this at an existing checkpoint}"
MERGED_DIR="${MERGED_DIR:-${OUTPUT_DIR}-merged}"
BASE_SNAPSHOT="${BASE_SNAPSHOT:-/scratch/network/ssd2/junlin/huggingface/hub/models--InternVL-U--InternVL-U/snapshots/f012d760e69712bb47f7d3d09a24280f346cee01}"
INTERNVLU_CKPT="${INTERNVLU_CKPT:-$BASE_SNAPSHOT}"
INTERNVLU_PKG_PATH="${INTERNVLU_PKG_PATH:-/scratch/network/ssd2/junlin/ssl_mllm/Model_Related/InternVLU/InternVL-U}"

if [ "$(readlink -f "$OUTPUT_DIR")" = "$(readlink -f "$INTERNVLU_CKPT")" ]; then
    echo "REFUSING: OUTPUT_DIR == INTERNVLU_CKPT -- this would overwrite the source checkpoint." >&2
    exit 1
fi

echo "=============================================================="
echo " TARGET_PRED multi_object_placement SFT (No-CE, lm_loss_weight=$LM_LOSS_WEIGHT)"
echo " GPUs=$GPUS ($CUDA_VISIBLE_DEVICES) epochs=$EPOCHS lr=$LR lora=r$LORA_RANK"
echo " init ckpt (read-only) : $INTERNVLU_CKPT"
echo " meta   : $META_PATH"
echo " output : $OUTPUT_DIR"
echo " merged : $MERGED_DIR"
echo "=============================================================="
mkdir -p "$OUTPUT_DIR"

# ---- 1) train ----
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
if [ -d "$OUTPUT_DIR/processor" ] && [ ! -d "$OUTPUT_DIR/processor_broken_bak" ]; then
    mv "$OUTPUT_DIR/processor" "$OUTPUT_DIR/processor_broken_bak"
fi
cp -rL "$BASE_SNAPSHOT/processor" "$OUTPUT_DIR/processor"
echo "[processor] replaced with base snapshot's (image_processor_kwargs restored)"

# ---- 3) merge LoRA -> inference-ready pipeline (never touches INTERNVLU_CKPT) ----
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
echo "Source checkpoint untouched: $INTERNVLU_CKPT"
