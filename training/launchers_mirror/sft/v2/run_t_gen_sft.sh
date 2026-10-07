#!/bin/bash
# Shared v2 (temporal SSL, T0-T4) launcher: train InternVL-U next-frame / masked-frame generation SFT
# on one setting, with in-training validation on that setting's evalmini rows, then fix the saved
# processor and merge the LoRA into an inference-ready pipeline. Called by run_t{0..4}_gen_sft.sh,
# which set SETTING (and LM_LOSS_WEIGHT / VAL_EVAL_JSONL defaults); see baseline_pipeline/docs/v2.md.
#
# Recipe = v1's run_s4_gen_sft.sh: full-unified (generation_decoder + LLM-LoRA r32 + mlp1), 1 epoch,
# lr 1e-5, imgen_ratio 1.0, gen_loss_weight 0.5 (warmup 20), global batch 16 (1 per GPU x 8 GPUs x
# accumulation 2) -> 60,000 rows = 3,750 steps. max_dynamic_patch 3 comes from the meta entry.
# INTERNVLU_VAE_COND=1: the VAE pixel condition is each row's cond_image (the shown frame nearest
# the target).
#
# One deliberate change from v1 (2026-09-26): the generation target and cond_image keep their aspect
# ratio at a fixed area ~512^2 (GEN_RESIZE_MODE=area, GEN_IMAGE_SIZE=512: scaled up or down, sides
# rounded to x16, e.g. 16:9 -> 688x384, 1:1 -> 512x512) -- v1 squashed every target to 512x512, so
# the pixel budget is v1's without the distortion. The 3 context frames still go to the ViT as one
# 448x448 tile each, as in v1 and the base model's own processor. area needs 1 row per GPU.
#
# LM_LOSS_WEIGHT: 0.0 for T0 / T2 (answer is <img> only), 0.5 for T1 / T3 / T4 (JSON answer line).
#
# Validation (internvl/train/v2_validation.py): 6 runs per training by default (step 0, every
# VAL_STEPS = total/5 steps, end; see VAL_INTERVALS below) on VAL_EVAL_JSONL (T4 also validates on
# T0_evalmini = its T4-A half); ~2 min per set on 8 GPUs.
# Logged as val/<T>/... to tensorboard + wandb (project WANDB_PROJECT, run WANDB_RUN_NAME).
#
# Runs from the internvl-u submodule's internvl_chat (NOT Model_Related/InternVLU/InternVL, which
# lacks the v2 validation code). Use the `internvlu` conda env (torch 2.6 / NCCL 2.21.5), on a node
# with free GPUs (check nvidia-smi -- shared cluster).
#
# Override anything via env, e.g.  GPUS=8 VAL_STEPS=500 bash run_t3_gen_sft.sh
# Smoke test:  SMOKE=1 GPUS=2 bash run_t3_gen_sft.sh   (4 steps, validation at steps 0/2/4 on the
#              first VAL_SMOKE_ROWS rows, throwaway output dir, no merge unless SMOKE_MERGE=1)
set -euo pipefail

: "${SETTING:?set SETTING=T0..T4 (use run_t{N}_gen_sft.sh)}"
N="${SETTING#T}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"   # .../internvl_chat
BP_ROOT="$(cd "$REPO_ROOT/../../../.." && pwd)"                           # .../baseline_pipeline
cd "$REPO_ROOT"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---- config (override via env) ----
GPUS="${GPUS:-8}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1e-5}"
LORA_RANK="${LORA_RANK:-32}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GEN_IMAGE_SIZE="${GEN_IMAGE_SIZE:-512}"
GEN_RESIZE_MODE="${GEN_RESIZE_MODE:-area}"
GEN_LOSS_WEIGHT="${GEN_LOSS_WEIGHT:-0.5}"
GEN_LOSS_WARMUP="${GEN_LOSS_WARMUP:-20}"
IMGEN_RATIO="${IMGEN_RATIO:-1.0}"
: "${LM_LOSS_WEIGHT:?set by run_t{N}_gen_sft.sh}"
export INTERNVLU_VAE_COND="${INTERNVLU_VAE_COND:-1}"

RUN="${RUN:-main}"
META_PATH="${META_PATH:-$BP_ROOT/data/v2/meta/$RUN/${SETTING}_train_meta.json}"
EVAL_DIR="$BP_ROOT/data/v2/datasets/temporal_ssl/settings/$RUN"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/network/ssd/junlin/models/internvlu-v2-t${N}-gen}"
MERGED_DIR="${MERGED_DIR:-${OUTPUT_DIR}-merged}"
BASE_SNAPSHOT="${BASE_SNAPSHOT:-/scratch/network/ssd2/junlin/huggingface/hub/models--InternVL-U--InternVL-U/snapshots/f012d760e69712bb47f7d3d09a24280f346cee01}"
INTERNVLU_CKPT="${INTERNVLU_CKPT:-$BASE_SNAPSHOT}"
INTERNVLU_PKG_PATH="${INTERNVLU_PKG_PATH:-/scratch/network/ssd2/junlin/ssl_mllm/Model_Related/InternVLU/InternVL-U}"
MASTER_PORT="${MASTER_PORT:-$((34230 + N))}"

# validation (VAL_SETS: space list of settings whose evalmini to use; default set by the wrapper)
VAL_SETS="${VAL_SETS:-$SETTING}"
if [ -z "${VAL_EVAL_JSONL:-}" ]; then
    VAL_EVAL_JSONL=""
    for s in $VAL_SETS; do VAL_EVAL_JSONL="${VAL_EVAL_JSONL:+$VAL_EVAL_JSONL,}$EVAL_DIR/${s}_evalmini.jsonl"; done
fi
# VAL_STEPS default: ceil(total optimizer steps / VAL_INTERVALS) -> validation at step 0, at
# (VAL_INTERVALS - 1) evenly spaced steps and at the end = VAL_INTERVALS + 1 runs (default 6; the
# callback skips a repeat when the last multiple is also the final step). T*: 60,000 rows / 16 =
# 3,750 steps -> every 750: 0 / 750 / 1500 / 2250 / 3000 / 3750.
VAL_INTERVALS="${VAL_INTERVALS:-5}"
if [ -z "${VAL_STEPS:-}" ]; then
    TOTAL_STEPS=$(python - "$META_PATH" "$EPOCHS" "$BATCH_SIZE" <<'PY'
import json, math, sys
meta, epochs, batch = json.load(open(sys.argv[1])), float(sys.argv[2]), int(sys.argv[3])
rows = sum(int(e["length"] * e.get("repeat_time", 1)) for e in meta.values())
print(max(1, math.floor(rows * epochs / batch)))
PY
)
    VAL_STEPS=$(( (TOTAL_STEPS + VAL_INTERVALS - 1) / VAL_INTERVALS ))
fi
export VAL_STEPS

export REPORT_TO="${REPORT_TO:-tensorboard wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-internvlu-v2}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-t${N}-gen}"

EXTRA="${EXTRA_TRAIN_ARGS:-}"
DO_MERGE=1
if [ "${SMOKE:-0}" = "1" ]; then
    OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-/scratch/network/ssd/junlin/models/_smoke_v2_t${N}}"
    MERGED_DIR="${OUTPUT_DIR}-merged"
    rm -rf "$OUTPUT_DIR" "$MERGED_DIR"
    SMOKE_ROWS="${VAL_SMOKE_ROWS:-4}"
    mkdir -p "$OUTPUT_DIR/_smoke_eval"
    smoke_eval=""
    for f in ${VAL_EVAL_JSONL//,/ }; do
        head -n "$SMOKE_ROWS" "$f" > "$OUTPUT_DIR/_smoke_eval/$(basename "$f")"
        smoke_eval="${smoke_eval:+$smoke_eval,}$OUTPUT_DIR/_smoke_eval/$(basename "$f")"
    done
    VAL_EVAL_JSONL="$smoke_eval"
    export VAL_STEPS=2 VAL_LOG_IMAGES="${VAL_LOG_IMAGES:-2}"
    export WANDB_MODE="${WANDB_MODE:-offline}" WANDB_RUN_NAME="smoke-t${N}"
    EXTRA="--max_steps 4 $EXTRA"
    [ "${SMOKE_MERGE:-0}" = "1" ] || DO_MERGE=0
fi
export VAL_EVAL_JSONL
export WANDB_DIR="${WANDB_DIR:-$OUTPUT_DIR}"

# ---- checks ----
if (( BATCH_SIZE % (PER_DEVICE_BATCH_SIZE * GPUS) != 0 )); then
    echo "BATCH_SIZE=$BATCH_SIZE is not divisible by PER_DEVICE_BATCH_SIZE*GPUS=$((PER_DEVICE_BATCH_SIZE*GPUS));"\
         "the engine would silently train with a smaller global batch. Adjust BATCH_SIZE or GPUS." >&2
    exit 1
fi
for f in "$META_PATH" ${VAL_EVAL_JSONL//,/ }; do
    [ -f "$f" ] || { echo "missing $f -- run data/v2/recipes/temporal_ssl/derive_settings.py" \
                          "and make_eval_mini.py (the jsonl are gitignored)" >&2; exit 1; }
done

echo "=============================================================="
echo " v2 $SETTING GEN SFT  |  GPUs=$GPUS batch=$BATCH_SIZE epochs=$EPOCHS lr=$LR lora=r$LORA_RANK"
echo " gen_loss_weight=$GEN_LOSS_WEIGHT (warmup $GEN_LOSS_WARMUP)  lm_loss_weight=$LM_LOSS_WEIGHT"
echo " gen image: $GEN_RESIZE_MODE, size $GEN_IMAGE_SIZE"
echo " meta   : $META_PATH"
echo " val    : $VAL_EVAL_JSONL  every $VAL_STEPS steps"
echo " wandb  : $WANDB_PROJECT / $WANDB_RUN_NAME   (report_to: $REPORT_TO)"
echo " output : $OUTPUT_DIR"
[ "$DO_MERGE" = "1" ] && echo " merged : $MERGED_DIR"
echo "=============================================================="
mkdir -p "$OUTPUT_DIR"
LOG="$OUTPUT_DIR/training_log.txt"
LOG_START=$( [ -f "$LOG" ] && wc -l < "$LOG" || echo 0 )

# ---- 1) train (full-unified launcher; trailing args override its hardcoded defaults) ----
GPUS="$GPUS" \
BATCH_SIZE="$BATCH_SIZE" \
PER_DEVICE_BATCH_SIZE="$PER_DEVICE_BATCH_SIZE" \
IMGEN_RATIO="$IMGEN_RATIO" \
GEN_IMAGE_SIZE="$GEN_IMAGE_SIZE" \
GEN_LOSS_WEIGHT="$GEN_LOSS_WEIGHT" \
GEN_LOSS_WARMUP="$GEN_LOSS_WARMUP" \
META_PATH="$META_PATH" \
OUTPUT_DIR="$OUTPUT_DIR" \
INTERNVLU_CKPT="$INTERNVLU_CKPT" \
INTERNVLU_PKG_PATH="$INTERNVLU_PKG_PATH" \
MASTER_PORT="$MASTER_PORT" \
bash shell/internvlu/engine/internvlu_4b_sft_full_unified.sh \
    --num_train_epochs "$EPOCHS" \
    --learning_rate "$LR" \
    --use_llm_lora "$LORA_RANK" \
    --lm_loss_weight "$LM_LOSS_WEIGHT" \
    --gen_resize_mode "$GEN_RESIZE_MODE" \
    $EXTRA
# (the engine pipes torchrun through tee, so check the log for success rather than the exit code)
# (no grep -q: with pipefail, an early grep exit can SIGPIPE tail and fail the check spuriously)
tail -n +"$((LOG_START + 1))" "$LOG" | grep "train_runtime" > /dev/null || { echo "training did not finish (see $LOG)" >&2; exit 1; }

[ "$DO_MERGE" = "1" ] || { echo "DONE (no merge). Output: $OUTPUT_DIR"; exit 0; }

# ---- 2) fix the degenerate processor the training save writes ----
# (save_pretrained drops image_processor_kwargs -> InternVLUPipeline load fails with
#  KeyError: 'image_processor_kwargs'. Replace with the base snapshot's.)
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
