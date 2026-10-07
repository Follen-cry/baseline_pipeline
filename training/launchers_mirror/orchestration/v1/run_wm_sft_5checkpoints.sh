#!/bin/bash
# Orchestrates the WorldPrediction-WM SFT comparison across all 5 stage-1
# pretrain checkpoints (base InternVL-U + S0/S1/S2/S3/S4-gen-merged): trains
# each with IDENTICAL hyperparameters (only the starting checkpoint differs,
# so results are comparable -- this is the whole point of the run), merges
# LoRA, assembles an eval-ready pipeline dir (tools/assemble_unified.py,
# combining the fine-tuned vlm/ with each source checkpoint's OWN unchanged
# generation_decoder/vae/scheduler/processor -- these differ across S0-S4
# since those stages trained the generation_decoder), then evaluates all 5
# on the reserved COIN-only WM eval split.
#
# Validated end-to-end on 2026-09-11 via a 2-step smoke train + assemble +
# 3-sample eval before launching this full run (see conversation/memory) --
# this is not a first attempt at the pipeline shape, just the full scale-up.
#
# Idempotent/resumable: each phase checks for its expected output and skips
# if already present, so this can be re-run after an interruption without
# redoing finished work. Runs strictly sequentially (one checkpoint at a
# time, one GPU set) rather than in parallel, deliberately -- this is a
# shared multi-tenant node and grad-accumulation makes wall-clock roughly
# invariant to GPU count anyway (halving GPUs ~doubles per-job time), so
# concurrent jobs would mostly just add CPU/IO contention risk for no net
# speedup.
#
# Usage:
#   conda activate internvlu
#   cd baseline_pipeline/training/models/internvl-u/internvl_chat
#   nohup bash shell/internvlu/orchestration/v1/run_wm_sft_5checkpoints.sh > wm_sft_5checkpoints.log 2>&1 &
set -uo pipefail  # NOT -e: we want to continue past a single checkpoint's failure and report it

TRAIN_REPO="/scratch/network/ssd2/junlin/ssl_mllm/baseline_pipeline/training/models/internvl-u/internvl_chat"
EVAL_REPO="/scratch/network/ssd2/junlin/ssl_mllm/baseline_pipeline/eval/suites/worldprediction"
MODELS_DIR="/scratch/network/ssd2/junlin/models"
BASE_SNAPSHOT="/scratch/network/ssd2/junlin/huggingface/hub/models--InternVL-U--InternVL-U/snapshots/f012d760e69712bb47f7d3d09a24280f346cee01"

# ---- SHARED hyperparameters, identical across all 5 runs (the whole point) ----
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,6,7}"
export GPUS="${GPUS:-4}"
export EPOCHS="${EPOCHS:-1}"
export LR="${LR:-4e-5}"
export LORA_RANK="${LORA_RANK:-16}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-12288}"

EVAL_GPU="${EVAL_GPU:-6}"
COIN_EVAL_DATA="/scratch/network/ssd2/junlin/ssl_mllm/baseline_pipeline/data/v1/datasets/worldprediction/WorldPrediction-WM-coin_eval.json"

# name : source checkpoint dir (must contain model_index.json)
declare -A SOURCE_CKPT=(
  [base]="$BASE_SNAPSHOT"
  [s0]="$MODELS_DIR/internvlu-s0-gen-merged"
  [s1]="$MODELS_DIR/internvlu-s1-gen-merged"
  [s2]="$MODELS_DIR/internvlu-s2-gen-merged"
  [s3]="$MODELS_DIR/internvlu-s3-gen-merged"
  [s4]="$MODELS_DIR/internvlu-s4-gen-merged"
)
NAMES=(base s0 s1 s2 s3 s4)
# NOTE: 6 entries above minus... there are exactly 6 names but the task says
# "5 models (base and S0-S4)" -- base + S0,S1,S2,S3,S4 = 6 total checkpoints.

echo "=================================================================="
echo " WorldPrediction-WM SFT x6 checkpoint comparison (base + S0-S4)"
echo " Shared hyperparams: GPUs=$GPUS epochs=$EPOCHS lr=$LR lora=r$LORA_RANK"
echo "   effective_batch=$BATCH_SIZE per_device=$PER_DEVICE_BATCH_SIZE max_seq_len=$MAX_SEQ_LENGTH"
echo "=================================================================="

FAILED=()

# ---- Phase 1: train + merge + assemble, one checkpoint at a time ----
cd "$TRAIN_REPO"
for name in "${NAMES[@]}"; do
  src="${SOURCE_CKPT[$name]}"
  output_dir="$MODELS_DIR/internvlu-wm-sft-$name"
  merged_dir="${output_dir}-merged"
  pipeline_dir="${merged_dir}-pipeline"

  echo ""
  echo "------------------------------------------------------------"
  echo " [$name] source=$src"
  echo "------------------------------------------------------------"

  if [ -d "$pipeline_dir/vlm" ]; then
    echo "[$name] pipeline dir already exists, skipping (resumable run)."
    continue
  fi

  if [ ! -d "$merged_dir" ]; then
    echo "[$name] training..."
    INTERNVLU_CKPT="$src/vlm" \
    OUTPUT_DIR="$output_dir" \
    MERGED_DIR="$merged_dir" \
    BASE_SNAPSHOT="$BASE_SNAPSHOT" \
    bash shell/internvlu/sft/v1/run_worldprediction_wm_sft.sh
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "[$name] TRAINING FAILED (exit $rc) -- skipping to next checkpoint."
      FAILED+=("$name:train")
      continue
    fi
  else
    echo "[$name] merged dir already exists, skipping training."
  fi

  echo "[$name] assembling eval-ready pipeline dir..."
  python tools/assemble_unified.py \
    --vlm "$merged_dir" \
    --snapshot "$src" \
    --output "$pipeline_dir"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$name] ASSEMBLE FAILED (exit $rc) -- skipping to next checkpoint."
    FAILED+=("$name:assemble")
    continue
  fi
  echo "[$name] done: $pipeline_dir"
done

# ---- Phase 2: COIN-only WM eval for every successfully assembled checkpoint ----
cd "$EVAL_REPO"
mkdir -p results
for name in "${NAMES[@]}"; do
  pipeline_dir="$MODELS_DIR/internvlu-wm-sft-${name}-merged-pipeline"
  result_dir="results/wm_sft_${name}"
  result_path="$result_dir/WM_results.json"

  if [ ! -d "$pipeline_dir/vlm" ]; then
    echo "[$name] no pipeline dir -- skipping eval (training/assemble must have failed)."
    continue
  fi
  if [ -f "$result_path" ]; then
    echo "[$name] eval result already exists, skipping."
    continue
  fi

  mkdir -p "$result_dir"
  echo ""
  echo "[$name] evaluating on COIN-only WM split (GPU $EVAL_GPU)..."
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" python run.py \
    --task WM \
    --data "$COIN_EVAL_DATA" \
    --start_index 0 --end_index -1 \
    --model_class VLM \
    --model_config "configs/vlm/internvlu/InternVL-U-wm-sft-${name}-4f.json" \
    --output_path "$result_path"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$name] EVAL FAILED (exit $rc)."
    FAILED+=("$name:eval")
  fi
done

# ---- Summary ----
echo ""
echo "=================================================================="
echo " SUMMARY"
echo "=================================================================="
python3 - "$EVAL_REPO" "${NAMES[@]}" <<'PY'
import json, os, sys
eval_repo, names = sys.argv[1], sys.argv[2:]
print(f"{'checkpoint':<8} {'mean_accuracy':>14} {'num_evaluated':>14}")
for name in names:
    p = os.path.join(eval_repo, f"results/wm_sft_{name}/WM_results.json")
    if not os.path.isfile(p):
        print(f"{name:<8} {'MISSING':>14} {'':>14}")
        continue
    d = json.load(open(p))
    print(f"{name:<8} {d.get('mean_accuracy'):>14.4f} {d.get('num_evaluated_samples'):>14}")
PY

if [ ${#FAILED[@]} -gt 0 ]; then
  echo ""
  echo "FAILURES: ${FAILED[*]}"
  exit 1
fi
echo ""
echo "ALL DONE."
