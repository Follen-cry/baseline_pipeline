#!/bin/bash
# "uniform data split regarding data source" experiment: same WM-SFT training
# recipe as run_wm_sft_5checkpoints.sh (run_worldprediction_wm_sft.sh,
# identical hyperparameters), but pointed at the new
# data/v1/recipes/uniform_data_split_regarding_data_source/ meta (row-level
# 80/20 split, uniform across all 4 sources, replacing the original COIN-only
# eval split) -- to isolate whether the data split itself was limiting the
# prior experiment's results. First pass: only base, S0, S2 (not the full
# S0-S4 sweep).
set -uo pipefail

TRAIN_REPO="/scratch/network/ssd2/junlin/ssl_mllm/baseline_pipeline/training/models/internvl-u/internvl_chat"
EVAL_REPO="/scratch/network/ssd2/junlin/ssl_mllm/baseline_pipeline/eval/suites/worldprediction"
MODELS_DIR="/scratch/network/ssd2/junlin/models"
BASE_SNAPSHOT="/scratch/network/ssd2/junlin/huggingface/hub/models--InternVL-U--InternVL-U/snapshots/f012d760e69712bb47f7d3d09a24280f346cee01"

# FRAMES selects which data build to use (matches build_worldprediction_wm.py
# / build_uniform_split.py's --frames): 4 (default) reads the original
# unsuffixed meta/dataset files; any other value (e.g. 8) reads the _{n}f
# suffixed ones and uses distinctly-named checkpoints/eval configs/results so
# a 4f and 8f run never collide or silently overwrite each other.
FRAMES="${FRAMES:-4}"
if [ "$FRAMES" = "4" ]; then
  SUFFIX=""
  META_PATH="/scratch/network/ssd2/junlin/ssl_mllm/baseline_pipeline/data/v1/meta/uniform_data_split_regarding_data_source_train_meta.json"
  UNIFORM_EVAL_DATA="/scratch/network/ssd2/junlin/ssl_mllm/baseline_pipeline/data/v1/datasets/uniform_data_split_regarding_data_source/WorldPrediction-WM-uniform_eval.json"
else
  SUFFIX="-${FRAMES}f"
  META_PATH="/scratch/network/ssd2/junlin/ssl_mllm/baseline_pipeline/data/v1/meta/uniform_data_split_regarding_data_source_${FRAMES}f_train_meta.json"
  UNIFORM_EVAL_DATA="/scratch/network/ssd2/junlin/ssl_mllm/baseline_pipeline/data/v1/datasets/uniform_data_split_regarding_data_source_${FRAMES}f/WorldPrediction-WM-uniform_eval.json"
fi

# ---- SHARED hyperparameters, identical to the original worldprediction experiment ----
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,6,7}"
export GPUS="${GPUS:-4}"
export EPOCHS="${EPOCHS:-1}"
export LR="${LR:-4e-5}"
export LORA_RANK="${LORA_RANK:-16}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-12288}"

EVAL_GPU="${EVAL_GPU:-2}"

declare -A SOURCE_CKPT=(
  [base]="$BASE_SNAPSHOT"
  [s0]="$MODELS_DIR/internvlu-s0-gen-merged"
  [s1]="$MODELS_DIR/internvlu-s1-gen-merged"
  [s2]="$MODELS_DIR/internvlu-s2-gen-merged"
  [s3]="$MODELS_DIR/internvlu-s3-gen-merged"
  [s4]="$MODELS_DIR/internvlu-s4-gen-merged"
)
NAMES=(${WM_NAMES:-base s0 s1 s2 s3 s4})

echo "=================================================================="
echo " uniform data split regarding data source -- ${NAMES[*]} (frames=$FRAMES)"
echo " Shared hyperparams: GPUs=$GPUS epochs=$EPOCHS lr=$LR lora=r$LORA_RANK"
echo "   effective_batch=$BATCH_SIZE per_device=$PER_DEVICE_BATCH_SIZE max_seq_len=$MAX_SEQ_LENGTH"
echo "   meta=$META_PATH"
echo "=================================================================="

FAILED=()

cd "$TRAIN_REPO"
for name in "${NAMES[@]}"; do
  src="${SOURCE_CKPT[$name]}"
  output_dir="$MODELS_DIR/internvlu-wm-sft-uniform${SUFFIX}-$name"
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
    META_PATH="$META_PATH" \
    bash shell/internvlu/sft/v1/run_worldprediction_wm_sft.sh
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "[$name] TRAINING FAILED (exit $rc) -- skipping to next checkpoint."
      FAILED+=("$name:train")
      continue
    fi
    # Disk hygiene: this node's shared disk is often near-full. The raw
    # LoRA-only output_dir is fully redundant once merged (merge_lora_u.py
    # already wrote an independent copy to merged_dir) -- delete it right
    # away rather than let 6 checkpoints' worth of raw+merged+pipeline pile
    # up simultaneously.
    rm -rf "$output_dir"
  else
    echo "[$name] merged dir already exists, skipping training."
  fi

  echo "[$name] assembling eval-ready pipeline dir..."
  # --copy-vlm-as-symlink: symlink the merged vlm/ into the pipeline dir
  # instead of copying it again -- same disk-hygiene reasoning as above.
  # (Fine as long as merged_dir isn't deleted before the pipeline is done
  # with it -- both get cleaned up together at the end of this experiment.)
  python tools/assemble_unified.py --vlm "$merged_dir" --snapshot "$src" --output "$pipeline_dir" --copy-vlm-as-symlink
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$name] ASSEMBLE FAILED (exit $rc) -- skipping to next checkpoint."
    FAILED+=("$name:assemble")
    continue
  fi
  echo "[$name] done: $pipeline_dir"
done

cd "$EVAL_REPO"
mkdir -p results
for name in "${NAMES[@]}"; do
  pipeline_dir="$MODELS_DIR/internvlu-wm-sft-uniform${SUFFIX}-${name}-merged-pipeline"
  result_dir="results/wm_sft_uniform${SUFFIX}_${name}"
  result_path="$result_dir/WM_results.json"

  if [ ! -d "$pipeline_dir/vlm" ]; then
    echo "[$name] no pipeline dir -- skipping eval."
    continue
  fi
  if [ -f "$result_path" ]; then
    echo "[$name] eval result already exists, skipping."
    continue
  fi

  mkdir -p "$result_dir"
  echo ""
  echo "[$name] evaluating on uniform-split eval set (GPU $EVAL_GPU)..."
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" python run.py \
    --task WM \
    --data "$UNIFORM_EVAL_DATA" \
    --start_index 0 --end_index -1 \
    --model_class VLM \
    --model_config "configs/vlm/internvlu/InternVL-U-wm-sft-uniform-${name}-${FRAMES}f.json" \
    --output_path "$result_path"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$name] EVAL FAILED (exit $rc)."
    FAILED+=("$name:eval")
  fi
done

echo ""
echo "=================================================================="
echo " SUMMARY"
echo "=================================================================="
python3 - "$EVAL_REPO" "$SUFFIX" "${NAMES[@]}" <<'PY'
import json, os, sys
eval_repo, suffix, names = sys.argv[1], sys.argv[2], sys.argv[3:]
print(f"{'checkpoint':<8} {'mean_accuracy':>14} {'num_evaluated':>14}")
for name in names:
    p = os.path.join(eval_repo, f"results/wm_sft_uniform{suffix}_{name}/WM_results.json")
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
