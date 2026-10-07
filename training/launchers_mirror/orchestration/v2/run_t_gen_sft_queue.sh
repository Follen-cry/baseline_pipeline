#!/bin/bash
# Queue v2 (temporal SSL) gen-SFT runs one after another on one set of GPUs, via the
# sft/v2/run_t{N}_gen_sft.sh launchers (recipe, validation and LoRA merge live there; see
# baseline_pipeline/docs/v2.md). Default: T0, T2, T3 on 4 GPUs (T1 / T4 not needed yet).
#
# Per setting: wait until the GPUs are free, train (6 evalmini validations) + merge, then delete the
# raw run's weight dirs (the merged dir holds a full copy; logs, val/, runs/, wandb/ are kept).
# Resumable: a setting whose merged dir already has vlm/ is skipped. A failed setting is recorded and
# the queue moves on. Ends with a table of every validation run's key metrics.
#
# 4 GPUs x 1 row x accumulation 4 = global batch 16, 3,750 steps: measured 10.2 s/step on A40s ->
# ~11-12 h per setting incl. validation and merge, ~35 h for T0 T2 T3.
#
# Usage (internvlu env), from anywhere:
#   nohup bash run_t_gen_sft_queue.sh > /scratch/network/ssd/junlin/models/v2_logs/queue.log 2>&1 &
#   SETTINGS="T3" CUDA_VISIBLE_DEVICES=0,1,3,6 bash run_t_gen_sft_queue.sh
#   DRY_RUN=1 bash run_t_gen_sft_queue.sh        # print the plan only
#   SMOKE=1 bash run_t_gen_sft_queue.sh          # 4 steps per setting, throwaway dirs, no merge
# Env: SETTINGS, CUDA_VISIBLE_DEVICES, GPUS, GPU_WAIT_MIN (default 120), GPU_FREE_MIB (default 2000),
#      KEEP_RAW=1 (keep raw weights), LOG_DIR, plus anything run_t_gen_sft.sh reads.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "$HERE/../../sft/v2" && pwd)"
MODELS_DIR="${MODELS_DIR:-/scratch/network/ssd/junlin/models}"
LOG_DIR="${LOG_DIR:-$MODELS_DIR/v2_logs}"

SETTINGS=(${SETTINGS:-T0 T2 T3})
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,3,6}"
export GPUS="${GPUS:-4}"
GPU_WAIT_MIN="${GPU_WAIT_MIN:-120}"
GPU_FREE_MIB="${GPU_FREE_MIB:-2000}"
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"

n_vis=$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | grep -c .)
if [ "$n_vis" -ne "$GPUS" ]; then
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES has $n_vis GPUs but GPUS=$GPUS" >&2
  exit 1
fi

run_dirs() {  # $1 = N -> sets OUT / MERGED (same defaults as run_t_gen_sft.sh)
  if [ "$SMOKE" = "1" ]; then
    OUT="$MODELS_DIR/_smoke_v2_t$1"
  else
    OUT="$MODELS_DIR/internvlu-v2-t$1-gen"
  fi
  MERGED="${OUT}-merged"
}

busy_gpus() {  # prints the listed GPUs using more than GPU_FREE_MIB
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES" \
    | awk -F', ' -v lim="$GPU_FREE_MIB" '$2 > lim {printf "%s(%s MiB) ", $1, $2}'
}

wait_for_gpus() {
  local waited=0 busy
  while busy=$(busy_gpus); [ -n "$busy" ]; do
    if [ "$waited" -ge "$GPU_WAIT_MIN" ]; then
      echo "[queue] GPUs still busy after $GPU_WAIT_MIN min: $busy"
      return 1
    fi
    [ "$waited" -eq 0 ] && echo "[queue] waiting for busy GPUs: $busy"
    sleep 60
    waited=$((waited + 1))
  done
  return 0
}

mkdir -p "$LOG_DIR"
echo "=================================================================="
echo " v2 gen-SFT queue: ${SETTINGS[*]}   $(date '+%F %T')  host $(hostname)"
echo " GPUs: $CUDA_VISIBLE_DEVICES (GPUS=$GPUS)   smoke=$SMOKE dry_run=$DRY_RUN"
echo " per-setting logs: $LOG_DIR/t<N>[_smoke].log"
echo "=================================================================="

FAILED=()
for s in "${SETTINGS[@]}"; do
  N="${s#T}"
  launcher="$SFT_DIR/run_t${N}_gen_sft.sh"
  run_dirs "$N"
  log="$LOG_DIR/t${N}$([ "$SMOKE" = "1" ] && echo _smoke).log"
  echo ""
  echo "------------------------------------------------------------"
  echo " [$s] $(date '+%F %T')  output $OUT"
  echo "------------------------------------------------------------"
  if [ ! -f "$launcher" ]; then
    echo "[$s] no launcher $launcher -- skipping"; FAILED+=("$s:launcher"); continue
  fi
  if [ "$SMOKE" != "1" ] && [ -d "$MERGED/vlm" ]; then
    echo "[$s] merged dir already exists, skipping (resumable run)."; continue
  fi
  if [ "$DRY_RUN" = "1" ]; then
    echo "[$s] would run: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES GPUS=$GPUS SMOKE=$SMOKE bash $launcher > $log"
    continue
  fi
  if [ "$SMOKE" != "1" ] && [ -d "$OUT" ]; then
    echo "[$s] note: $OUT exists from an unfinished run; retraining from scratch (log is appended)."
  fi
  wait_for_gpus || { FAILED+=("$s:gpus_busy"); continue; }

  t0=$(date +%s)
  SMOKE="$SMOKE" bash "$launcher" > "$log" 2>&1
  rc=$?
  mins=$(( ($(date +%s) - t0) / 60 ))
  if [ $rc -ne 0 ]; then
    echo "[$s] FAILED (exit $rc) after ${mins} min -- see $log"; tail -n 5 "$log" | sed 's/^/    /'
    FAILED+=("$s:train"); continue
  fi
  echo "[$s] done in ${mins} min"

  # Disk hygiene: the merged dir is a full, independent copy of the weights.
  if [ "$SMOKE" != "1" ] && [ "${KEEP_RAW:-0}" != "1" ] && [ -d "$MERGED/vlm" ]; then
    rm -rf "$OUT/vlm" "$OUT/generation_decoder" "$OUT/vae"
    echo "[$s] removed raw weight dirs from $OUT (logs, val/, runs/, wandb/ kept)"
  fi
done

[ "$DRY_RUN" = "1" ] && exit 0

echo ""
echo "=================================================================="
echo " VALIDATION SUMMARY (evalmini, rule-based; read dpsnr / beat_copy / dmotion_psnr vs copy)"
echo "=================================================================="
for s in "${SETTINGS[@]}"; do
  run_dirs "${s#T}"
  python3 - "$OUT/val" "$s" <<'PY'
import glob, json, os, sys
root, setting = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(os.path.join(root, "step*", "*", "summary.json")))
if not files:
    print(f"[{setting}] no validation results under {root}")
    sys.exit()
cols = ["dpsnr", "beat_copy", "dmotion_psnr", "psnr", "parse_ok", "gap_acc", "order_exact", "missing_acc"]
print(f"[{setting}]")
print(f"  {'step':>6} {'set':>4}" + "".join(f"{c:>13}" for c in cols))
for f in files:
    m = json.load(open(f))["metrics"]
    step = os.path.basename(os.path.dirname(os.path.dirname(f)))[4:]
    name = os.path.basename(os.path.dirname(f))
    cells = "".join(f"{m[f'all/{c}']:>13.3f}" if f"all/{c}" in m else f"{'-':>13}" for c in cols)
    print(f"  {int(step):>6} {name:>4}{cells}")
PY
done

if [ ${#FAILED[@]} -gt 0 ]; then
  echo ""
  echo "FAILURES: ${FAILED[*]}"
  exit 1
fi
echo ""
echo "ALL DONE $(date '+%F %T')."
