#!/bin/bash
# IntPhys2 plausibility (Yes/No) LoRA SFT on 3 base models x {8f,16f}.
# Phase A: zero-shot baseline eval of base/gen/ffs on the held-out 200.
# Phase B: understanding-LoRA finetune -> merge -> assemble -> eval, per config.
set -uo pipefail
cd /scratch/network/ssd2/junlin/ssl_mllm/InternVL/internvl_chat
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

TRAIN_DEVS="${TRAIN_DEVS:-1,2,3,7}"; TRAIN_GPUS="${TRAIN_GPUS:-4}"
EVAL_DEV="${EVAL_DEV:-1}"
EPOCHS="${EPOCHS:-3}"; BATCH_SIZE="${BATCH_SIZE:-16}"; LR="${LR:-4e-5}"

BASE_SNAP=/homes/55/junlin/.cache/huggingface/hub/models--InternVL-U--InternVL-U/snapshots/f012d760e69712bb47f7d3d09a24280f346cee01
declare -A CKPT=(
  [base]="$BASE_SNAP"
  [gen]="/scratch/network/ssd2/junlin/models/internvlu-intphys2-gen-c1to2s-merged"
  [ffs]="/scratch/network/ssd2/junlin/models/internvlu-intphys2-ffs-mcq-merged"
  [vaecond]="/scratch/network/ssd2/junlin/models/internvlu-intphys2-gen-c1to2s-vaecond-merged"
)
DATA=/scratch/network/ssd2/junlin/ssl_mllm/data/datasets/intphy2/intphys2_plausibility
OUTROOT=/scratch/network/ssd2/junlin/models/plaus
RES=$DATA/results
mkdir -p "$OUTROOT" "$RES"
EVAL=/scratch/network/ssd2/junlin/ssl_mllm/InternVL-U/run_plausibility_eval.py

echo "############ PHASE A: zero-shot baselines ($(date)) ############"
for n in 8 16; do
  for m in base gen ffs vaecond; do
    o="$RES/zs_${m}_${n}f.json"
    [ -f "$o" ] && { echo "skip $o"; continue; }
    echo "== zero-shot $m ${n}f =="
    CUDA_VISIBLE_DEVICES=$EVAL_DEV python "$EVAL" "${CKPT[$m]}" \
       "$DATA/plausibility_${n}f_eval.jsonl" "$o" "zs_${m}_${n}f"
  done
done

echo "############ PHASE B: finetune x6 ($(date)) ############"
for n in 8 16; do
  for m in base gen ffs vaecond; do
    tag="plaus_${m}_${n}f"
    RAW="$OUTROOT/${tag}-raw"; VLM="$OUTROOT/${tag}-vlm"; PIPE="$OUTROOT/${tag}"
    o="$RES/ft_${m}_${n}f.json"
    [ -f "$o" ] && { echo "skip done $tag"; continue; }
    echo "===== [$tag] train ($(date)) ====="
    CUDA_VISIBLE_DEVICES=$TRAIN_DEVS GPUS=$TRAIN_GPUS \
    INTERNVLU_CKPT="${CKPT[$m]}/vlm" \
    META_PATH="$DATA/plausibility_${n}f_meta.json" \
    OUTPUT_DIR="$RAW" BATCH_SIZE=$BATCH_SIZE LEARNING_RATE=$LR EPOCHS=$EPOCHS \
    bash shell/internvlu/legacy/train_plausibility.sh
    [ -f "$RAW/config.json" ] || { echo "FATAL: $tag training failed"; continue; }

    echo "===== [$tag] merge + assemble ====="
    python tools/merge_lora_u.py "$RAW" "$VLM" || { echo "FATAL merge $tag"; continue; }
    python tools/assemble_unified.py --vlm "$VLM" --output "$PIPE" --snapshot "${CKPT[$m]}" \
       || { echo "FATAL assemble $tag"; continue; }

    echo "===== [$tag] eval ====="
    CUDA_VISIBLE_DEVICES=$EVAL_DEV python "$EVAL" "$PIPE" \
       "$DATA/plausibility_${n}f_eval.jsonl" "$o" "ft_${m}_${n}f"

    # reclaim disk: drop raw + intermediate vlm, keep assembled pipeline
    rm -rf "$RAW" "$VLM"
  done
done
echo "############ ALL DONE ($(date)) ############"
