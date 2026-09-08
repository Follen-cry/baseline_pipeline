#!/bin/bash
# Unified evaluation for the four InternVL-U checkpoints on the spatial MCQ
# benchmarks, using the same prompt / answer-extraction treatment that VLMEvalKit
# applies to other models (InternVL, Qwen2.5-VL).
#
# The InternVLU adapter now implements use_custom_prompt/build_prompt, so each
# benchmark prompt carries the standard "Answer with the option's letter from the
# given choices directly." instruction. This makes the four checkpoints comparable
# both to each other and to other models on the VLMEvalKit leaderboard.
#
# Notes
#  * Writes to a FRESH work-dir so stale predictions produced with the old bare
#    prompt are not reused. Override with WORK_DIR=... if desired.
#  * For letter-MCQ the rule-based extractor (exact_matching) is deterministic,
#    free, and equivalent to the gpt-4o-mini default once the model emits a clean
#    letter. To reproduce the framework default instead, set JUDGE=gpt-4o-mini and
#    export OPENAI_API_KEY.
set -x
set -e

cd "$(dirname "$0")/.."

MODELS="${MODELS:-InternVL-U-base InternVL-U-ffs InternVL-U-gen1ep InternVL-U-gen3ep}"
DATA="${DATA:-CV-Bench-2D CV-Bench-3D RealWorldQA}"
WORK_DIR="${WORK_DIR:-outputs_internvlu_fixed}"
JUDGE="${JUDGE:-exact_matching}"
GPU="${GPU:-$(nvidia-smi --list-gpus | wc -l)}"

torchrun --nproc-per-node="${GPU}" run.py \
    --model ${MODELS} \
    --data ${DATA} \
    --judge "${JUDGE}" \
    --work-dir "${WORK_DIR}" \
    --verbose
