#!/bin/bash
# Inference + judging + scoring for the 4 edit_sft_probe checkpoints (base_probe/T0_probe/T2_probe/
# T3_probe) on v2_suite_1_sub's 200-item manifest, reusing the parent suite's existing
# gen.py / run_judge.sh / score.py unchanged (config-driven, from ../config.json). These probe
# checkpoints are NOT added to config.models/report_models (ad hoc axis, like base_ft/T0_ft/T2_ft/
# T3_ft) -- score.py's auto-detection still picks them up into table.csv/scores_item.csv/coverage.csv
# once judged.
#
# Checkpoints must already exist under /scratch/network/ssd/junlin/models/edit_sft_probe_sub/
# (run train_{base,T0,T2,T3}.sh first, see README.md).
#
# Usage (env: internvlu for gen/judge-client/score; vllm for the judge server):
#   cd edit_sft_probe
#   ./run_probe_eval.sh gen                                  # inference, 4 models, one GPU, foreground
#   ../scripts/serve_judge.sh <gpu> <port>                    # on a GPU node, separately
#   EP=http://<node>:<port>/v1 ./run_probe_eval.sh judge      # judge (4 benches) + score, 4 models
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SCRIPTS="$HERE/../scripts"
PY=$(python3 "$SCRIPTS/common.py" envs.internvlu_python)
MODELS_ROOT=/scratch/network/ssd/junlin/models/edit_sft_probe_sub

declare -A CKPTS=(
  [base_probe]="$MODELS_ROOT/base-merged"
  [T0_probe]="$MODELS_ROOT/T0-merged"
  [T2_probe]="$MODELS_ROOT/T2-merged"
  [T3_probe]="$MODELS_ROOT/T3-merged"
)
NAMES="base_probe,T0_probe,T2_probe,T3_probe"

cmd="${1:?usage: run_probe_eval.sh gen|judge}"

case "$cmd" in
  gen)
    for name in base_probe T0_probe T2_probe T3_probe; do
      ckpt="${CKPTS[$name]}"
      if [ ! -d "$ckpt" ]; then
        echo "skip $name: $ckpt not found (run train_${name%_probe}.sh first)" >&2
        continue
      fi
      echo "=== generating $name from $ckpt ==="
      (cd "$SCRIPTS" && CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" gen.py --model "$name" --model_path "$ckpt")
    done
    ;;
  judge)
    : "${EP:?set EP=http://host:port/v1[,...] (see ../scripts/serve_judge.sh)}"
    (cd "$SCRIPTS" && EP="$EP" ./run_judge.sh "$NAMES")
    ;;
  *)
    echo "usage: run_probe_eval.sh gen|judge" >&2
    exit 1
    ;;
esac
