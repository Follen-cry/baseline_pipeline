#!/bin/bash
# Launch the edit_sft_probe for `T0`. See run_probe_sft.sh for the recipe.
#   GPUS=4 bash train_T0.sh
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
CKPT=$(python3 "$D/../scripts/common.py" models.T0)
MODEL=T0 CKPT="$CKPT" GPUS="${GPUS:-4}" bash "$D/run_probe_sft.sh"
