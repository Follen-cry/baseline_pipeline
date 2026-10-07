#!/bin/bash
# Launch the edit_sft_probe for `T6`. See run_probe_sft.sh for the recipe.
#   GPUS=4 bash train_T6.sh
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
CKPT=$(python3 "$D/../scripts/common.py" models.T6)
MODEL=T6 CKPT="$CKPT" GPUS="${GPUS:-4}" bash "$D/run_probe_sft.sh"
