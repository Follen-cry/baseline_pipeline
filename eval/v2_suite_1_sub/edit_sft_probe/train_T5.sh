#!/bin/bash
# Launch the edit_sft_probe for `T5`. See run_probe_sft.sh for the recipe.
#   GPUS=4 bash train_T5.sh
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
CKPT=$(python3 "$D/../scripts/common.py" models.T5)
MODEL=T5 CKPT="$CKPT" GPUS="${GPUS:-4}" bash "$D/run_probe_sft.sh"
