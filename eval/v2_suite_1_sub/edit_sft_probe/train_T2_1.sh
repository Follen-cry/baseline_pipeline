#!/bin/bash
# Launch the edit_sft_probe for `T2_1` (training setting T2.1). See run_probe_sft.sh for the recipe.
#   GPUS=4 bash train_T2_1.sh
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
CKPT=$(python3 "$D/../scripts/common.py" models.T2_1)
MODEL=T2_1 CKPT="$CKPT" GPUS="${GPUS:-4}" bash "$D/run_probe_sft.sh"
