#!/bin/bash
# v2 T0: F0 F1 F2 + GAP -> F3 (forecasting baseline)
# LM_LOSS_WEIGHT=0.0; validation on: T0 evalmini. Recipe, options and smoke mode: run_t_gen_sft.sh.
# Usage (internvlu env):  bash run_t0_gen_sft.sh        smoke:  SMOKE=1 GPUS=2 bash run_t0_gen_sft.sh
export SETTING=T0
export LM_LOSS_WEIGHT="${LM_LOSS_WEIGHT:-0.0}"
export VAL_SETS="${VAL_SETS:-T0}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_t_gen_sft.sh" "$@"
