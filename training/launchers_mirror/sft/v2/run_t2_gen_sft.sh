#!/bin/bash
# v2 T2: 3 of F0..F3 in order + GAP + MISSING -> the missing frame
# LM_LOSS_WEIGHT=0.0; validation on: T2 evalmini. Recipe, options and smoke mode: run_t_gen_sft.sh.
# Usage (internvlu env):  bash run_t2_gen_sft.sh        smoke:  SMOKE=1 GPUS=2 bash run_t2_gen_sft.sh
export SETTING=T2
export LM_LOSS_WEIGHT="${LM_LOSS_WEIGHT:-0.0}"
export VAL_SETS="${VAL_SETS:-T2}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_t_gen_sft.sh" "$@"
