#!/bin/bash
# v2 T1: F0 F1 F2 -> {"gap"} + F3
# LM_LOSS_WEIGHT=0.5; validation on: T1 evalmini. Recipe, options and smoke mode: run_t_gen_sft.sh.
# Usage (internvlu env):  bash run_t1_gen_sft.sh        smoke:  SMOKE=1 GPUS=2 bash run_t1_gen_sft.sh
export SETTING=T1
export LM_LOSS_WEIGHT="${LM_LOSS_WEIGHT:-0.5}"
export VAL_SETS="${VAL_SETS:-T1}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_t_gen_sft.sh" "$@"
