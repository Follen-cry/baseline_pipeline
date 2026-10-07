#!/bin/bash
# v2 T4: 50% T4-A (= T0) / 50% T4-B: shuffled A B C, no GAP/MISSING -> {"order","gap","missing"} + missing frame
# LM_LOSS_WEIGHT=0.5; validation on: T4 T0 evalmini. Recipe, options and smoke mode: run_t_gen_sft.sh.
# Usage (internvlu env):  bash run_t4_gen_sft.sh        smoke:  SMOKE=1 GPUS=2 bash run_t4_gen_sft.sh
export SETTING=T4
export LM_LOSS_WEIGHT="${LM_LOSS_WEIGHT:-0.5}"
export VAL_SETS="${VAL_SETS:-T4 T0}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_t_gen_sft.sh" "$@"
