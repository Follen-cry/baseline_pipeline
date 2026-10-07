#!/bin/bash
# v2 T3: F0 F1 F2 shuffled as A B C + GAP -> {"order"} + F3
# LM_LOSS_WEIGHT=0.5; validation on: T3 evalmini. Recipe, options and smoke mode: run_t_gen_sft.sh.
# Usage (internvlu env):  bash run_t3_gen_sft.sh        smoke:  SMOKE=1 GPUS=2 bash run_t3_gen_sft.sh
export SETTING=T3
export LM_LOSS_WEIGHT="${LM_LOSS_WEIGHT:-0.5}"
export VAL_SETS="${VAL_SETS:-T3}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_t_gen_sft.sh" "$@"
