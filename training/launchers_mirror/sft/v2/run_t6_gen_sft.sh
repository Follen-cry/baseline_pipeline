#!/bin/bash
# v2 T6: drop Fk (k in 1..3) + GAP + MISSING -> Fk (T2's shape), PhysicTran38K only, gap fixed
# at 1.0 s, cond_image forced to F0. Same pool as T5 (RUN=phystran_g1); see run_t5_gen_sft.sh
# for how it's built. 9,634 train / 125 eval rows.
# LM_LOSS_WEIGHT=0.0; validation on: T6 evalmini (the full 125-row eval set, not subsampled).
# Recipe, options and smoke mode: run_t_gen_sft.sh.
# Usage (internvlu env):  bash run_t6_gen_sft.sh        smoke:  SMOKE=1 GPUS=2 bash run_t6_gen_sft.sh
export SETTING=T6
export LM_LOSS_WEIGHT="${LM_LOSS_WEIGHT:-0.0}"
export VAL_SETS="${VAL_SETS:-T6}"
export RUN="${RUN:-phystran_g1}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_t_gen_sft.sh" "$@"
