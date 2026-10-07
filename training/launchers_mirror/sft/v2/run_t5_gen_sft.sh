#!/bin/bash
# v2 T5: F0 F1 F2 + GAP -> F3 (T0's shape), PhysicTran38K only, gap fixed at 1.0 s, cond_image
# forced to F0 instead of T0's F2. Its own pool (RUN=phystran_g1), built by
# merge_pools.py --name phystran_g1 --sources physictran38k --dt 1.0 --quota physictran38k=9634
# --eval-clips-from main, then derive_settings.py --run phystran_g1 --settings T5,T6.
# 9,634 train / 125 eval rows (PhysicTran38K's 3.27 s clips give exactly one gap=1.0 window
# each, so this is the ceiling given the local source -- see baseline_pipeline/docs/v2.md).
# LM_LOSS_WEIGHT=0.0; validation on: T5 evalmini (the full 125-row eval set, not subsampled).
# Recipe, options and smoke mode: run_t_gen_sft.sh.
# NOTE: PhysicTran38K's caption states the outcome ("By the end ..."), which for this
# forecasting task can just tell the model in text what F3 looks like -- see docs/v2.md.
# Usage (internvlu env):  bash run_t5_gen_sft.sh        smoke:  SMOKE=1 GPUS=2 bash run_t5_gen_sft.sh
export SETTING=T5
export LM_LOSS_WEIGHT="${LM_LOSS_WEIGHT:-0.0}"
export VAL_SETS="${VAL_SETS:-T5}"
export RUN="${RUN:-phystran_g1}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_t_gen_sft.sh" "$@"
