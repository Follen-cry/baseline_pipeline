#!/bin/bash
# v2 T2.1: T2 restricted to gap_s in {1.0, 2.0} (drops gap_s == 0.5), cond_image forced to F0
# instead of the nearest shown frame. Same pool as T0-T4 (RUN=main), so it only makes sense
# with RUN=main (the default). 33,046 train / 428 eval rows (vs T2's 60,000 / 750) --
# data/v2/recipes/temporal_ssl/derive_settings.py --settings T2.1.
# LM_LOSS_WEIGHT=0.0; validation on: T2.1 evalmini. Recipe, options and smoke mode: run_t_gen_sft.sh.
# MASTER_PORT is set explicitly here because run_t_gen_sft.sh derives its default via
# $((34230 + N)) with N="${SETTING#T}" -- that arithmetic breaks on "2.1".
# Usage (internvlu env):  bash run_t2_1_gen_sft.sh        smoke:  SMOKE=1 GPUS=2 bash run_t2_1_gen_sft.sh
export SETTING=T2.1
export LM_LOSS_WEIGHT="${LM_LOSS_WEIGHT:-0.0}"
export VAL_SETS="${VAL_SETS:-T2.1}"
export MASTER_PORT="${MASTER_PORT:-34237}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_t_gen_sft.sh" "$@"
