#!/bin/bash
# 5% sanity re-judge: (1) same Qwen3-VL-30B judge, second run; (2) different judge Qwen3-VL-8B.
cd "$(dirname "$0")"; PY=/homes/55/junlin/miniconda3/envs/internvlu/bin/python; L=../results/judge_logs
for b in phyeditbench picabench risebench phyedit_anti imgedit_basic imgedit_uge magicbrush; do
  $PY judge.py --bench $b --uids sanity_uids.txt --judge_name qwen3vl30b_fp8_rerun --workers 32 \
     --endpoints http://torrnode8:8036/v1,http://torrnode15:8037/v1 >> $L/sanity_rerun.log 2>&1
  $PY judge.py --bench $b --uids sanity_uids.txt --judge_name qwen3vl8b --served qwen3-vl-8b --workers 32 \
     --endpoints http://torrnode15:8041/v1 >> $L/sanity_8b.log 2>&1
done
echo DONE >> $L/sanity_8b.log
