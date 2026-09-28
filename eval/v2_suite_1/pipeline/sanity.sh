#!/bin/bash
: "${EP:?set EP=<Qwen3-VL-30B-FP8 endpoints> (see docs/RUNBOOK.md)}"
: "${EP_8B:?set EP_8B=<Qwen3-VL-8B endpoint, served as qwen3-vl-8b> (see docs/RUNBOOK.md)}"
# 5% sanity re-judge: (1) same Qwen3-VL-30B judge, second run; (2) different judge Qwen3-VL-8B.
cd "$(dirname "$0")"; PY=/homes/55/junlin/miniconda3/envs/internvlu/bin/python; L=../results/judge_logs
for b in phyeditbench picabench risebench phyedit_anti imgedit_basic imgedit_uge magicbrush; do
  $PY judge.py --bench $b --uids sanity_uids.txt --judge_name qwen3vl30b_fp8_rerun --workers 32 \
     --endpoints "$EP" >> $L/sanity_rerun.log 2>&1
  $PY judge.py --bench $b --uids sanity_uids.txt --judge_name qwen3vl8b --served qwen3-vl-8b --workers 32 \
     --endpoints "$EP_8B" >> $L/sanity_8b.log 2>&1
done
echo DONE >> $L/sanity_8b.log
