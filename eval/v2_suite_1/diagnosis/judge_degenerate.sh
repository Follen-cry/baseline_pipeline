#!/bin/bash
: "${EP:?set EP=http://host:port/v1[,...] (vLLM judge endpoints; see docs/RUNBOOK.md)}"
# Cause-1: judge the random and GT degenerate baselines with the unchanged main judge harness.
cd /scratch/network/ssd2/junlin/ssl_mllm/baseline_pipeline/eval/v2_suite_1/pipeline
for b in phyeditbench picabench risebench phyedit_anti imgedit_basic imgedit_uge magicbrush; do
  /homes/55/junlin/miniconda3/envs/internvlu/bin/python judge.py --bench $b --models random,gt --workers 96 --endpoints "$EP" >> ../results/judge_logs/degenerate.log 2>&1
done
echo DONE >> ../results/judge_logs/degenerate.log
