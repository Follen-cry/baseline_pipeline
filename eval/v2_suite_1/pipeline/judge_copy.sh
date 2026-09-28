#!/bin/bash
: "${EP:?set EP=http://host:port/v1[,...] (vLLM judge endpoints; see docs/RUNBOOK.md)}"
cd "$(dirname "$0")"
for b in phyeditbench picabench risebench phyedit_anti imgedit_basic imgedit_uge magicbrush; do
  /homes/55/junlin/miniconda3/envs/internvlu/bin/python judge.py --bench $b --models copy --workers 96 --endpoints "$EP" >> ../results/judge_logs/copy.log 2>&1
done
echo DONE >> ../results/judge_logs/copy.log
