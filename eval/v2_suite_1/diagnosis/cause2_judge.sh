#!/bin/bash
: "${EP:?set EP=http://host:port/v1[,...] (vLLM judge endpoints; see docs/RUNBOOK.md)}"
# Cause-2: after the 4 fine-tuned models have generated the 150-item subset, judge them with the unchanged harness.
L=/scratch/network/ssd/junlin/ssl_eval/diag_sft/logs
until grep -q ALLDONE $L/cause2_gen.log 2>/dev/null; do sleep 30; done
cd /scratch/network/ssd2/junlin/ssl_mllm/baseline_pipeline/eval/v2_suite_1/pipeline
for b in phyeditbench picabench risebench phyedit_anti imgedit_basic; do
  /homes/55/junlin/miniconda3/envs/internvlu/bin/python judge.py --bench $b --models base_editsft,T0_editsft,T2_editsft,T3_editsft      --uids ../diagnosis/cause2_subset_uids.txt --workers 96 --endpoints "$EP" >> ../results/judge_logs/cause2_judge.log 2>&1
done
echo DONE >> ../results/judge_logs/cause2_judge.log
