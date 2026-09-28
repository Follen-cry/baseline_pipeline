#!/bin/bash
cd "$(dirname "$0")"
for b in phyeditbench picabench risebench phyedit_anti imgedit_basic imgedit_uge magicbrush; do
  /homes/55/junlin/miniconda3/envs/internvlu/bin/python judge.py --bench $b --models copy --workers 96 --endpoints http://torrnode12:8031/v1,http://torrnode15:8032/v1,http://torrnode8:8033/v1,http://torrnode12:8034/v1,http://torrnode12:8035/v1,http://torrnode8:8036/v1,http://torrnode15:8037/v1 >> ../results/judge_logs/copy.log 2>&1
done
echo DONE >> ../results/judge_logs/copy.log
