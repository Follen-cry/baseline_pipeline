#!/bin/bash
# Re-runs every benchmark's judge (resumable) until all 4 x 1210 outputs exist, then one final pass.
EP=${EP:-http://torrnode12:8031/v1,http://torrnode15:8032/v1,http://torrnode8:8033/v1}
PY=/homes/55/junlin/miniconda3/envs/internvlu/bin/python
cd "$(dirname "$0")"
L=../results/judge_logs; mkdir -p $L
while true; do
  n=$(find ../outputs/{base,T0,T2,T3} -name "*.png" ! -name "*.tmp.png" | wc -l)
  for b in phyeditbench picabench risebench phyedit_anti imgedit_basic imgedit_uge magicbrush; do
    $PY judge.py --bench $b --endpoints $EP --workers 64 >> $L/$b.log 2>&1
  done
  echo "$(date) pass done, outputs=$n" >> $L/loop.log
  [ "$n" -ge 4840 ] && break
  sleep 120
done
echo "$(date) FINAL" >> $L/loop.log
