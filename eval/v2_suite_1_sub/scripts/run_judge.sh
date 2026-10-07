#!/bin/bash
# Judge one or more systems on all 4 benchmarks (resumable), then rescore.
# usage: EP=http://node:port/v1[,http://node2:port/v1] ./run_judge.sh <model>[,<model2>...]
set -e
: "${EP:?set EP=http://host:port/v1[,...] (see serve_judge.sh)}"
D=$(cd "$(dirname "$0")" && pwd); cd "$D"
PY=$(python3 common.py envs.internvlu_python)
L=$(python3 common.py results_root)/judge_logs; mkdir -p "$L"
for b in phyeditbench picabench risebench imgedit_basic; do
  "$PY" judge.py --bench $b --models "$1" --endpoints "$EP" --workers ${WORKERS:-64} 2>&1 | tee -a "$L/$b.log"
done
"$PY" score.py
