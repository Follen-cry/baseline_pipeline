#!/bin/bash
# All datapoints x 3 VAE conditions x 3 sizes, sharded over the given GPUs of THIS node (one gen.py per GPU, resumable).
# usage: bash run_all.sh <gpu> [<gpu> ...]       e.g. bash run_all.sh 0 1 2 3 4 5
# logs: logs/shard<i>.log ; then: python make_grid.py / size_sheet.py / build_artifact.py (see README)
set -e
D=$(cd "$(dirname "$0")" && pwd)
cd "$D"
PY=$(python3 common.py internvlu_python)
mkdir -p logs
N=$#
i=0
for g in "$@"; do
  CUDA_VISIBLE_DEVICES=$g nohup $PY gen.py --shard_idx $i --num_shards $N > logs/shard$i.log 2>&1 < /dev/null &
  echo "shard $i/$N on gpu $g -> logs/shard$i.log"
  i=$((i+1))
done
wait
echo "all shards finished"
