#!/bin/bash
# Start one detached gen.py per shard on a node (all paths from ../config.json).
# usage: launch_gen.sh <model> <model_path|-> <node> <num_shards> <shard:gpu> [<shard:gpu> ...]
#   e.g. ./launch_gen.sh T4 /scratch/network/ssd/junlin/models/internvlu-v2-t4-gen-merged torrnode12 2 0:0 1:1
#   "-" as model_path = use config.models[<model>]. ~8 s/task on an A40 -> 200 tasks ~ 27 min on one GPU.
set -e
M=$1; CK=$2; N=$3; NS=$4; shift 4
D=$(cd "$(dirname "$0")" && pwd)
PY=$(python3 "$D/common.py" envs.internvlu_python)
OUT=$(python3 "$D/common.py" outputs_root)
L=$OUT/$M/_logs; mkdir -p "$L"
MP=""; [ "$CK" != "-" ] && MP="--model_path $CK"
for sg in "$@"; do
  i=${sg%%:*}; g=${sg##*:}
  timeout 20 ssh -f -n -o BatchMode=yes "$N" "cd $D && CUDA_VISIBLE_DEVICES=$g setsid nohup $PY gen.py --model $M $MP --shard_idx $i --num_shards $NS > $L/shard$i.log 2>&1 < /dev/null &"
  echo "launched $M shard $i on $N gpu $g -> $L/shard$i.log"
done
