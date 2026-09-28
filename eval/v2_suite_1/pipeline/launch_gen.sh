#!/bin/bash
# usage: launch_gen.sh <model> <node> <num_shards> <shard:gpu> [<shard:gpu> ...]
# starts one detached gen.py per shard on <node>; logs -> outputs/<model>/_logs/shard<i>.log
M=$1; N=$2; NS=$3; shift 3
D=/scratch/network/ssd2/junlin/ssl_mllm/baseline_pipeline/eval/v2_suite_1/pipeline
L=/scratch/network/ssd/junlin/ssl_eval/outputs/$M/_logs; mkdir -p $L
for sg in "$@"; do
  i=${sg%%:*}; g=${sg##*:}
  timeout 20 ssh -f -n -o BatchMode=yes $N "cd $D && CUDA_VISIBLE_DEVICES=$g setsid nohup /homes/55/junlin/miniconda3/envs/internvlu/bin/python gen.py --model $M --shard_idx $i --num_shards $NS > $L/shard$i.log 2>&1 < /dev/null &"
done
