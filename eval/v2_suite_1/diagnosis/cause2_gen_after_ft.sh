#!/bin/bash
# Cause-2: when a probe fine-tune finishes, generate the fixed 150-item subset with it on the same
# node's freed GPUs (4 shards, identical inference settings via gen.py / config.json).
# Outputs: eval/v2_suite_1/outputs/<model>_editsft/...  (model names: base_editsft, T0_editsft, ...)
S=/scratch/network/ssd2/junlin/ssl_mllm/baseline_pipeline/eval/v2_suite_1/pipeline
U=/scratch/network/ssd2/junlin/ssl_mllm/baseline_pipeline/eval/v2_suite_1/diagnosis/cause2_subset_uids.txt
L=/scratch/network/ssd/junlin/ssl_eval/diag_sft/logs
PY=/homes/55/junlin/miniconda3/envs/internvlu/bin/python
run(){ m=$1; node=$2; gpus=$3
  until grep -q "^DONE " $L/$m.log || grep -q "training did not finish\|Traceback" $L/$m.log; do sleep 30; done
  grep -q "^DONE " $L/$m.log || { echo "$m FT FAILED" >> $L/cause2_gen.log; return; }
  i=0; for g in ${gpus//,/ }; do
    ssh -n -o BatchMode=yes $node "cd $S && CUDA_VISIBLE_DEVICES=$g $PY gen.py --model ${m}_editsft --model_path /scratch/network/ssd/junlin/models/diag_editsft/${m}-merged --uids $U --shard_idx $i --num_shards 4 > $L/gen_${m}_$i.log 2>&1" &
    i=$((i+1)); done; wait
  echo "$m gen done $(date)" >> $L/cause2_gen.log
}
run base torrnode12 0,1,2,3 & run T0 torrnode15 0,1,2,3 & run T2 torrnode8 0,1,2,3 & run T3 torrnode11 0,1,3,4 & wait
echo ALLDONE >> $L/cause2_gen.log
