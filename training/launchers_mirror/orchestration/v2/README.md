# `orchestration/v2/` — v2 multi-run drivers

## `run_t_gen_sft_queue.sh`

Trains v2 settings one after another on one set of GPUs through the `../../sft/v2/run_t{N}_gen_sft.sh`
launchers (recipe, 6 evalmini validations and LoRA merge live there). Default: **T0, T2, T3 on 4 GPUs**
(`CUDA_VISIBLE_DEVICES=0,1,3,6`, 1 row per GPU × accumulation 4 = global batch 16); T1 / T4 are not
queued for now (`SETTINGS="T1 T4"` adds them).

Per setting: wait for the GPUs to be free (≤ `GPU_WAIT_MIN`, default 120 min; "free" = < 2 GB used),
train + validate + merge, then delete the raw run's weight dirs (the `-merged` dir is a full copy;
`training_log.txt`, `val/`, `runs/`, `wandb/` are kept; `KEEP_RAW=1` keeps everything). A setting whose
`-merged/vlm` exists is skipped, so re-running the script resumes the queue; a failed setting is
recorded and the queue moves on. At the end it prints every validation run's key metrics.

Time (A40, measured 10.2 s/step on 4 GPUs): ~11–12 h per setting (3,750 steps + 6 validations +
merge), ~35 h for T0 T2 T3.

```bash
conda activate internvlu
cd baseline_pipeline/training/models/internvl-u/internvl_chat/shell/internvlu/orchestration/v2
mkdir -p /scratch/network/ssd/junlin/models/v2_logs
nohup bash run_t_gen_sft_queue.sh > /scratch/network/ssd/junlin/models/v2_logs/queue.log 2>&1 &

DRY_RUN=1 bash run_t_gen_sft_queue.sh        # print the plan
SMOKE=1 bash run_t_gen_sft_queue.sh          # 4 steps per setting, throwaway dirs, no merge
SETTINGS="T3" CUDA_VISIBLE_DEVICES=0,1,3,6 bash run_t_gen_sft_queue.sh
```

Outputs: `/scratch/network/ssd/junlin/models/internvlu-v2-t{N}-gen(-merged)`, per-setting logs in
`$LOG_DIR` (default `/scratch/network/ssd/junlin/models/v2_logs/t{N}.log`), wandb project
`internvlu-v2`, runs `t{N}-gen`.

Setting definitions: `baseline_pipeline/docs/v2.md`. v1 drivers live in `../v1/`.
