# Runbook

How every result in this suite was produced, in order, with the exact commands. Paths are relative to
`eval/v2_suite_1/` unless absolute. Every generation and judging step is **resumable**: re-running
skips what already exists, so an interrupted step is simply started again.

## Environments

| Env | Used for |
|---|---|
| `internvlu` (`~/miniconda3/envs/internvlu`) | generation (`gen.py`), judge client (`judge.py`), analysis, fine-tuning |
| `vllm` (vLLM 0.11.2) | serving the judges (`vllm serve`) |
| `geneval-eval-env` | MagicBrush CLIP-I / DINO (`rule_metrics.py --part magicbrush`) |

The cluster has no working Slurm from torrnode11; GPUs were found by `ssh torrnodeN nvidia-smi`
(A40 nodes: 8, 11, 12, 13, 14, 15). All data paths are on network storage, so any node can run any step.

## 0. Suite definition

```bash
cd pipeline && $PY build_manifest.py      # -> manifest.jsonl (1,210 tasks) + staged inputs
```
Fixed seed, stratified allocation; see `../SUITE.md`. `config.json` holds the 4 checkpoints and the
inference settings (area ~512² with aspect kept, 20 steps, all_cfg 4.5, part_cfg 2.0, seed 42,
`generation_mode="image"`).

## 1. Generation

```bash
cd pipeline
./launch_gen.sh <model> <node> <num_shards> <shard:gpu> [<shard:gpu> ...]
# e.g. ./launch_gen.sh T0 torrnode12 5 0:0 1:1 2:2 3:3 4:4
```
One detached `gen.py` per shard (`ssh -f -n` + `setsid`); logs in `outputs/<model>/_logs/`,
failures in `failures_shard*.jsonl`. About 8 s per task on an A40, i.e. ~40 min per model on 5 GPUs.
`gen.py --model_path <dir> --uids <file>` runs any checkpoint on a subset without touching `config.json`.

## 2. Judge servers

```bash
J=/scratch/network/ssd/junlin/models/judges/Qwen3-VL-30B-A3B-Instruct-FP8
CUDA_VISIBLE_DEVICES=<g> ~/miniconda3/envs/vllm/bin/vllm serve $J --served-model-name qwen3-vl-30b-fp8 \
  --host 0.0.0.0 --port <p> --max-model-len 20480 --max-num-seqs 32 \
  --limit-mm-per-prompt '{"image": 3}' --gpu-memory-utilization 0.90 --seed 0
export EP=http://<node>:<p>/v1,http://<node2>:<p2>/v1      # used by every judge script
```
3 images per prompt is required (PhyEditBench sends input, GT and prediction). FP8 runs through
Marlin on the A40s; the first start compiles for a few minutes. Second judge for the sanity check:
the same command with `Qwen/Qwen3-VL-8B-Instruct` from the HF cache, served as `qwen3-vl-8b`
(`export EP_8B=...`). The user's own server on torrnode11 GPU 5 (port 8010, 2 images max) is not used.

## 3. Judging

```bash
cd pipeline
./judge_loop.sh                 # all 7 benchmarks for base/T0/T2/T3, repeats until all outputs exist
```
`judge.py` imports each benchmark's **official** judge module from `…/ssl_eval/src/` and swaps only the
API client (temperature 0). Parse failures are retried (up to 3x at temperature 0.7, seeded) and kept,
never dropped. UGE/MagicBrush use a documented `Score: N` fallback because the official UGE prompt
has no score line. One JSONL row per judged unit in `results/judge_raw/<judge>/<bench>/<model>.jsonl`.

## 4. Rule metrics and the copy reference

```bash
cd pipeline
$PY rule_metrics.py --part pica                                          # PICABench masked PSNR
CUDA_VISIBLE_DEVICES=0 ~/miniconda3/envs/geneval-eval-env/bin/python rule_metrics.py --part magicbrush
$PY make_copy_reference.py      # outputs/copy/ (input unchanged) + results/rule/editmag.csv
./judge_copy.sh                 # judge the copy reference like a model
```

## 5. Judge sanity check

```bash
cd pipeline && ./sanity.sh && $PY agreement.py      # needs EP and EP_8B
```

## 6. Analysis, report, page

```bash
cd pipeline && $PY analyze.py && $PY report.py
$PY ../results/build_artifact.py                    # -> results/eval_report.html (<16 MB)
```
Publish or update the page with the Artifact tool on `results/eval_report.html`
(existing URL: https://claude.ai/artifact/DDBbysKBGP8UKhxkApMuta). The conclusions text lives in
`results/conclusions.json` with `{{...}}` placeholders, so every number on the page comes from a CSV.

## 7. Diagnosis

**Cause 1 (judge).**
```bash
cd diagnosis
$PY make_degenerate_baselines.py    # outputs/random/, outputs/gt/ + results/rule/editmag_ref.csv
./judge_degenerate.sh               # needs EP
$PY cause1_rescore.py               # baseline table + no-edit penalty at tau 0.02 / 0.03 / 0.05
```

**Cause 2 (format).** An identical small editing fine-tune of all four models, then a fixed 150-item subset.
```bash
cd diagnosis/edit_sft && $PY build_edit_sft.py      # 640 MagicBrush-train + 640 PICA-100K, overlap-checked
MODEL=T0 CKPT=<merged ckpt> GPUS=4 bash run_edit_sft_probe.sh    # once per model, 4 A40s, ~30 min
cd .. && ./cause2_gen_after_ft.sh   # waits for each fine-tune, generates the subset on its node
./cause2_judge.sh                   # needs EP; waits for generation
$PY cause2_analyze.py && $PY ftdata_editmag.py
```
The launcher uses the v2 T-series recipe (LoRA r32, lr 1e-5, decoder lr 5e-5, gen-loss 0.5,
area-512), 2 epochs = 160 steps, and writes to `/scratch/network/ssd/junlin/models/diag_editsft/`.

**Cause 3 (training).**
```bash
cd diagnosis
python3 pool_target_change.py       # target vs VAE-condition change over the 60K training windows
$PY failure_modes.py --endpoints "$EP"   # auto checks on all items + VLM failure tags on the loss set
```
`results/diagnosis/cause3_code_review.csv` is hand-curated (file:line findings from reading the v2 data
and training code); edit it directly.

**Control check** (is the fine-tune comparison valid?).
```bash
cd diagnosis/edit_sft
MODEL=base_null CKPT=<base snapshot> GPUS=4 LR=0 GEN_DECODER_LR=0 EXTRA="--max_steps 2" bash run_edit_sft_probe.sh
cd ../../pipeline && $PY gen.py --model base_nullft --model_path <…/diag_editsft/base_null-merged> \
   --uids ../diagnosis/cause2_subset_uids.txt
cd ../diagnosis && $PY weight_diff.py <…/base_null-merged> <…/base-merged> && $PY control_check.py
```

**Write-up.**
```bash
cd diagnosis && $PY select_examples.py && $PY build_diagnosis.py && $PY ../results/build_artifact.py
```
`build_diagnosis.py` holds the narrative once and writes both `results/DIAGNOSIS.md` and the page text
(`results/diagnosis/diagnosis.json`); numbers are `{{d:<table>:<filter>:<field>}}` placeholders.

## Gotchas

- `pkill -f <pattern>` from a shell whose own command line contains the pattern kills that shell
  (exit 144). Kill by PID filtered on the process name instead.
- Plain `ssh node "nohup … &"` can hang the caller; use `ssh -f -n … setsid nohup … < /dev/null &`.
- `vllm` and `internvlu` envs differ: the judge client (`judge.py`) needs `internvlu` (pandas,
  tenacity, openai); the `vllm` env lacks pandas.
- HF downloads of Xet-backed datasets stall with the hub client; `curl -L …/resolve/main/<file>` works.
- The fp32 → bf16 dtype change on 36 generation-decoder tensors when saving trained checkpoints is inert
  (the pipeline loads everything in bf16); the control check confirms pixel-identical outputs.
