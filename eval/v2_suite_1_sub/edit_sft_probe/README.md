# edit_sft_probe: a lightweight instruction-editing SFT probe for v2_suite_1_sub

A smaller, faster sibling of [`../../v2_suite_1/diagnosis/edit_sft`](../../v2_suite_1/diagnosis/edit_sft)
(the probe behind `base_ft`/`T0_ft`/`T2_ft`/`T3_ft` in `../results/table.csv`), paired with **this**
200-item subset manifest (`../manifest.jsonl`) instead of v2_suite_1's full 1,210-item suite.

**What it's for.** `../results/DIAGNOSIS.md` Cause 2 found that a tiny bit of instruction-editing
SFT (identical data, only the starting checkpoint differs) closes most of the base-vs-T-series gap
on this suite's benchmarks, mostly by *lowering* `base` rather than raising the T-series — evidence
that the suite partly measures "did you see edit-style instructions during training," not physics
understanding. This probe re-runs that check cheaper (1 epoch instead of 2, ~800 rows instead of
1,280) so it's viable to re-verify each time `v2_suite_1_sub` itself changes, without touching the
full suite.

**Status:** trained and evaluated for `base`, `T2_1`, `T5` and `T6` (2026-09-28/29, scored 2026-10-03);
`T0`, `T2` and `T3` were never run through this probe. See "What's been run" below and `../README.md`
("New-setting checkpoints") for the numbers.

## Data

`build_probe_data.py` (already run; kept for provenance, re-running is deterministic) samples from
the same two sources as the original probe:

- **osunlp/MagicBrush**, official *train* split (4 parquet shards, `mb_train-*.parquet`) — real
  image edits with natural-language instructions.
- **Andrew613/PICA-100K** (1 shard, `pica100k_00000.parquet`) — synthetic, video-derived physics
  edits with a superficial (non-physics-revealing) prompt.

Both were already downloaded to `RAW = /scratch/network/ssd/junlin/ssl_eval/diag_sft/raw` by the
original probe's setup; this script reuses those parquets as-is, no new download.

- **~800 rows**, same 1:1 MagicBrush:PICA-100K ratio as the original 640:640 split — **400 +
  400** here.
- **Seed 20260929** (fixed; one day after the subset's own sampling seed 20260928, chosen so this
  is an independent draw, not a truncation of the original 1,280-row probe which reused
  seed 20260928).
- **No overlap with the 200 eval datapoints**: every candidate source/target image is dhash'd
  (64-bit difference hash) and compared against every input/reference image in `../manifest.jsonl`
  (this subset, not the full suite); any row within Hamming distance ≤ 4 of an eval image is
  dropped. MagicBrush `img_id`s used by a `bench=="magicbrush"` eval row are also excluded by id
  (a no-op here — the 200-item subset happens to contain no `magicbrush` bench rows at all, since
  that bench wasn't sampled into it — kept for parity/safety if the subset is ever resampled).

Outputs (on scratch, not in git):
```
/scratch/network/ssd/junlin/ssl_eval/v2_eval_suite_1_sub/edit_sft_probe/data/
├── images/               *_src.png / *_tgt.png for every kept row
├── train.jsonl           the sampled+formatted training set
├── meta.json             trainer dataset-registry entry ("edit_sft_probe_sub")
└── overlap_report.json   {rows, per_source, eval_images_checked, hamming_max, dropped:[...]}
```
Row format (matches `sft/v1/run_reasoning_edit_sft.sh`'s edit format):
```json
{"id": "...", "task_type": "imgen", "image": ["<src.png>"], "target_image": "<tgt.png>",
 "conversations": [{"from": "human", "value": "<image>\n<instruction>"},
                    {"from": "gpt", "value": "Here is the edited image: <img>"}]}
```

To re-sample from scratch (e.g. a different seed or count): edit `N_PER_SOURCE`/`SEED` at the top
of `build_probe_data.py` and re-run (env: `internvlu`):
```bash
python3 build_probe_data.py
```

## Training config

Reuses the original probe's recipe exactly (`../../v2_suite_1/diagnosis/edit_sft/run_edit_sft_probe.sh`
== the v2 T-series recipe, `sft/v2/run_t_gen_sft.sh`) — **only the data and the epoch count change**:

| | original probe (`../../v2_suite_1/.../edit_sft`) | this probe |
|---|---|---|
| rows | 1,280 (640 + 640) | ~800 (400 + 400) |
| epochs | 2 (160 steps) | **1 (~50 steps)** |
| learning rate | 1e-5 | 1e-5 |
| gen_decoder_lr | 5e-5 | 5e-5 |
| batch size (global / per-device) | 16 / 1 | 16 / 1 |
| resolution (gen target) | area, 512 side | area, 512 side |
| LLM LoRA | r32 | r32 |
| gen_loss_weight / warmup | 0.5 / 20 | 0.5 / 20 |
| imgen_ratio | 1.0 | 1.0 |
| lm_loss_weight | 0 | 0 |

`run_probe_sft.sh` is a copy of the original `run_edit_sft_probe.sh` with `--num_train_epochs 1`,
pointed at this probe's `meta.json` and a separate output root so it never collides with the
original probe's checkpoints:
```
/scratch/network/ssd/junlin/models/edit_sft_probe_sub/<MODEL>          training output (LoRA + state)
/scratch/network/ssd/junlin/models/edit_sft_probe_sub/<MODEL>-merged   merged checkpoint (what gen.py consumes)
```
It does the same post-processing as the original: restore the real processor over the one the
trainer writes, then merge the LoRA into a full checkpoint.

## Launching training

One wrapper script per model, each pulling the right starting checkpoint from `../config.json`
(`models.base`/`models.T0`/`models.T2`/`models.T3`) and calling `run_probe_sft.sh`:
```bash
cd edit_sft_probe
GPUS=4 bash train_base.sh
GPUS=4 bash train_T0.sh
GPUS=4 bash train_T2.sh
GPUS=4 bash train_T3.sh
```
(`run_probe_sft.sh` can also be called directly: `MODEL=T0 CKPT=/path/to/ckpt GPUS=4 bash run_probe_sft.sh`.)
Each run finishes with `DONE /scratch/network/ssd/junlin/models/edit_sft_probe_sub/<MODEL>-merged`.

## Launching evaluation

Once the four merged checkpoints exist, `run_probe_eval.sh` drives this suite's existing,
unmodified `../scripts/gen.py` / `../scripts/run_judge.sh` / `../scripts/score.py` — same
generation settings as `base`/`T0`/`T2`/`T3` (`config.inference`: area ~512², 20 steps, all_cfg 4.5,
part_cfg 2.0, seed 42) and the same 4 benchmarks (PhyEditBench, PICABench, RISEBench, ImgEdit
Basic), on the same 200-item manifest.

```bash
cd edit_sft_probe

# 1. Inference (4 models, one GPU, foreground; ~8 s/task on an A40, ~7 min/model for 200 tasks)
CUDA_VISIBLE_DEVICES=0 ./run_probe_eval.sh gen

# 2. Judge server (on a GPU node; env: vllm)
../scripts/serve_judge.sh <gpu> <port>
export EP=http://<node>:<port>/v1

# 3. Judging (all 4 benches, resumable) + scoring
./run_probe_eval.sh judge
```
This produces `base_probe`, `T0_probe`, `T2_probe`, `T3_probe` (named to avoid clashing with the
existing `base_ft`/`T0_ft`/`T2_ft`/`T3_ft` systems from the original probe). Like those, they are
**not** added to `config.models`/`config.report_models` — `score.py`'s auto-detection (any judged
system under `results/judge_raw` not already in `models`/`references`) picks them up into
`results/table.csv`, `results/scores_item.csv` and `results/coverage.csv` alongside every other
system already there. To show them on the shared page, add their names to `config.report_models`
and re-run `build_artifact.py` (see `../README.md`).

## Layout

```
edit_sft_probe/
├── README.md             this file
├── build_probe_data.py   data sampling (already run; kept for provenance)
├── run_probe_sft.sh       shared training recipe (MODEL / CKPT / GPUS env vars)
├── train_base.sh          wrapper: MODEL=base,   CKPT=config.models.base
├── train_T0.sh            wrapper: MODEL=T0,     CKPT=config.models.T0
├── train_T2.sh            wrapper: MODEL=T2,     CKPT=config.models.T2
├── train_T3.sh            wrapper: MODEL=T3,     CKPT=config.models.T3
└── run_probe_eval.sh      inference (`gen`) + judging/scoring (`judge`) for all 4 probe checkpoints
```
Training/eval outputs are on scratch, not in this git-tracked folder:
```
/scratch/network/ssd/junlin/ssl_eval/v2_eval_suite_1_sub/edit_sft_probe/data/   sampled train set
/scratch/network/ssd/junlin/models/edit_sft_probe_sub/<MODEL>(-merged)/        checkpoints
/scratch/network/ssd/junlin/ssl_eval/v2_eval_suite_1_sub/outputs/<MODEL>_probe/  generated images
../results/{table,scores_item,coverage}.csv                                    scores (once judged)
```

## What's been run

- [x] `build_probe_data.py` — **done 2026-09-28.** 800 rows kept (400 MagicBrush + 400 PICA-100K,
      exact 1:1 as planned), 19 of 819 sampled candidates dropped for near-duplicate overlap with
      the 200 eval images (all 19 were PICA-100K rows near-duplicating `phy/State_Change_&_Environment/...`
      or `rise/temporal_reasoning_15`; Hamming distance 1–4), 0 dropped by MagicBrush `img_id`
      (expected — no `magicbrush` bench in this subset), 260 eval images checked. Full detail in
      `data/overlap_report.json`; the sampled list itself is `data/train.jsonl` (1,600 images in
      `data/images/`).
- [x] Training — `train_base.sh` **done 2026-09-28** (17 min, 800 samples, `base-merged`); `train_T2_1.sh`,
      `train_T5.sh`, `train_T6.sh` (new wrappers, same pattern as `train_T0.sh`, reading `models.T2_1` /
      `models.T5` / `models.T6`) **done 2026-09-29**, one node of 4 GPUs each, run with
      `EXTRA="--dataloader_drop_last True --save_strategy steps --save_steps 20 --save_total_limit 2"` as a
      safety net (800 rows / batch 16 = 50 steps divides evenly and all three finished normally).
      `train_T0.sh` / `train_T2.sh` / `train_T3.sh` — not run.
- [x] Evaluation — `base_probe`, `T2_1_probe`, `T5_probe`, `T6_probe` generated (200/200 each) and judged
      (all 4 benchmarks, 0 unparsed rows). `run_probe_eval.sh` still lists only base/T0/T2/T3 and was **not**
      used for the new three: inference used `scripts/launch_gen.sh <name>_probe <ckpt> ...` and judging
      `scripts/run_judge.sh` directly (commands in `../README.md`).

Config reference for this probe: `../config.json → probe_models`.
