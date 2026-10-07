# v2_suite_1_sub: a 200-datapoint subset of v2_suite_1

A smaller, self-contained version of [`../v2_suite_1`](../v2_suite_1) for evaluating new checkpoints quickly
(~30 GPU-minutes of generation plus a few judge-minutes per model, instead of 1,210 tasks). Every datapoint keeps
its v2_suite_1 `uid`, so any score here maps back to the full suite. The base, T0, T2 and T3 outputs and scores
were **imported, not re-run**, together with the copy and random references from the diagnosis step.

Page: https://claude.ai/artifact/2i598fqeNMQa4eakWp7jL6 (private until shared), built by
`scripts/build_artifact.py`.

## How the subset was sampled

`scripts/sample_subset.py` (already run; kept for provenance) reads v2_suite_1's `pipeline/manifest.jsonl` as a
data file and copies the chosen rows verbatim, so `manifest.jsonl` has the same fields and uids as the original.

- Seed **20260928**. Each pool has its own RNG stream: `random.Random(f"20260928:{pool}").sample(sorted(uids), n)`.
- Split 3:3:3:1:

| Benchmark | Pool in v2_suite_1 | Pool size | Sampled |
|---|---|---:|---:|
| RISEBench | group A: Temporal + Causal | 175 | 60 |
| PhyEditBench | group A: Types A–E (Anti-Physics excluded) | 480 | 60 |
| PICABench | group A: superficial prompts, 8 laws | 260 | 60 |
| ImgEdit | ImgEdit Basic (9 edit types; this is group C in v2_suite_1) | 108 | 20 |
| **total** | | | **200** |

Two scope choices, both set in `config.json → sampling.pools`:
- ImgEdit is not part of group A (Target) in v2_suite_1. Its 20 datapoints come from ImgEdit Basic, the group-C
  control. UGE (16 items, a different prompt format) is not included.
- PhyEditBench's Anti-Physics items (35, counterfactual instructions) are left out. v2_suite_1 reads them
  separately, and they expect the opposite direction of effect.

Realised mix: RISE 33 temporal / 27 causal; PhyEditBench Types A–E 16/9/14/7/14; PICABench Mechanics 27 /
State 19 / Optics 14; ImgEdit 8 of the 9 edit types.

## Layout

```
v2_suite_1_sub/
├── README.md
├── config.json        the only place paths live: checkpoints, outputs root, judge weights, benchmark code, envs
├── manifest.jsonl     200 rows, same fields as v2_suite_1's manifest
├── scripts/
│   ├── common.py              config loader (every script imports only this)
│   ├── gen.py, launch_gen.sh  inference (resumable, sharded)
│   ├── serve_judge.sh         start a Qwen3-VL-30B-A3B-Instruct-FP8 vLLM server
│   ├── judge.py, run_judge.sh official judge code per benchmark, client swapped (resumable)
│   ├── score.py               judge rows -> per-item scores, table, coverage check
│   ├── scoring_rules.py       exact judge prompts + scoring code + worked examples -> results/scoring_rules.json
│   ├── build_artifact.py, report_template.html   the shareable page
│   ├── sample_subset.py       how manifest.jsonl was made (one-off)
│   └── import_v2_suite_1.py   how the existing outputs/scores were copied in (one-off)
└── results/
    ├── judge_raw/qwen3vl30b_fp8/<bench>/<system>.jsonl   raw judge rows (one per judged unit)
    ├── scores_item.csv   per item, system, metric
    ├── table.csv         system × benchmark native metrics + Overall
    ├── coverage.csv      system × benchmark: items, images, scored, unparsed judge rows kept
    ├── missing.csv       (system, uid) pairs without an image or primary score; empty = complete
    ├── import_log.csv    what import_v2_suite_1.py copied
    ├── scoring_rules.json  judge prompts / parsing / weights per benchmark (page's "How scoring works")
    └── report.html       built page (git-ignored)
```

Generated images live on network storage at `config.outputs_root`
(`/scratch/network/ssd/junlin/ssl_eval/v2_eval_suite_1_sub/outputs/<system>/<out_rel>`, ~350 MB for 7 systems).
They are real copies, so this suite still works if v2_suite_1's outputs are deleted.

Nothing here imports from v2_suite_1. `gen.py` and `judge.py` are copies of v2_suite_1's scripts, trimmed to
these 4 benchmarks and made config-driven. The only v2_suite_1 paths are under `config.provenance`, and only the
two one-off scripts read them. External dependencies are the InternVL-U package (`inference.internvlu_pkg`), the
official benchmark repos and staged inputs (`judge.benchmark_src`, `judge.staged`), and the judge weights.

## Evaluate a new checkpoint

Environments: `internvlu` for generation, judging client and scoring; `vllm` for the judge server (paths in
`config.envs`). `<name>` below is any label, e.g. `T4`.

**1. Inference.** Settings come from `config.inference`: area ~512² with the input aspect ratio kept,
`generation_mode="image"`, 20 steps, all_cfg 4.5, part_cfg 2.0 and seed 42. These are identical to v2_suite_1.
```bash
cd scripts
# one GPU, in the foreground (~8 s per task on an A40, ~27 min for 200):
CUDA_VISIBLE_DEVICES=0 $(python3 common.py envs.internvlu_python) gen.py --model <name> --model_path /path/to/merged_ckpt
# or detached shards on a node:
./launch_gen.sh <name> /path/to/merged_ckpt torrnode12 4 0:0 1:1 2:2 3:3
```
Outputs go to `<outputs_root>/<name>/<out_rel>`; failures to `<outputs_root>/<name>/_logs/failures_shard*.jsonl`.
Re-running skips finished images. You can also add the checkpoint to `config.models` and then omit `--model_path`.

**2. Judge server** (Qwen3-VL-30B-A3B-Instruct-FP8, temperature 0; 3 images per prompt needed):
```bash
./serve_judge.sh <gpu> <port>          # on the GPU node; first start compiles for a few minutes
export EP=http://<node>:<port>/v1      # comma-separate several servers to spread the load
```

**3. Judging** with each benchmark's rules. `judge.py` imports the official judge module from the benchmark repo
and swaps only the API client:

| Benchmark | Official code | Per-item metric |
|---|---|---|
| PhyEditBench | `PhyEditBench/utils.py::score_one_dimension` (input, GT and prediction) | 4 dimensions 1–10; overall = 0.2 cons + 0.3 instr + 0.4 phys + 0.1 quality |
| PICABench | `PICABench/PicaEval_qwen.py` (ROI crop per checklist question, yes/no) | Acc = share of questions answered correctly |
| RISEBench | `RISEBench/gpt_eval.py::eval_vanilla` (reasoning / consistency / plausibility) | score 1–5, solved 0/1 (Acc) |
| ImgEdit | `ImgEdit/Benchmark/Basic/basic_bench.py` (per-edit-type rubric) | score 1–5 |

```bash
EP=$EP ./run_judge.sh <name>           # all 4 benchmarks, resumable, then runs score.py
```
Parse failures are retried up to 3× (temperature 0.7) and kept with `parse_ok=false`, never dropped. An unparsed
PICABench answer counts as incorrect, as in the official code. Logs go to `results/judge_logs/`.

**4. Scores and page.**
```bash
$(python3 common.py envs.internvlu_python) score.py            # table.csv, coverage.csv, missing.csv (prints all three)
# to show the new model on the page, add it to config.report_models, then:
$(python3 common.py envs.internvlu_python) scoring_rules.py     # prompts + worked examples for the page
$(python3 common.py envs.internvlu_python) build_artifact.py    # -> results/report.html; republish with the Artifact tool
```
`score.py` picks up any judged system automatically. `Overall` is only filled in when a system has every primary score.

## FT variants: base_ft / T0_ft / T2_ft / T3_ft

Four checkpoints already exist that further fine-tune `base`/`T0`/`T2`/`T3` on the same small instruction-editing
set — 1,280 rows (640 MagicBrush *train* turns, disjoint from this suite's MagicBrush test slice, + 640
PICA-100K pairs), 2 epochs / 160 steps, otherwise the v2 T-series recipe (LLM-LoRA r32, lr 1e-5, gen_decoder_lr
5e-5). Paths, base checkpoint and the exact recipe are in `config.json → ft_models`; the data/training scripts
are `../v2_suite_1/diagnosis/edit_sft/{build_edit_sft.py,run_edit_sft_probe.sh}` (built for a different
diagnosis probe, reused here unchanged). They use this suite's usual generation settings (`config.inference`,
identical to `base`/`T0`/`T2`/`T3`) — nothing benchmark-specific changes.

They were run through the same 4 steps above, just with the `_ft` names:
```bash
cd scripts
CUDA_VISIBLE_DEVICES=0 $(python3 common.py envs.internvlu_python) gen.py --model base_ft --model_path /scratch/network/ssd/junlin/models/diag_editsft/base-merged
CUDA_VISIBLE_DEVICES=1 $(python3 common.py envs.internvlu_python) gen.py --model T0_ft   --model_path /scratch/network/ssd/junlin/models/diag_editsft/T0-merged
CUDA_VISIBLE_DEVICES=3 $(python3 common.py envs.internvlu_python) gen.py --model T2_ft   --model_path /scratch/network/ssd/junlin/models/diag_editsft/T2-merged
CUDA_VISIBLE_DEVICES=6 $(python3 common.py envs.internvlu_python) gen.py --model T3_ft   --model_path /scratch/network/ssd/junlin/models/diag_editsft/T3-merged

./serve_judge.sh <gpu> <port>; export EP=http://<node>:<port>/v1
EP=$EP ./run_judge.sh base_ft,T0_ft,T2_ft,T3_ft     # all 4 benchmarks, then score.py
```
Outputs land at `<outputs_root>/{base_ft,T0_ft,T2_ft,T3_ft}/<out_rel>`, alongside `base`/`T0`/`T2`/`T3`/`copy`/
`random`. They are **not** added to `config.models` or `config.report_models` (they are a separate, ad hoc
axis — same checkpoint, plus edit-SFT — not new checkpoints to feature on the page), so the artifact page and
gallery are unchanged; `score.py`'s auto-detection (any judged system under `results/judge_raw` not already in
`config.models`/`config.references`) still picks them up into `table.csv`/`scores_item.csv`/`coverage.csv`.

## New-setting checkpoints: T2_1 / T5 / T6 (zero-shot and edit-SFT probe)

Three checkpoints from the extra v2 settings were evaluated in two forms on the same 200 items (2026-10-03).
The settings are defined in `../../docs/v2.md` ("Extra settings"): **T2.1** = T2 restricted to gap ∈ {1, 2} s;
**T5** / **T6** = PhysicTran38K-only, gap = 1 s, VC2I-F / VC2I-M; all three condition the VAE on F0. The name
`T2_1` is training setting "T2.1" (no dot: the config CLI's dotted-path lookup and shell array keys break on it).

| System | Checkpoint | Notes |
|---|---|---|
| `T2_1`, `T5`, `T6` | `config.models` → `/scratch/network/ssd/junlin/models/internvlu-v2-t{2.1,5,6}-gen-merged` | zero-shot |
| `T2_1_probe`, `T5_probe`, `T6_probe` | `config.probe_models` → `/scratch/network/ssd/junlin/models/edit_sft_probe_sub/{T2_1,T5,T6}-merged` | the three above + the `edit_sft_probe/` edit-SFT (800 rows = 400 MagicBrush train + 400 PICA-100K, 1 epoch / 50 steps, 4 GPUs, otherwise the v2 T-series recipe) |

They are in `config.models` / `config.probe_models` but **not** in `config.report_models`, so the published page
and gallery are unchanged; `score.py` puts them in `table.csv` / `scores_item.csv` / `coverage.csv`.

**Provenance caveat.** The T2.1, T5 and T6 training runs froze at their very last optimizer step on every
attempt (6 of 6; root cause not found, see `../../docs/v2.md`), so the merged weights come from the last
periodic checkpoint: `checkpoint-2000` of 2066 steps for T2_1, `checkpoint-600` of 603 for T5 and T6. The cosine
LR was already about 2e-8 (T2_1) and 1e-9 (T5/T6) there, so the weights differ negligibly from a true final
checkpoint. The edit-SFT probes ran to completion normally.

```bash
# edit-SFT probe (wrappers read config.models; EXTRA is the periodic-checkpoint safety net)
cd edit_sft_probe
EXTRA="--dataloader_drop_last True --save_strategy steps --save_steps 20 --save_total_limit 2" GPUS=4 bash train_T2_1.sh   # also train_T5.sh, train_T6.sh
# inference, 2 shards per model (zero-shot models resolve their path from config.models via "-")
cd ../scripts
./launch_gen.sh T2_1 - <node> 2 0:0 1:1
./launch_gen.sh T2_1_probe /scratch/network/ssd/junlin/models/edit_sft_probe_sub/T2_1-merged <node> 2 0:2 1:3   # likewise T5, T6
# judge + score, all 4 benchmarks, resumable
EP=http://<node>:<port>/v1 ./run_judge.sh T2_1,T5,T6,T2_1_probe,T5_probe,T6_probe
```

Judge-server note (cost about 16 hours here): with the shared NFS saturated, vLLM's startup (Python imports, flashinfer
JIT, torch.compile cache, weights) stalled for hours. What worked was the same flags and byte-identical weights
read from node-local disk (`/scratch/local/ssd/junlin/models/Qwen3-VL-30B-A3B-Instruct-FP8`) with
`XDG_CACHE_HOME`, `VLLM_CACHE_ROOT`, `FLASHINFER_WORKSPACE_BASE`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR` and
`TMPDIR` also on local disk. Once up, one server judged all six systems (about 2.5 h).

## Metrics and references

`table.csv` holds each benchmark's native aggregate: PhyEditBench overall (1–10; per-dimension means, then
weighted), PICABench Acc (%), RISEBench score (1–5) and Acc (%), and ImgEdit (1–5). **Overall (0–1)** is the mean
over all 200 items of the primary metric rescaled to 0–1 of its range. The primary metrics are PhyEditBench
overall, PICABench Acc, RISE score and ImgEdit score, so benchmarks weigh 60:60:60:20.

References, judged exactly like models: **copy** returns the input unchanged at the generation size. **random**
is an unrelated image. `gt` images (the PhyEditBench GT next state) are shown in the gallery and never scored;
none of the 60 sampled RISE items has a reference image, so RISE shows its text reference. With this judge, copy
scores above every model on PhyEditBench; see `../v2_suite_1/results/DIAGNOSIS.md` (Cause 1).

Current numbers (`results/table.csv`): the originals, the T2_1 / T5 / T6 zero-shot rows (2026-10-03), the
`*_ft` rows (2026-09-28, the 1,280-row / 2-epoch probe) and the `*_probe` rows (the 800-row / 1-epoch probe;
`base_probe` was trained on 2026-09-28, the other three on 2026-09-29; `T0/T2/T3_probe` were never run):

| System | PhyEditBench | PICABench Acc % | RISE score | RISE Acc % | ImgEdit | Overall |
|---|---:|---:|---:|---:|---:|---:|
| base | 7.34 | 58.6 | 2.91 | 10.0 | 4.80 | 0.625 |
| T0 | 7.55 | 49.0 | 2.29 | 10.0 | 4.57 | 0.551 |
| T2 | 7.78 | 50.7 | 2.53 | 5.0 | 4.82 | 0.588 |
| T3 | 7.56 | 49.4 | 2.50 | 10.0 | 4.85 | 0.575 |
| **T2_1** | 7.78 | 53.7 | 2.77 | 10.0 | 4.58 | 0.609 |
| **T5** | 7.46 | 53.1 | 2.94 | 13.3 | 4.55 | 0.608 |
| **T6** | 7.46 | 52.9 | 2.78 | 8.3 | 4.78 | 0.602 |
| base_ft | 7.55 | 54.4 | 2.31 | 10.0 | 4.60 | 0.570 |
| T0_ft | 7.31 | 54.4 | 2.75 | 8.3 | 4.78 | 0.600 |
| T2_ft | 7.28 | 54.7 | 2.83 | 11.7 | 4.68 | 0.603 |
| T3_ft | 7.22 | 50.5 | 2.42 | 11.7 | 4.70 | 0.558 |
| base_probe | 7.58 | 55.8 | 2.52 | 5.0 | 4.75 | 0.595 |
| **T2_1_probe** | 7.77 | 55.3 | 2.82 | 11.7 | 4.62 | 0.618 |
| **T5_probe** | 7.42 | 53.0 | 2.72 | 11.7 | 4.78 | 0.597 |
| **T6_probe** | 7.42 | 53.2 | 2.65 | 15.0 | 4.77 | 0.591 |
| copy | 7.86 | 24.5 | 1.51 | 6.7 | 3.17 | 0.394 |
| random | 5.00 | 28.9 | 1.06 | 0.0 | 1.68 | 0.242 |

**Does edit-SFT change the base vs. T0/T2/T3 ranking?** Yes. Without edit-SFT, `base` (0.625) is clearly ahead
of `T2`/`T3`/`T0` (0.588/0.575/0.551). Edit-SFT hits `base` hardest (−0.055 → 0.570) while lifting `T0` most
(+0.049 → 0.600) and `T2` slightly (+0.015 → 0.603); `T3` drops a little (−0.017 → 0.558). Net: the FT order is
`T2_ft` (0.603) > `T0_ft` (0.600) > `base_ft` (0.570) > `T3_ft` (0.558) — `base` is no longer on top, and `T0`
goes from worst to second-best. This matches the direction (not the exact size — that run used PhyEditBench
only, a different subset, and a null-FT control) of `../v2_suite_1/results/DIAGNOSIS.md` Cause 2: FT closes the
gap mostly by lowering `base`, not by raising the T-series.

**How do T2_1 / T5 / T6 compare?** Zero-shot, all three (Overall 0.609 / 0.608 / 0.602) sit above T0/T2/T3
(0.551–0.588) and just below `base` (0.625; −0.016 / −0.017 / −0.023). PICABench carries most of that: 52.9–53.7 %
against 49.0–50.7 % for T0/T2/T3, though `base` is still at 58.6 %. `T5` matches `base` on RISE score (2.94 vs
2.91); `T2_1` and `T5` are lower than `base` on ImgEdit (4.58 / 4.55 vs 4.80). On PhyEditBench `T2_1` ties `T2`
(7.78) and all three are at or above `base` (7.34), but `copy` (7.86) is above every model, as before.
With the edit-SFT probe, `base` again drops most (`base_probe` 0.595, −0.030). `T2_1_probe` rises slightly to 0.618
(+0.009), the best of all probe/ft systems, and `T5_probe` / `T6_probe` fall by about 0.01 (0.597 / 0.591). After
the probe the order is `T2_1_probe` > `T5_probe` ≈ `base_probe` > `T6_probe`, but the unprobed `base` (0.625)
still leads every system. Compare `*_probe` rows with `base_probe`, not with the `*_ft` rows (different data and
epochs). All Overall differences among base, T2_1, T5, T6 and their probes are ≤ 0.034 (base vs `T6_probe`), well
inside the roughly 0.1 noise bound stated below: n = 200, one seed, one judge. Only the PICABench gap to T0/T2/T3 (about 3–5 points on 60 items) is even
suggestive, and it is not a significance test.

Coverage: all 17 systems have an image and a primary score for all 200 uids (`results/missing.csv` is empty), with
0 unparsed judge rows. Per-item values match v2_suite_1's `results/scores_item.csv` exactly for base, T0, T2, T3
and copy (3,390 rows, max difference 0). With n = 20–60 per benchmark, differences below roughly 0.1 normalised
are within noise; use v2_suite_1 for confirmatory tests.
