# eval/

```
suites/<name>/   one eval type per suite — code specific to scoring one kind of task
```

Ten suites so far: **`suites/vbvr/`** (migrated from `Evaluation/VBVR-CustomEval/`),
**`suites/magicbrush/`** (migrated from `Evaluation/editing_benchmarks/MagicBrush/`, InternVL-U
only — see its own README for why), **`suites/risebench/`** / **`suites/aurorabench/`** /
**`suites/imgedit/`** / **`suites/phyeditbench/`** (new, set up directly in this repo, same
InternVL-U-only / inference-only scope as magicbrush — data freshly downloaded/cloned, official
scorer ported only for `imgedit`, which ships a redistributable local judge checkpoint; the
other three document their GPT-based official scorer but don't port it), **`suites/worldprediction/`**
(a git submodule, not a plain migrated dir — see its own section below), and
**`suites/realworldqa/`** / **`suites/seedbench/`** (thin wrappers, not standalone kits — both
just dataset names inside the shared `eval/inference/vlmevalkit/` harness; see its own section
below). No suite's code has been generalized to be suite-agnostic — each is a self-contained
`inference/` (+ `evaluators/`/`scorers/` + `artifacts/` + `results/` where scoring has been
ported) tree, same shape but no shared code between them yet. Worth a look before an eighth
suite is added: every non-submodule/non-wrapper suite's `inference/` scripts hardcode an
absolute path to `Model_Related/InternVLU/InternVL-U` for the `internvlu` package import (see
"Known dependency gap" in `suites/magicbrush/README.md`) — that's the first concrete candidate
for something actually shared (a proper `training/models/internvl-u` inference-package fix or
an `eval/inference/` helper), rather than inventing a generic layer speculatively.

`eval/inference/` (sibling to `suites/`, documented in its own `README.md`) is the home for
generic model-inference runners shared across suites — currently just `vlmevalkit/`, a trimmed
vendored snapshot of `Evaluation/VLMEvalKit/` from the old tree.

Excluded from migration (debug/scratch material, not part of the reproducible pipeline):
`debug_out/`, `eval_samples/`, `scratch/`.

## `v2_suite_1/`

Self-contained multi-benchmark evaluation of the v2 temporal-SSL checkpoints (base vs T0/T2/T3):
1,210 tasks from PhyEditBench, PICABench, RISEBench, ImgEdit and MagicBrush, grouped into
target / secondary / control evidence, judged with each benchmark's official judge code on a local
Qwen3-VL-30B-A3B-FP8, plus a diagnosis of why the trained models score below base. Code, results,
reports and the shareable page all live under it; start at [`v2_suite_1/README.md`](v2_suite_1/README.md).

## `suites/vbvr/`

Scores stage-2 target_pred checkpoints (`internvlu-{base,s0,s1,s2,s3}-4task500sft`) via
rule-based scorers (`scorers/` + `validation/score_target_pred_eval_v2.py` driver) and an LLM
judge (`judge_eval/score_all_with_judge.py`, using `evaluators/llm_judge_full.py`). Comparison
report: `artifacts/gen_s0s3_4task_comparison.py` → `artifacts/s0s3_4task_comparison.html`. Full
detail, including which scorer-output filename is current vs. superseded
(`scored_new.json` vs. the older `scored.json`): `../../PROVENANCE.md` §5.

## `suites/magicbrush/`

Scores InternVL-U (base + 8 SSL/fine-tuned checkpoints) on MagicBrush single-turn instruction
editing: `inference/gen_magicbrush_internvlu.py` generates, `evaluators/eval_magicbrush.py`
scores (L1/L2/CLIP-I/DINO/CLIP-T, ported from the official `image_eval.py`). Comparison report:
`artifacts/gen_magicbrush_report.py` → `artifacts/magicbrush_comparison.md`. Test data (public
mirror of the withheld official test split) lives outside the repo — see the suite's README.

## `suites/risebench/`

Generation-only (inference, no scoring ported yet): `inference/gen_risebench_internvlu.py`
runs InternVL-U over the 360-item official RISEBench set (4 reasoning categories — temporal,
causal, spatial, logical; no GT image, scored upstream by GPT-judge). Data lives outside the
repo (third-party licensed) — see the suite's README, including a note on HF's Xet backend
stalling `snapshot_download` and the direct-`curl` workaround used to fetch it.

## `suites/aurorabench/`

Generation-only (inference, no scoring ported yet): `inference/gen_aurorabench_internvlu.py`
runs InternVL-U over the 400-item official AURORA-Bench `test` split (8 source datasets × 50;
no GT image, scored upstream by LMM judge / human raters). `extract_from_hf.py` materializes
the HF parquet (no stable per-row id) into a json + `images/` layout. Data lives outside the
repo (third-party licensed) — see the suite's README. Not to be confused with this repo's
existing custom EPIC-derived AURORA-*style* evals under `results/04_unified_benchmarks/editing/`.

## `suites/imgedit/`

Generation + local-judge scoring: `inference/gen_imgedit_internvlu.py` runs InternVL-U over the
737-item official ImgEdit-Bench Basic-Bench set (9 edit-type categories; no GT image).
`evaluators/imgedit_judge.py` + `evaluators/score_imgedit.py` port the official scoring via the
**local** `ImgEdit_Judge` checkpoint (a redistributable Qwen2.5-VL-7B fine-tune upstream ships as
a non-API alternative to its GPT-4o judge) — the only inference-only suite here with a ported
scorer besides `magicbrush`. Data (48M, small enough to vendor whole) and the judge checkpoint
(16.6GB) both live outside the repo — see the suite's README.

## `suites/phyeditbench/`

Generation-only (inference, no scoring ported yet): `inference/gen_phyeditbench_internvlu.py`
runs InternVL-U over the 238 real four-state physical-editing trajectories (expanded into 1190
Type A-E generation tasks per the official protocol) plus 35 synthetic Anti-Physics items from
[Previsior/PhyEditBench](https://github.com/Previsior/PhyEditBench) — 1225 items total; no GT
image for most tasks, scored upstream by a GPT-4o judge across 4 dimensions (Consistency,
Instruction Following, Physical Plausibility, Image Quality). Data (1.3G, plain git clone, not
Xet-backed) lives outside the repo — see the suite's README.

## `suites/worldprediction/`

**A git submodule**, not a plain migrated dir — the only other one besides
`training/models/internvl-u/`. Upstream: [facebookresearch/WorldPrediction](
https://github.com/facebookresearch/WorldPrediction) (video-based World Modeling / Procedural
Planning MCQ tasks: pick the action, or ordered action sequence, that explains an observed
state transition). Documented here rather than with a suite-level `README.md` because, like
`training/models/internvl-u`, the suite dir *is* the submodule root — see its own
`README.md`/`CONTROL_SETTINGS_README.md`/`INTPHYS2_CONTROL_README.md` inside for upstream and
control-group protocol detail.

- Its local git history is two commits, not upstream's real history: `d0659ef` "Initial commit"
  (bare clone of upstream, made when this local fork was first set up) then `85e4124` (this
  migration pass — committed 56 previously-uncommitted working-tree files in one snapshot: the
  InternVL-U integration itself — `model/vlms/internvlu.py`, `configs/vlm/internvlu/` (30+
  checkpoint configs), `run_internvlu.sh`/`run_internvlu_sft.sh`, `evaluator.py`/`model/vlm.py`/
  `model/__init__.py`/`run.py` changes — plus equivalent integrations for bagel/qwen3vl/
  sensenova-u1/cosmos3, IntPhys2-derived control-group settings, and general control-group
  tooling. Same **caveat as internvl-u**: `.gitmodules` URL is an absolute local path
  (`/scratch/network/ssd2/junlin/ssl_mllm/Evaluation/WorldPrediction`) — re-point before cloning
  this repo elsewhere.
- **Known quirk**: `coin/annotations` is committed as an embedded git repo (mode `160000`) with
  no matching `.gitmodules` entry — git warned about this on commit. It was already like that in
  the source tree; harmless for local use (the dir just won't populate on a fresh clone/`git
  submodule update`) but worth fixing (either a real nested submodule entry or de-embedding it)
  if this ever needs to clone cleanly elsewhere.
- **Not yet done**: evaluation data. `run_internvlu.sh` reads `data/WorldPrediction-{WM,PP}.json`
  plus the video sources those reference (COIN, CrossTask, IKEA-ASM — `download_*.py`/`.sh`
  scripts exist for these but haven't been re-run/verified from this repo). Same "code migrated,
  raw data deferred" status as the 5 `data/v1/sources/` — not yet traced against PROVENANCE.md.
- **Not yet done**: no comparison-report step ported into `eval/suites/vbvr/artifacts/`-style
  reporting yet — `compare_wp.py` (inside the submodule) is the closest existing tool.
- **Base-checkpoint reference results** (added 2026-09-08, commit `af65e93` inside the
  submodule): `results/internvlu_base_rerun_4f/{WM,PP}_results.json` (plain accuracy, most
  complete sample count), `results/internvlu_4f/` (earlier partial run), and
  `results/control_group/base_{wm,pp}/` (source data for a *different* control-framing F1
  metric, summarized in `results/CONTROL_MASTER_TABLE.md`/`FULL_BENCHMARK_TABLE.md`) — all for
  the **raw, untrained InternVL-U snapshot**, not the S0-S3 baseline's `internvlu-
  vbvr-4task-sft-base-4task500sft` checkpoint (that plain "base"-sounding name was retired
  precisely because of this ambiguity — see `BASELINE_COMPARISON.md`'s naming note). See
  `results/BASE_MODEL_PROVENANCE.md` (inside the submodule) for
  which of the three WM/PP numbers is which — they're different methodologies, not conflicting
  measurements. All six `-4task500sft` checkpoints have since been run (see
  `../BASELINE_COMPARISON.md`), which also folds this untrained-model reference point into one
  combined table alongside them.

## `suites/realworldqa/` and `suites/seedbench/`

**Thin wrappers**, not standalone kits (see their own `README.md`s) — both are just `--data`
values (`RealWorldQA`, `SEEDBench_IMG`) inside the shared `eval/inference/vlmevalkit/` harness,
which already has an InternVL-U model adapter. Letter-MCQ, scored by deterministic exact-match
— no judge/evaluator code needed, unlike `vbvr`/`magicbrush`. RealWorldQA has a prior
(pre-S0S3-checkpoint) reference run to sanity-check the harness against; SeedBench has never
been run against this model family before, so validate against one checkpoint first (its option
format may stress the InternVL-U prompt adapter differently than CV-Bench/RealWorldQA did).
Neither has a comparison-report script ported yet — see each README for the shape one would
take (reading `_acc.csv` per checkpoint, same as `vbvr/artifacts/gen_s0s3_4task_comparison.py`
reads its scorer outputs).
