# eval/

```
suites/<name>/   one eval type per suite — code specific to scoring one kind of task
```

Seven suites so far: **`suites/vbvr/`** (migrated from `Evaluation/VBVR-CustomEval/`),
**`suites/magicbrush/`** (migrated from `Evaluation/editing_benchmarks/MagicBrush/`, InternVL-U
only — see its own README for why), **`suites/risebench/`** / **`suites/aurorabench/`**
(new, set up directly in this repo, same InternVL-U-only / inference-only scope as magicbrush —
data freshly downloaded from HF, no evaluator ported yet), **`suites/worldprediction/`**
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
  raw data deferred" status as the 5 `data/sources/` — not yet traced against PROVENANCE.md.
- **Not yet done**: no comparison-report step ported into `eval/suites/vbvr/artifacts/`-style
  reporting yet — `compare_wp.py` (inside the submodule) is the closest existing tool.

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
