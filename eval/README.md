# eval/

```
suites/<name>/   one eval type per suite — code specific to scoring one kind of task
```

Two suites so far: **`suites/vbvr/`** (migrated from `Evaluation/VBVR-CustomEval/`) and
**`suites/magicbrush/`** (migrated from `Evaluation/editing_benchmarks/MagicBrush/`, InternVL-U
only — see its own README for why). Neither suite's code has been generalized to be
suite-agnostic — each is a self-contained `inference/` + `evaluators/`/`scorers/` + `artifacts/`
+ `results/` tree, same shape but no shared code between them yet. Now that there are two real
examples, worth a look before a third suite is added: both suites' `inference/` scripts hardcode
an absolute path to `Model_Related/InternVLU/InternVL-U` for the `internvlu` package import (see
"Known dependency gap" in `suites/magicbrush/README.md`) — that's the first concrete candidate
for something actually shared (a proper `training/models/internvl-u` inference-package fix or a
`eval/inference/` helper), rather than inventing a generic layer speculatively.

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
