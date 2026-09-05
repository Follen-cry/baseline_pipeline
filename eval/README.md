# eval/

```
suites/<name>/   one eval type per suite — code specific to scoring one kind of task
```

Currently one suite: **`suites/vbvr/`** (migrated from `Evaluation/VBVR-CustomEval/`). Its
internal layout (`scorers/`, `judge_eval/`, `evaluators/`, `validation/`, `inference/`,
`artifacts/`, `tests/`) is kept exactly as it was — none of this code has been generalized to be
suite-agnostic yet, so there's no honest "generic inference/reports layer" to factor out today.
If a second suite is added later (e.g. for one of the benchmarks in the old
`DATA_ACCESS_GUIDE.md` — MVP, CausalVQA, WorldPrediction), that's the point to look for what's
actually shared between the two and pull it up a level; don't invent that abstraction now with
only one real example to generalize from.

Excluded from migration (debug/scratch material, not part of the reproducible pipeline):
`debug_out/`, `eval_samples/`, `scratch/`.

## `suites/vbvr/`

Scores stage-2 target_pred checkpoints (`internvlu-{base,s0,s1,s2,s3}-4task500sft`) via
rule-based scorers (`scorers/` + `validation/score_target_pred_eval_v2.py` driver) and an LLM
judge (`judge_eval/score_all_with_judge.py`, using `evaluators/llm_judge_full.py`). Comparison
report: `artifacts/gen_s0s3_4task_comparison.py` → `artifacts/s0s3_4task_comparison.html`. Full
detail, including which scorer-output filename is current vs. superseded
(`scored_new.json` vs. the older `scored.json`): `../../PROVENANCE.md` §5.
