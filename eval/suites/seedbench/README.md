# SeedBench

**Scope of this suite: a thin wrapper, not a standalone kit.** Same shape as `../realworldqa/`
— SeedBench isn't its own benchmark repo here, just the `SEEDBench_IMG` `--data` value inside
the shared harness at `../../inference/vlmevalkit/`. See `../../inference/README.md` for what's
vendored there and the checkpoint-registry fix that makes the S0-S3 baseline checkpoints
selectable (`InternVL-U-{vbvr-4task-sft-base,s0,s1,s2,s3,s4}-4task500sft`).

## What it is

`SEEDBench_IMG`: ~14K image multiple-choice questions across 9 dimensions (scene/instance
understanding, instance identity/attributes/location/counting, spatial relations, instance
interaction, visual reasoning, text understanding). Self-downloads its TSV on first run
(`vlmeval/dataset/image_mcq.py`'s `DATASET_URL['SEEDBench_IMG']`) — no local data staging
needed, unlike `risebench`/`aurorabench`.

## Known-good reference run

None yet — unlike RealWorldQA, SeedBench was never run against any InternVL-U checkpoint in
the old tree (checked: no `*SEEDBench*` files anywhere under the old `Evaluation/VLMEvalKit/
outputs_*`). First real run here is the harness's first SeedBench run for this model family.

## Running it

```bash
cd ../../inference/vlmevalkit
MODELS="InternVL-U-vbvr-4task-sft-base-4task500sft InternVL-U-s0-4task500sft InternVL-U-s1-4task500sft \
        InternVL-U-s2-4task500sft InternVL-U-s3-4task500sft InternVL-U-s4-4task500sft" \
DATA="SEEDBench_IMG" \
WORK_DIR="outputs_s0s3_baseline" \
bash scripts/run_internvlu.sh
```

Output: `outputs_s0s3_baseline/<model>/<model>_SEEDBench_IMG.xlsx` (+ `_acc.csv`,
`_exact_matching_result.{pkl,xlsx}`) per checkpoint — same `WORK_DIR` as `../realworldqa/` can
be reused (it's fresh per `--data`+`--model` pair, not per benchmark). No report script ported
here yet.

## Validate-then-run

**Confirmed working (2026-09-08, GPU 7)**: a single-sample smoke test against
`InternVL-U-vbvr-4task-sft-base-4task500sft` built the dataset (14,232 rows, TSV auto-downloaded), loaded the
real merged checkpoint, and produced a coherent MCQ answer — the prompt-format risk originally
flagged here (SeedBench's options differing from CV-Bench/RealWorldQA's) did not materialize.
Still worth running one checkpoint's full pass before all six, just to catch anything
sample-count-dependent (rate limits, edge-case images) the single-row test wouldn't surface.
