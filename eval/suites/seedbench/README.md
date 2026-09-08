# SeedBench

**Scope of this suite: a thin wrapper, not a standalone kit.** Same shape as `../realworldqa/`
— SeedBench isn't its own benchmark repo here, just the `SEEDBench_IMG` `--data` value inside
the shared harness at `../../inference/vlmevalkit/`. See `../../inference/README.md` for what's
vendored there and the checkpoint-registry fix that makes the S0-S3 baseline checkpoints
selectable (`InternVL-U-{base,s0,s1,s2,s3,s4}-4task500sft`).

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
MODELS="InternVL-U-base-4task500sft InternVL-U-s0-4task500sft InternVL-U-s1-4task500sft \
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

**Higher risk than RealWorldQA here**: this is the first time this harness's InternVL-U prompt
handling (`vlmeval/vlm/internvlu.py`'s `build_prompt`) meets SeedBench's option format, which
may differ from CV-Bench/RealWorldQA's. Run `--data SEEDBench_IMG` against a single checkpoint
first and manually inspect a few rows of the resulting `.xlsx` before running all six.
