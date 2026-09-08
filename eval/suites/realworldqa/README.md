# RealWorldQA

**Scope of this suite: a thin wrapper, not a standalone kit.** Unlike `vbvr`/`magicbrush`/
`risebench`/`aurorabench`, RealWorldQA isn't its own benchmark repo — it's one `--data` value
inside the shared harness at `../../inference/vlmevalkit/`, which already has an InternVL-U
model adapter (`vlmeval/vlm/internvlu.py`) wired up. See `../../inference/README.md` for what's
vendored there and the checkpoint-registry fix that makes the S0-S3 baseline checkpoints
selectable (`InternVL-U-{base,s0,s1,s2,s3,s4}-4task500sft`).

## What it is

1,760 real-world spatial-reasoning multiple-choice questions (single image + question + 4
options), collected by X.AI. Letter-MCQ format — scored by deterministic exact-matching, no
GPT judge needed (`--judge exact_matching`).

## Known-good reference run

`../../inference/vlmevalkit/reference_results/internvlu_fixed/InternVL-U-{base,ffs,gen1ep,
gen3ep,...}/InternVL-U-*_RealWorldQA*` — an already-completed run, but against **pre-S0S3
exploratory checkpoints**, not the baseline's actual `-4task500sft` checkpoints. Useful as a
"the harness works end-to-end" sanity check, not as a baseline comparison point.

## Running it

```bash
cd ../../inference/vlmevalkit
MODELS="InternVL-U-base-4task500sft InternVL-U-s0-4task500sft InternVL-U-s1-4task500sft \
        InternVL-U-s2-4task500sft InternVL-U-s3-4task500sft InternVL-U-s4-4task500sft" \
DATA="RealWorldQA" \
WORK_DIR="outputs_s0s3_baseline" \
bash scripts/run_internvlu.sh
```

Output: `outputs_s0s3_baseline/<model>/<model>_RealWorldQA.xlsx` (+ `_acc.csv`,
`_exact_matching_result.{pkl,xlsx}`) per checkpoint. `_acc.csv` is the number to pull into a
comparison report — no report script has been ported here yet (would follow the shape of
`../vbvr/artifacts/gen_s0s3_4task_comparison.py`, reading the six `_acc.csv` files instead of
the rule-based/judge scorer outputs).

## Validate-then-run

Restrict `MODELS`/`DATA` to one checkpoint before committing to all six — a full RealWorldQA
run per checkpoint is 1,760 questions of VLM inference.
