# S0-S3(-S4) baseline — full comparison (6 checkpoints + untrained reference)

RealWorldQA, SeedBench, and WorldPrediction (WM+PP), run against the six S0-S3(-S4) baseline
stage-2 (`-4task500sft-merged`) checkpoints plus the true untrained InternVL-U model as a
reference point. `s0`/`s1`/`s2`/`s3` run 2026-09-09 sequentially on a single free GPU (idx 7),
`vbvr-4task-sft-base`/`s4` run 2026-09-08 concurrently on two GPUs (idx 6/7), and the true
`InternVL-U-base` row's SeedBench cell run 2026-09-10 on GPU 7 (its RealWorldQA/WM/PP cells
were already available from an earlier, separate run — see the naming note below) — see
`inference/README.md` / `suites/worldprediction/README.md` for how each was run, and the
"Known run-quality notes" section below for the concurrent-run race condition that S0-S3
avoided by running strictly sequentially.

**Naming note**: `vbvr-4task-sft-base` is the stage-2 target_pred-SFT checkpoint trained
*from* the base InternVL-U snapshot (no stage-1 pretrain — vbvr's own "control"). It is a
different model from the **`InternVL-U-base`** row above it, which is the actual untrained
InternVL-U model with no vbvr SFT applied at all — the true zero-point every other row's
gen-SFT curriculum starts from. (Earlier versions of this table called the vbvr control
checkpoint plain "base", which read as if it were this row — renamed to avoid that
confusion.) `InternVL-U-base`'s WM/PP numbers are sourced from
`suites/worldprediction/results/internvlu_base_rerun_4f/` and its RealWorldQA from
`inference/vlmevalkit/reference_results/internvlu_fixed/InternVL-U-base/` — both pre-existing,
unrelated to this table's own run. A different control-framing F1 metric for the same
untrained model also exists at `suites/worldprediction/results/control_group/base_{wm,pp}/`
(28.2%/25.6% WM/PP) — see `suites/worldprediction/results/BASE_MODEL_PROVENANCE.md` for why
that number differs from the 36.4%/27.9% used here (different methodology, not disagreement).

| Checkpoint | RealWorldQA (n=765) | SeedBench (n=14,232) | WorldPrediction-WM (n=612) | WorldPrediction-PP (n=390) |
|---|--:|--:|--:|--:|
| **InternVL-U-base** (true, untrained — reference point, not part of the S0-S3(-S4) curriculum) | 56.60% | 73.64% | 36.4% (223/612) | 27.9% (109/390) |
| vbvr-4task-sft-base | 56.34% | 73.66% | 37.6% (230/612) | 27.7% (108/390) |
| s0 | 56.99% | 73.71% | 36.3% (222/612) | 29.5% (115/390) |
| s1 | 57.25% | 73.95% | **38.7%** (237/612) | 27.9% (109/390) |
| s2 | 56.73% | 73.62% | 36.3% (222/612) | 29.2% (114/390) |
| s3 | 55.95% | 73.77% | 28.3% (173/612) | 27.4% (107/390) |
| s4 | 56.34% | 73.38% | **17.8%** (109/612) | 24.9% (97/390) |

## Reading this

General spatial-VQA (RealWorldQA, SeedBench — single-image MCQ, no temporal reasoning) is
**flat across all seven rows, including the untrained model** (RealWorldQA: 55.95–57.25%, a
1.3pt spread; SeedBench: 73.38–73.95%, a 0.57pt spread — `InternVL-U-base` itself sits right in
the middle of both ranges, 56.60%/73.64%) — matches the same flat pattern
`suites/worldprediction/results/FULL_BENCHMARK_TABLE.md` found for an earlier, unrelated set of
IntPhys2-SFT checkpoints. Neither vbvr SFT nor the S0-S3(-S4) gen-SFT curriculum moves general
spatial reasoning at all, in either direction, relative to the untrained model.

WorldPrediction (temporal/procedural: pick the action, or ordered action sequence, that
explains a state transition) tells a real, different story:

- **WM declines from s1 onward**: 38.7% (s1, the peak) → 36.3% (s2) → 28.3% (s3) → **17.8%**
  (s4) — more than halved from peak to s4. The untrained `InternVL-U-base` (36.4%), s0 (36.3%),
  and vbvr-4task-sft-base (37.6%) all sit close together near the top — vbvr's own 4-task SFT
  doesn't move WM either, essentially replicating the untrained model's score — so the decline
  is specific to the S0-S3(-S4) gen-SFT curriculum, not a byproduct of any SFT in general, and
  isn't purely "more stages = worse" (s0→s1 actually improves slightly) — it looks more like a
  late-stage (s2 onward) degradation that compounds sharply at s4.
- **PP is much flatter**: `InternVL-U-base` (27.9%), vbvr-4task-sft-base (27.7%), and s0-s3
  (27.4–29.5%) are all within ~2pts of each other with no clear trend, only dropping at s4
  (24.9%) — the same late/final-stage pattern as WM, just much less severe.

This is the first evidence in this repo that the S0-S3(-S4) gen-SFT pretraining curriculum
affects *temporal* world-modeling differently than it affects static spatial-VQA, that the
effect isn't monotonic across all four early stages — it concentrates in the back half (s2→s3→
s4) — and, with the untrained-model reference point now in the table, that it's specifically
*this* curriculum causing the WM decline rather than SFT/fine-tuning in general (vbvr's own SFT
recipe leaves WM/PP essentially at the untrained baseline). Worth reconciling against
`suites/vbvr/`'s own task-level results (s4 is vbvr's *strongest* checkpoint per that suite's
scoring — see `project_magicbrush_s0s4_eval` / vbvr artifacts), which makes this WM/PP decline
a genuine tension worth investigating, not dismissing as noise: whatever makes s4 strongest on
vbvr's tasks may be actively costing it temporal world-modeling ability.

## Source files

- RealWorldQA / SeedBench: `inference/vlmevalkit/outputs_s0s3_baseline/InternVL-U-
  {base,vbvr-4task-sft-base,s0,s1,s2,s3,s4}[-4task500sft]/InternVL-U-{...}_
  {RealWorldQA,SEEDBench_IMG}_acc.csv` (the untrained model's directory has no `-4task500sft`
  suffix — it's `InternVL-U-base/`, not `InternVL-U-base-4task500sft/`)
- WorldPrediction: `suites/worldprediction/results/internvlu_{vbvr-4task-sft-base,s0,s1,s2,s3,
  s4}_4task500sft_4f/{WM,PP}_results.json` for the six trained checkpoints;
  `suites/worldprediction/results/internvlu_base_rerun_4f/{WM,PP}_results.json` for the
  untrained model (`mean_accuracy` field in all cases)

## Known run-quality notes

- A handful of WorldPrediction samples were skipped for **every** row identically (same
  missing COIN videos `LYJNYfwKc5o.mp4`/`OaTttCmD_RM.mp4`, same corrupt EPIC-KITCHENS packet in
  `P30_08.MP4`) — affects all seven rows equally, not a confound between them.
- `vbvr-4task-sft-base` and `s4` were run **concurrently** (different GPUs, shared image
  cache) and hit a write race on VLMEvalKit's `~/LMUData/images/` cache —
  vbvr-4task-sft-base's SeedBench and s4's RealWorldQA each failed on a corrupted/truncated
  cached image on the first attempt, both cleanly rerun individually afterward. `s0`-`s3` were
  run strictly **sequentially** on one GPU specifically to avoid a repeat of this race, and had
  no such failures.
