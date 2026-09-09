# S0-S3(-S4) baseline — full 6-checkpoint comparison

RealWorldQA, SeedBench, and WorldPrediction (WM+PP), run against all six S0-S3(-S4) baseline
stage-2 (`-4task500sft-merged`) checkpoints: `base` (no stage-1 pretrain, control),
`s0`/`s1`/`s2`/`s3` run 2026-09-09 sequentially on a single free GPU (idx 7), `base`/`s4` run
2026-09-08 concurrently on two GPUs (idx 6/7) — see `inference/README.md` /
`suites/worldprediction/README.md` for how each was run, and the "Known run-quality notes"
section below for the base/s4 concurrent-run race condition that S0-S3 avoided by running
strictly sequentially.

| Checkpoint | RealWorldQA (n=765) | SeedBench (n=14,232) | WorldPrediction-WM (n=612) | WorldPrediction-PP (n=390) |
|---|--:|--:|--:|--:|
| base | 56.34% | 73.66% | 37.6% (230/612) | 27.7% (108/390) |
| s0 | 56.99% | 73.71% | 36.3% (222/612) | 29.5% (115/390) |
| s1 | 57.25% | 73.95% | **38.7%** (237/612) | 27.9% (109/390) |
| s2 | 56.73% | 73.62% | 36.3% (222/612) | 29.2% (114/390) |
| s3 | 55.95% | 73.77% | 28.3% (173/612) | 27.4% (107/390) |
| s4 | 56.34% | 73.38% | **17.8%** (109/612) | 24.9% (97/390) |

## Reading this

General spatial-VQA (RealWorldQA, SeedBench — single-image MCQ, no temporal reasoning) is
**flat across all six checkpoints** (RealWorldQA: 55.95–57.25%, a 1.3pt spread; SeedBench:
73.38–73.95%, a 0.57pt spread) — matches the same flat pattern
`suites/worldprediction/results/FULL_BENCHMARK_TABLE.md` found for an earlier, unrelated set of
IntPhys2-SFT checkpoints. Whatever the S0-S3(-S4) gen-SFT curriculum is doing, it isn't moving
general spatial reasoning.

WorldPrediction (temporal/procedural: pick the action, or ordered action sequence, that
explains a state transition) tells a real, different story:

- **WM declines from s1 onward**: 38.7% (s1, the peak) → 36.3% (s2) → 28.3% (s3) → **17.8%**
  (s4) — more than halved from peak to s4. s0 (36.3%) and base (37.6%) sit close together near
  the top, so the decline isn't purely "more stages = worse" (s0→s1 actually improves slightly)
  — it looks more like a late-stage (s2 onward) degradation that compounds sharply at s4.
- **PP is much flatter**: 27.4–29.5% across s0-s3 (no clear trend), only dropping at s4
  (24.9%) — the same late/final-stage pattern as WM, just much less severe.

This is the first evidence in this repo that the S0-S3(-S4) gen-SFT pretraining curriculum
affects *temporal* world-modeling differently than it affects static spatial-VQA, and that the
effect isn't monotonic across all four early stages — it concentrates in the back half (s2→s3→
s4). Worth reconciling against `suites/vbvr/`'s own task-level results (s4 is vbvr's *strongest*
checkpoint per that suite's scoring — see `project_magicbrush_s0s4_eval` / vbvr artifacts),
which makes this WM/PP decline a genuine tension worth investigating, not dismissing as noise:
whatever makes s4 strongest on vbvr's tasks may be actively costing it temporal world-modeling
ability.

## Source files

- RealWorldQA / SeedBench: `inference/vlmevalkit/outputs_s0s3_baseline/InternVL-U-{base,s0,s1,
  s2,s3,s4}-4task500sft/InternVL-U-{...}-4task500sft_{RealWorldQA,SEEDBench_IMG}_acc.csv`
- WorldPrediction: `suites/worldprediction/results/internvlu_{base,s0,s1,s2,s3,s4}_4task500sft_
  4f/{WM,PP}_results.json` (`mean_accuracy` field)

## Known run-quality notes

- A handful of WorldPrediction samples were skipped for **every** checkpoint identically (same
  missing COIN videos `LYJNYfwKc5o.mp4`/`OaTttCmD_RM.mp4`, same corrupt EPIC-KITCHENS packet in
  `P30_08.MP4`) — affects all six checkpoints equally, not a confound between them.
- `base` and `s4` were run **concurrently** (different GPUs, shared image cache) and hit a
  write race on VLMEvalKit's `~/LMUData/images/` cache — base's SeedBench and s4's RealWorldQA
  each failed on a corrupted/truncated cached image on the first attempt, both cleanly rerun
  individually afterward. `s0`-`s3` were run strictly **sequentially** on one GPU specifically
  to avoid a repeat of this race, and had no such failures.
