# base vs. S4 — WorldPrediction / RealWorldQA / SeedBench

First real comparison run of the S0-S3(-S4) baseline's `base` and `s4` stage-2
(`-4task500sft-merged`) checkpoints across the three newly-integrated suites (2026-09-08, GPUs
6/7 — see `inference/README.md` / `suites/worldprediction/README.md` for how each was run).

| Benchmark | base | S4 | Δ (S4 − base) |
|---|--:|--:|--:|
| RealWorldQA (n=765) | 56.34% | 56.34% | 0.0 |
| SeedBench (n=14,232) | 73.66% | 73.38% | −0.28 |
| WorldPrediction-WM (n=612) | 37.6% (230/612) | 17.8% (109/612) | **−19.8** |
| WorldPrediction-PP (n=390) | 27.7% (108/390) | 24.9% (97/390) | **−2.8** |

## Reading this

General spatial-VQA (RealWorldQA, SeedBench — single-image MCQ, no temporal reasoning) is
**essentially flat** between base and S4, matching the same flat pattern
`suites/worldprediction/results/FULL_BENCHMARK_TABLE.md` found for an earlier, unrelated set of
IntPhys2-SFT checkpoints — general spatial reasoning doesn't move much regardless of what
downstream SFT stage a checkpoint went through.

WorldPrediction (temporal/procedural: pick the action, or ordered action sequence, that
explains a state transition — COIN/CrossTask/IKEA-ASM/EPIC-KITCHENS-100 video pairs, not static
images) tells a different story: **S4 regresses substantially on WM (37.6%→17.8%, more than
halved) and moderately on PP (27.7%→24.9%)**. This is the first evidence in this repo that the
S0-S3(-S4) gen-SFT pretraining curriculum affects *temporal* world-modeling differently than it
affects static spatial-VQA — worth checking against the other four checkpoints (s0/s1/s2/s3) to
see whether this is monotonic across stages or S4-specific, and worth checking whether the same
divergence shows up in `suites/vbvr/`'s own task-level results (S4 is vbvr's strongest
checkpoint per that suite's scoring — see `project_magicbrush_s0s4_eval` / vbvr artifacts —
which makes the WM/PP regression here a genuine tension worth investigating, not dismissing as
noise).

## Source files

- RealWorldQA / SeedBench: `inference/vlmevalkit/outputs_s0s3_baseline/InternVL-U-{base,s4}-
  4task500sft/InternVL-U-{base,s4}-4task500sft_{RealWorldQA,SEEDBench_IMG}_acc.csv`
- WorldPrediction: `suites/worldprediction/results/internvlu_{base,s4}_4task500sft_4f/
  {WM,PP}_results.json` (`mean_accuracy` field)

## Known run-quality notes

- A handful of WorldPrediction samples were skipped for both models identically (same missing
  COIN video `LYJNYfwKc5o.mp4`/`OaTttCmD_RM.mp4`, same corrupt EPIC-KITCHENS packet in
  `P30_08.MP4`) — affects both checkpoints equally, not a base-vs-S4 confound.
- RealWorldQA and SeedBench were initially run **concurrently** for both checkpoints against a
  shared image-cache directory (`~/LMUData/images/`) and hit a write race — base's SeedBench
  and S4's RealWorldQA each failed on a corrupted/truncated cached image on the first attempt.
  Both were re-run individually (sequentially, no concurrent writer) and succeeded; the numbers
  above are from the clean reruns, not the racy first attempt.
