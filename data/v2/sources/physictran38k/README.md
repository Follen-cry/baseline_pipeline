# `data/v2/sources/physictran38k/` — PhysicTran38K (v2 source)

> **v2 pool (2026-09-26):** all 10,000 clips of `selection_10k.jsonl` are candidates (no trim to
> 5K, no moving-list exclusion); static windows are dropped by the pool's motion filter. Caption =
> the full `prompt`, verbatim (outcome descriptions are wanted).

- **Repo:** [`metazlb/PhysicTran38K`](https://huggingface.co/datasets/metazlb/PhysicTran38K)
  (same content is mirrored at `diffusion-cot/PhysicTran38K`), Apache-2.0, ~98 GB total.
- **What it is:** **model-generated** videos of physical state transitions (from PhysicEdit,
  arXiv 2602.21778). The taxonomy is 5 domains → 16 sub-domains → **46 transition types**, with
  ~1,000 videos per type (45,991 prompts in total) at ~0.15–0.25 MB each.
- **Per-type folder:** `N.mp4`, `metadata.json` (list of `{idx, prompt, State, Transition}`;
  `idx` N ↔ `N.mp4`), `final_filter_videos.txt` (the authors' final filtered list),
  `filtered_videos.txt` (a subset of the final list), `moving_videos.txt` (meaning
  undocumented; absent for 10 types), plus `intrinsics/ pose/ vipe/ mask/ rgb/` (camera and
  mask annotations, not downloaded).
- **Raw location:** `/scratch/network/ssd/junlin/raw/physictran38k/` (mirrors the repo paths).

## Filtering & sampling summary

**Category-stratified + authors' filter lists.**

1. Pool: 46 transition types (~1,000 model-generated videos each).
2. Tier A: every video in the authors' `final_filter_videos.txt` (7,371).
3. Tier B2: 2,629 unlisted videos water-filled into the types with the fewest tier-A videos (≥ 185 per type), seed 42.
4. No CV / VLM QC yet (`moving_videos.txt` recorded as `in_moving_list`; measured to correlate with camera motion).
5. Final: `selection_10k.jsonl` = 10,000 (1.5 GB); **to be trimmed to 5K**, preferring tier A and not in the moving list (5,678 such).

Details below.

## Selection (`download_physictran38k.py`, seed 42)

Target is a 10K oversample, later trimmed to 5K for v2.

| Tier | Rule | Count |
|---|---|---|
| A | in `final_filter_videos.txt` (all of them) | 7,371 |
| B2 | not in any list; water-filled into the types with the fewest tier-A videos | 2,629 |

The final list only holds 7,371 videos in total, so 10K requires 2,629 unfiltered ones (B1,
"in `filtered_videos.txt` but not final", turned out to be empty). Every type ends with ≥185
videos. The manifest is `selection_10k.jsonl` in the raw dir: one row per video with `tier`,
`in_moving_list`, `prompt`, and the domain/sub-domain/transition.

## Status (2026-09-24)

All 10,000 downloaded (1.53 GB), each size-checked against the repo listing, 200/200 random
samples decode. Download notes: free HF accounts get **1,000 API requests / 5 min**, and the
Xet path costs one API call per file, so the script disables Xet, throttles to 2.5 files/s,
pauses 5 min on any 429, and reuses the frozen `selection_10k.jsonl` instead of re-listing the
repo (≈ 1 h for 10K files).

**Full repo (2026-10-10):** `download_physictran38k.py --all` lists every type and writes
`RAW/manifest_all.jsonl` — **45,990 videos, 7.04 GB** (tier `A` = 7,371 in the final list,
`unlisted` = 38,619; `B1` is empty), same row schema as the 10K manifest plus
`in_selection_10k` (all 10,000 matched). Downloads the remaining 35,990 into the same tree
(≈ 4 h at 2.5 files/s; log `RAW/download_all.log`). `selection_10k.jsonl` is unchanged, so the
v2 pools (`main`, `phystran_g1`) are unaffected; using the extra clips needs a manifest switch
in `clip_sources.py`.

Video specs: **3.27 s, 15 fps (49 frames), 832×480**. So Δt ≤ ~1.0 s, and gaps should be set
in frames (0.25 s = 3.75 frames isn't exact; use 3 f = 0.2 s or 4 f = 0.267 s).

## Caveats for building v2 windows

- **Caption:** `prompt` describes start state → trigger → **final state** ("By the end …"), so
  it must not be used verbatim (it gives the target away). Planned: keep only the start state +
  trigger sentences.
- **`moving_videos.txt` ≈ camera motion.** On 25 vs 25 sampled videos, median frame-corner
  displacement (ORB + RANSAC homography) is 9.5 px for listed videos vs 3.1 px for unlisted, and
  48% vs 28% exceed 10 px. So exclude listed videos. Tier A minus the moving list = **5,678**,
  enough for 5K; still run the per-video camera-motion filter, since 28% of unlisted videos move too.
- When trimming to 5K, prefer tier A and not-moving.
