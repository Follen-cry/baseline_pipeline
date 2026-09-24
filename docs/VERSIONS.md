# Versions

This repo holds two generations of the video self-supervised training pipeline. **v1 is frozen
in place** (no files moved or renamed; git tag `v1` pins it); **v2 is purely additive** — every
v2 file lives under a path containing `v2`, so a path alone tells you which version it belongs to.

| | **v1** — S0-S4 | **v2** — T0-T4 (temporal SSL) |
|---|---|---|
| Doc | [`v1.md`](v1.md) | [`v2.md`](v2.md) |
| Task family | 3 context frames → 1 generated frame; settings vary caption / target direction / MCQ / progression text | Caption-conditioned 4-frame window: 3 frames in, 1 frame generated, plus optional GAP / ORDER / MISSING text prediction |
| Frame sampling | Fixed 2 fps grid (0.5 s spacing; NWM native 4 fps), target fixed 0.5 s ahead | 4 equally spaced frames F0-F3, gap Δt sampled per window from a discrete set, e.g. {0.25, 0.5, 1.0, 2.0} s |
| Context order | Always ordered | Ordered (T0-T2) or shuffled (T3, T4-B) |
| Data sources | VBVR (5 tasks), EPIC-Kitchens, NWM/RECON, Panda-70M-EPIC, Panda-70M v1 | Overlapping real-video sources re-sampled + new sources (TBD); frames re-extracted by timestamp |
| Frames on disk | `/scratch/network/ssd/junlin/{vbvr_next_frame,epic_ssl_frames,nwm_data,panda70m_*_frames}/` | `/scratch/network/ssd/junlin/v2_frames/<source>/` |
| Stage-1 checkpoints | `/scratch/network/ssd2/junlin/models/internvlu-s{0..4}-gen(-merged)` | `/scratch/network/ssd/junlin/models/internvlu-v2-t{0..4}-gen(-merged)` |
| Git | tag `v1` (repo `a419e12`, internvl-u submodule `8685c94`) | `master` from `v1` onward |

## Where each version's files live

| Layer | v1 (unchanged locations) | v2 (new, additive) |
|---|---|---|
| Per-source processing | `data/sources/<name>/processing/*.py` | `data/sources/<name>/processing/v2/` (reused sources); `data/sources/<new_name>/` (v2-only sources, README says so) |
| Cross-source code | `data/common/*.py` | `data/common/v2/` |
| Recipes | `data/recipes/{s0s3_baseline,s4_progression,uniform_data_split_regarding_data_source}/` | `data/recipes/v2_temporal_ssl/` |
| Final datasets | `data/datasets/{final_s0s3,s4_progression,vbvr_target_pred_4task,worldprediction,uniform_*,*_Composed_intermediate*}/` | `data/datasets/v2_temporal_ssl/` |
| Meta | `data/meta/*.json` (no prefix) | `data/meta/v2_*.json` |
| Training launchers | `training/models/internvl-u/internvl_chat/shell/internvlu/{sft,orchestration}/*.sh` | `.../shell/internvlu/{sft,orchestration}/v2/` |
| Eval | `eval/suites/*` (all existing suites) | `eval/suites/v2_temporal_ssl/`; existing suites gain v2 checkpoints as extra rows |

**Rules for v2 work:** don't edit v1 files (fix a v1 bug in its own commit and say so); don't
reuse v1 frame directories for v2 rows (v1 frames sit on a 2 fps grid that can't express every
v2 gap); new code reads paths from `data/common/v2/` rather than hardcoding them.

To get v1 exactly as it was: `git checkout v1 && git submodule update`.
