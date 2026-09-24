# Versions

The repo holds two generations of the video self-supervised training pipeline, in parallel
folders: **`data/v1/`** and **`data/v2/`**, with matching launcher folders in the internvl-u
submodule (`shell/internvlu/{sft,orchestration}/{v1,v2}/`). Eval suites are shared, because
they exist to compare v1 and v2 checkpoints side by side.

| | **v1** — S0-S4 | **v2** — T0-T4 (temporal SSL) |
|---|---|---|
| Doc | [`v1.md`](v1.md) | [`v2.md`](v2.md) |
| Task family | 3 context frames → 1 generated frame; settings vary caption / target direction / MCQ / progression text | Caption-conditioned 4-frame window: 3 frames in, 1 generated, plus optional GAP / ORDER / MISSING text prediction |
| Frame sampling | Fixed 2 fps grid (0.5 s; NWM native 4 fps), target fixed 0.5 s ahead | F0-F3 equally spaced, gap Δt sampled per window from a discrete set (TBD) |
| Context order | Always ordered | Ordered (T0-T2) or shuffled (T3, T4-B) |
| Caption | Per-source (VBVR scene description, EPIC narration, NWM pose text, Panda captions); none in S0/S1 | Captions shipped with the raw data |
| Data sources | VBVR (5 tasks), EPIC-Kitchens, NWM/RECON, Panda-70M-EPIC, Panda-70M v1 | Overlapping real-video sources re-sampled + new sources (to be downloaded) |
| Frames on disk | `/scratch/network/ssd/junlin/{vbvr_next_frame,epic_ssl_frames,nwm_data,panda70m_*_frames}/` | `/scratch/network/ssd/junlin/v2_frames/<source>/` |
| Stage-1 checkpoints | `/scratch/network/ssd2/junlin/models/internvlu-s{0..4}-gen(-merged)` | `/scratch/network/ssd/junlin/models/internvlu-v2-t{0..4}-gen(-merged)` |

## Path map

| Layer | v1 | v2 |
|---|---|---|
| Per-source processing | `data/v1/sources/<name>/` | `data/v2/sources/<name>/` (reused and new sources alike) |
| Cross-source code | `data/v1/common/` | `data/v2/common/` |
| Recipes | `data/v1/recipes/{s0s3_baseline,s4_progression,uniform_data_split_regarding_data_source}/` | `data/v2/recipes/temporal_ssl/` |
| Final datasets | `data/v1/datasets/{final_s0s3,s4_progression,vbvr_target_pred_4task,worldprediction,uniform_*,*_Composed_intermediate*}/` | `data/v2/datasets/temporal_ssl/` |
| Meta | `data/v1/meta/*.json` | `data/v2/meta/*.json` |
| Training launchers | `training/models/internvl-u/internvl_chat/shell/internvlu/sft/v1/` | `.../sft/v2/` |
| Multi-run drivers | `.../shell/internvlu/orchestration/v1/` | `.../orchestration/v2/` |
| Eval | `eval/suites/*` (shared) | `eval/suites/v2_temporal_ssl/` + v2 rows added to the shared suites |

Not versioned: `training/models/internvl-u/.../shell/internvlu/{engine,dualssl,legacy}/`
(shared or pre-v1) and the training code itself.

## Rules

- v2 work doesn't edit `data/v1/` or `sft/v1/`. Fix a v1 bug in its own commit and say so.
- v2 rows never point at v1 frame directories: v1 frames sit on a 2 fps grid that can't
  express every v2 gap.
- v2 code reads its roots from `data/v2/common/` (planned `paths.py`) rather than hardcoding them.

## History

- Tag **`v1`** (repo `a419e12`, internvl-u submodule `8685c94`): v1 content from before the
  split, when it sat directly under `data/` and `sft/` / `orchestration/`.
- The split moved v1 into `data/v1/` and `{sft,orchestration}/v1/` and rewrote every
  repo-internal absolute path (meta annotations, ~73k image paths in jsonl, launchers,
  builders, eval scripts) to match. Historical run outputs (`eval/**/results/`, training
  logs) were left untouched and still show the old paths, which is what those runs actually used.
- Scripts that `cd` into the original `ssl_mllm/Model_Related/InternVLU/InternVL` checkout
  (S0-S4 launchers' `REPO_ROOT`, `orchestrate_s0s3_torrnode8.sh`, the composed-intermediate
  drivers) still call `shell/internvlu/sft/<script>.sh` *inside that checkout*, which was not
  restructured.
