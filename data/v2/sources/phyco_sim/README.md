# `data/v2/sources/phyco_sim/` — PhyCo-Sim (v2 source)

> **Not used in the v2 window pool** (dropped by the user on 2026-09-26). Manifest and loader
> (`recipes/temporal_ssl/clip_sources.py`) are kept in case it is added back.

- **Repo:** [`nnsriram97/phyco_kubric`](https://huggingface.co/datasets/nnsriram97/phyco_kubric)
  (CC-BY-ND-4.0). The repo is **gated**: the HF account whose token is used (`foolen`) must click
  "request access" on the repo page first (auto-approved). Until then downloads return 403.
- **What it is:** PyBullet-simulated, Kubric/Blender-rendered physics clips from the PhyCo paper
  (arXiv 2604.28169): 4 s at 24 fps, 432×768, with friction / restitution / deformation /
  force varied systematically. Each sample ships RGB (`rgba.mp4`), depth, segmentation, and
  physical-parameter metadata.
- **Size / layout:** 29.0 GB across 9 scene folders — `ball_drop_soft_v4` (9.6 GB),
  `friction_slide_flat_v2` (4.4), `jenga_force` (3.2), `ball_wall_collision` (2.5),
  `cube_deform_soft_v2_noeff` (2.3), `friction_slide_flat_force_v3` (2.2), `ball_drop_v3` (1.9),
  `pool_table_force` (1.5), `ball_drop_v2` (1.3). Each folder holds per-date `*.tar.gz` shards,
  `common_caption_cosmos.txt` (one scene-level caption), `props_of_interest.json` (which
  physical parameters vary), and `data_stats_json.tar.gz`.
- **Raw location:** `/scratch/network/ssd/junlin/raw/phyco_sim/`.

## Filtering & sampling summary

**Stratified by scene + physical-parameter bins.**

1. Pool: full dataset downloaded (127,440 samples); `*_test` shards dropped.
2. Motion filter: object must move ≥ 18 frames (`settle_frame`; the renderer freezes the last frame after objects settle).
3. Sampling: equal quota per scene (9 scenes, ~1,111 each); within a scene, equal share per bin of its varying parameter (5 quantile bins of restitution / friction / force, or categories), seed 42.
4. No caption filter (captions are scene-level templates), no camera QC (synthetic, fixed camera).
5. Final: `selection_10k.jsonl` = 10,000 rgba.mp4 (0.8 GB); **still to be trimmed to 5K**.

Details below.

## Status (2026-09-24)

1. **Downloaded + extracted** (`download_phyco_sim.py --extract`): 29 GB of tar.gz → 127,440
   samples, 69 GB on disk including the tars. Each sample dir:
   `rgba.mp4` (98 frames @ 24 fps = 4.08 s, 768×432), `depth.mp4`, `segmentation.mp4`,
   `metadata.json`, `animation_data.pkl`.
2. **10K selected** (`select_phyco_sim.py`, seed 42) → `selection_10k.jsonl` in the raw dir.

## Selection rules and result

- **Motion filter.** The renderer stops simulating once objects settle
  (`metadata.rendering_efficiency.settle_frame`) and repeats the last frame for the rest of the
  clip, so real motion is often far shorter than 4 s. Keep samples moving ≥ 18 frames (one
  Δt≈0.25 s window). `settle_frame = None` means the object moves for the whole clip.
- Drop `*_test` shards. Equal quota per scene (1,111–1,112). Within a scene, stratify by its
  physical parameter (5 quantile bins, or categories) with an equal share per bin.

| Scene | Samples | Eligible (motion ≥ 18 f) | Stratified by |
|---|---|---|---|
| ball_drop_soft_v4 | 10,197 | **1,191** | none (no varying scalar found) |
| ball_drop_v2 | 10,671 | 4,884 | `ball_restitution` (5 bins) |
| ball_drop_v3 | 10,051 | 7,993 | `num_balls` (2/3) |
| ball_wall_collision | 13,099 | 13,099 | `ball_restitution` (5 bins) |
| cube_deform_soft_v2_noeff | 7,914 | 7,914 | `jelly_texture` (3) |
| friction_slide_flat_force_v3 | 15,022 | 14,997 | `force_magnitude` (5 bins) |
| friction_slide_flat_v2 | 30,501 | 14,816 | `platform_friction` (5 bins) |
| jenga_force | 14,981 | 14,701 | push direction (x/y) |
| pool_table_force | 14,999 | 13,624 | `force_magnitude` (5 bins) |

Motion length of the 10K: p10 / p50 / p90 = 21 / 51 / 98 frames (0.9 / 2.1 / 4.1 s).

## Caveats for building v2 windows

- **Δt must fit inside the motion span, not the clip length.** Use `motion_frames` from the
  manifest when sampling windows (3Δt ≤ motion span), otherwise late windows are static.
- **Caption:** only a scene-level `common_caption_cosmos.txt` exists (identical for every clip
  in a scene, and it describes the outcome, e.g. "…bounces up after impact"). `ball_drop_v3`
  has none. A per-sample template filled from the metadata parameters is still an option.
- `ball_drop_soft_v4` is nearly exhausted by the motion filter (1,191 eligible for 1,112 picked).
