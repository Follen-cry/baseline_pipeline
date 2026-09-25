# `data/v2/sources/vbvr/` — VBVR synthetic task videos (v2 source)

- **Raw videos (not copied):** `/scratch/network/ssd/junlin/vbvr_next_frame/raw/<task>/<chunk>/<prefix>_task/<task_id>/`
  with `ground_truth.mp4`, `first_frame.png`, `final_frame.png`, `metadata.json`, `prompt.txt`.
  This is the pool v1's S0-S3 next-frame data was built from: 14 tasks, 138,000 samples,
  40.6 GB of mp4 (42.9 GB incl. frames/metadata). Sample names restart in every chunk, so a
  sample is identified by its path (`id = vbvr__<task>__<chunk>__<task_id>`).
- `/scratch/network/ssd/junlin/vbvr_target_pred/` (the target_pred train/eval pool) has
  first/final frames only, **no video**, so it can't feed v2 windows.
- **Manifest:** `build_vbvr_manifest.py` → `/scratch/network/ssd/junlin/raw/vbvr/{manifest_all.jsonl, summary.json}`.

## Filtering & sampling summary

**No sampling — whole pool minus eval-reserved and duplicates.**

1. Pool: all 14 tasks of `vbvr_next_frame/raw` (138,000 generator videos).
2. Excluded as `eval_reserved` (1,000): source videos of v1 S3 eval windows (998) + samples whose first frame is byte-identical to a target_pred eval input (2).
3. Excluded as `duplicate` (2,905): byte-identical `ground_truth.mp4` (mostly mirror_reflection).
4. No caption / CV / VLM filter (synthetic, clean).
5. Final: `split == "train"` in `manifest_all.jsonl` = 134,095 (39.8 GB). **A per-task stratified subsample is still needed** if v2 uses ~5K per source.

Details below.

## Train vs eval (so the existing VBVR evals stay reusable)

| split | rule | count |
|---|---|---|
| `eval_reserved` | source video of a v1 **S3 eval** window (`final_s0s3/S3_eval.jsonl` → `window_meta.json:source_video`; 200 per task for 5 tasks, 998 unique) **or** `first_frame.png` byte-identical to a **target_pred eval** input (ID + sotm + OOD, 1,000 unique images; 2 grid_shift hits) | 1,000 |
| `duplicate` | `ground_truth.mp4` byte-identical to an earlier sample (almost all mirror_reflection: 2,858) | 2,905 |
| `train` | everything else | **134,095 (39.8 GB mp4)** |

The target_pred eval (ID 5×100 + sotm 100, OOD 4×100) was generated separately from this pool;
only the 2 grid_shift samples overlap by first frame, and those are reserved. A shared first
frame alone is not treated as a duplicate: most mirror_reflection samples share a start state
but have different videos.

## Video specs (1024×1024, synthetic)

| task | fps | frames | duration |
|---|---|---|---|
| grid_shift, grid_shortest_path, mirror_reflection | 16 | 35 | 2.2 s |
| animal_size_sorting | 16 | 40 | 2.5 s |
| multi_object_placement | 16 | 48 | 3.0 s |
| shape_color_then_move / _then_scale | 16 | 60 | 3.8 s |
| shape_outline_then_move | 16 | 64 | 4.0 s |
| 2d_geometric_transformation | 15 | 70 | 4.7 s |
| ball_bounces_given_time | 16 | 80 | 5.0 s |
| maze | 16 | 93 | 5.8 s |
| stable_sort | 16 | 96 | 6.0 s |
| rotation_puzzle | 16 | 97 | 6.1 s |
| key_door_matching | 10 | 79 | 7.9 s |

(Frame counts from one sample per task; lengths vary within a task, e.g. maze with path length.)
`glass_refraction`, which is in the locked 10-task eval set, has no raw videos in this pool.
