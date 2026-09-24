# `data/v1/datasets/`

Final, merged, training-ready jsonl — the output of a recipe in `../recipes/`. These are what
each training script's `META_PATH` actually points at.

- **`final_s0s3/`** — output of the `s0s3_baseline` recipe. `S{0,1,2,3}_train.jsonl` (49,113 /
  49,113 / 49,113 / 47,770 rows) + `S3_eval.jsonl` (2,075 rows). Row counts already reflect
  `filter_final_s0s3.py`'s post-merge cleanup (~2.7% dropped for missing images). Each row's
  `image`/`target_image`/`cond_image` paths point at
  `/scratch/network/ssd/junlin/vbvr_next_frame/samples/...` and similar per-source frame roots
  — **not yet migrated into this repo** (see each source's README under `../sources/`), so these
  jsonl files alone are not sufficient to actually run training until that raw-data pass
  happens.
- **`vbvr_target_pred_4task/`** — the stage-2 continuation-SFT dataset
  (`target_pred_4task_train_no_ce.jsonl`, 2,000 rows = 4 tasks × 500). Feeds
  `run_target_pred_mop_sft.sh` per `../../PROVENANCE.md` §5.
- **`<task>_Composed_intermediate/`** (`2d_geo_trans`, `multi_object_placement`,
  `rotation_puzzle`, `shape_color_then_move`; 500 train / 100 eval each) and
  **`multi_object_placement_Composed_intermediate_cross/`** (3,000 / 20) — VBVR composed
  2x2-grid stage-2 SFT sets, built by `../sources/vbvr/processing/build_*_composed_intermediate*.py`.
  Only jsonl + README are committed; `images/` is gitignored and regenerable.

`../meta/*.json` files' `annotation` field points at these files by absolute path — repointed
during migration from the old `ssl_mllm/data/datasets/...` location to this repo's location.
If this repo is ever moved, those paths need updating again.
