# `data/sources/vbvr/`

The VBVR synthetic-task pool used by the S0-S3 baseline. Full provenance:
`../../../PROVENANCE.md` §1-§2.

## `generators/`

Vendored snapshots (no git history preserved, per project decision) of the 5 VBVR-DataFactory
generators that actually feed S0-S3 — the `"pilot5"` preset in `processing/vbvr_task_presets.py`.
Renamed from their original `VBVR-DataFactory/<O|G>-<n>_<name>_data-generator` naming to plain
task names, matching how they're referred to everywhere else (jsonl `task_name` field, meta
files, eval scorers):

| Task name (used everywhere else) | Original generator dir |
|---|---|
| `multi_object_placement` | `G-5_multi_object_placement_data-generator` |
| `shape_color_then_move` | `O-11_shape_color_then_move_data-generator` |
| `shape_outline_then_move` | `O-13_shape_outline_then_move_data-generator` |
| `rotation_puzzle` | `O-44_rotation_puzzle_data-generator` |
| `2d_geometric_transformation` | `O-6_2d_geometric_transformation_data-generator` |

The other 11 generators in the original `VBVR-DataFactory/` are **not** part of S0-S3 (they
feed an unrelated 140k-row dataset) and were not migrated here.

## `processing/`

Scripts that turn generator output into training rows:
`vbvr_next_frame_generate.py` (drives generators → raw video) →
`vbvr_next_frame_sample.py` (window sampling) →
`vbvr_next_frame_prompt_adapt.py` (prompt text) →
`vbvr_next_frame_make_dual.py` (S3's MCQ construction + the actual train/eval window split).
`vbvr_task_presets.py` defines the `pilot5` task list these scripts read.

## `raw/` — not migrated yet

Raw generator output (videos + sampled frame windows) lives outside this repo, on a different
volume than the rest of this pipeline:
`/scratch/network/ssd/junlin/vbvr_next_frame/{raw,samples}/<task>/...` (11,000 raw videos per
task). This is deliberately **out of scope for this migration pass** — it's large, and some
sibling real-world sources' raw data is node-pinned (only reachable from `torrnode11`). It'll be
restructured and copied in as a separate pass; see the repo-plan memory / PROVENANCE.md for the
migration-rule context. Until then, re-running `processing/vbvr_next_frame_generate.py` against
the original `VBVR-DataFactory` paths (in the `ssl_mllm` working tree) is the only way to
regenerate raw data from scratch.
