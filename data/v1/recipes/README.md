# `data/v1/recipes/`

A **recipe** = a specific way of combining already-split per-source data (from
`data/v1/sources/*/`) into one final, mixed, training-ready dataset (landing in
`data/v1/datasets/<recipe-name>/`). New training data mixes get a new recipe dir here, without
touching source code.

Note on naming: the originally-sketched plan called this tier `data/splits/` ("generic
train/eval separation logic"). Migration revealed that's not where reality lives — every one of
the 5 sources decides its own train/eval split internally, inside its own
`data/v1/sources/<name>/processing/` code (video/trajectory/clip-level, always seeded
`SPLIT_SEED=42` — see each source's README). What's actually centralized and cross-source is
the **assembly** of pre-split source outputs into one mix — hence `recipes/`, not `splits/`.

## `s0s3_baseline/`

The S0-S3 baseline recipe. `merge_final_s0s3.py` combines the VBVR pilot5 pool with 4
real-world sources per S0/S1/S2/S3 setting (full table: `../../PROVENANCE.md` §1);
`filter_final_s0s3.py` runs after merging to drop rows referencing missing image files
(~2.7% loss, confirmed to have run exactly once — see PROVENANCE.md §6.4). Output lands in
`../datasets/final_s0s3/`.

## `s4_progression/`

S3 with the MCQ dropped, replaced by a progression-description + next-frame text
prediction trained jointly with the existing image-generation prediction. Built from S2's
full window set (not S3's MCQ-filtered subset — see this recipe's own README for why those
differ). Full design/decision log, prompt templates, and verification:
`s4_progression/README.md`.
