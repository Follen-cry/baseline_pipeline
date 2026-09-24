# multi_object_placement_Composed_intermediate_cross

Variant of `multi_object_placement_Composed_intermediate` (same composed 2x2-grid task:
partial grid with first/final frame in → grid with the two intermediate frames filled in),
with two changes:

- a dashed "+" divider is drawn across the grid in both input and target, and the prompt
  tells the model about it;
- **3,000 train / 20 eval**, freshly generated with `SEED = SPLIT_SEED = 20260918`
  (`param_hash`-disjoint from the original 600-sample set).

Rendering reuses the fidelity-checked helpers of the original builder. Build script:
`data/v1/sources/vbvr/processing/build_multi_object_placement_composed_intermediate_cross.py`
(raw output → `data/v1/sources/vbvr/raw/multi_object_placement_cross/`, gitignored).
Meta: `data/v1/meta/multi_object_placement_Composed_intermediate_cross{,_train}_meta.json`.
Training driver: `training/models/internvl-u/internvl_chat/shell/internvlu/orchestration/v1/run_mop_cross_sft.sh`.

`images/` is gitignored — rerun the build script to regenerate it.
