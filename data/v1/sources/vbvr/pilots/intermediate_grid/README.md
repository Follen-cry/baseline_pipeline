# VBVR `2d_geometric_transformation` — intermediate-grid task

Reformats the existing VBVR `2d_geometric_transformation` (planar rotation)
eval task from its original **first frame → final frame** single-image
format into a **first + final frame → 2×2 grid of 4 frames** format.

- **Input:** `first_frame` + `final_frame` (the rotation's start and end states)
- **Target:** the model must infer the intermediate rotation steps and
  output one composite image — a 2×2 grid:
  - top-left: first frame
  - top-right: intermediate frame 1
  - bottom-left: intermediate frame 2
  - bottom-right: final frame
  - reading top-left → top-right → bottom-left → bottom-right in temporal order

## Provenance

The 10 datapoints are the same 10 samples already used for the original
single-final-frame task, at:
```
Evaluation/VBVR-CustomEval/eval_samples/raw/2d_geometric_transformation/
  transformation_worlds_2d_geometric_transformation_task/
    transformation_worlds_2d_geometric_transformation_0000000{0..9}/
```
Each source sample ships `first_frame.png`, `final_frame.png`, `prompt.txt`,
and `metadata.json`. The rotation is fully deterministic (shape, rotation
center, initial angle, rotation angle, and color are all recorded in
`metadata.json`), so the two intermediate frames were **re-rendered**, not
hand-drawn, by reusing the original generator's private rendering methods
(`_create_base_shape`, `_rotate_and_translate_shape`, `_draw_dashed_polygon`,
`_draw_rotation_center`) from:
```
VBVR-DataFactory/O-6_2d_geometric_transformation_data-generator
```
at intermediate progress fractions 1/3 and 2/3 along the
`initial_angle -> initial_angle + rotation_angle` sweep.

`scripts/build_intermediate_grid_samples.py` does this end to end and,
before trusting any re-rendered frame, asserts that re-deriving the FIRST
frame from `metadata.json` byte-matches the original `first_frame.png` for
every sample (this passed for all 10 — see the script's stdout / each
sample's `metadata.json` `"fidelity_check"` field). Re-run it if this task
ever needs to be rebuilt or extended to more samples.

## Layout

```
data/v1/sources/vbvr/pilots/intermediate_grid/
├── README.md
├── manifest.json                  # index of all 10 samples
├── scripts/
│   └── build_intermediate_grid_samples.py
└── samples/
    └── 00 .. 09/
        ├── first_frame.png        # copied from the original raw sample
        ├── intermediate_1.png     # re-rendered, progress = 1/3
        ├── intermediate_2.png     # re-rendered, progress = 2/3
        ├── final_frame.png        # copied from the original raw sample
        ├── gt_grid.png            # GT 2x2 composite (1024x1024, 512px cells)
        ├── prompt_original.txt    # original single-final-frame task prompt
        ├── prompt_grid.txt        # modified prompt: asks for the 2x2 grid
        └── metadata.json          # rotation params + grid layout + provenance
```

Each `prompt_grid.txt` wraps the sample's original task-rule text with a
fixed instruction (see `GRID_PROMPT_WRAPPER` in the build script) that
explicitly asks the model to infer the intermediate steps and output the
complete 2×2 grid in the layout above.
