# 2d_geo_trans_Composed_intermediate

500 train / 100 eval, scaled up from the 10-sample "composed input" probe
(`data/v1/sources/vbvr/pilots/intermediate_grid/scripts/build_partial_grid_input.py` +
`Evaluation/vbvr_baseline_runners/run_intermediate_grid_infer_composed.py`).

- **Input (1 image):** a partial 2x2 grid — top-left = first frame, bottom-right
  = final frame, top-right/bottom-left = blank.
- **Target:** the same canvas with the two blank cells filled in with the
  **real** rendered intermediate frames (not a placeholder) — a genuine
  same-canvas image edit, so VAE pixel-level conditioning on the single input
  image is appropriate here (unlike the 2-separate-input-images format).

## Provenance

600 fresh `2d_geometric_transformation` samples were generated with the
vendored generator at `data/v1/sources/vbvr/generators/2d_geometric_transformation/`,
`seed=20260916` — disjoint by construction, and defensively checked here
(`param_hash` comparison), from the original 10-sample probe's `seed=777`.
For each sample, the two intermediate frames were rendered deterministically
from its own `metadata.json` the same fidelity-checked way as
`data/v1/sources/vbvr/pilots/intermediate_grid/scripts/build_intermediate_grid_samples.py`: every
one of the 600 re-derived first frames was asserted byte-identical to the
generator's own `first_frame.png` before its intermediate frames were
trusted (all 600 passed — see
`data/v1/sources/vbvr/processing/build_2d_geo_trans_composed_intermediate.py`).

600 samples were shuffled (`seed=20260916`) and split 500/100.

## Layout

```
data/v1/sources/vbvr/raw/2d_geometric_transformation/   # raw generator output, untouched
  transformation_worlds_2d_geometric_transformation_task/
    transformation_worlds_2d_geometric_transformation_{00000000..00000599}/
      first_frame.png, final_frame.png, prompt.txt, metadata.json

data/v1/datasets/2d_geo_trans_Composed_intermediate/
  train.jsonl (500 rows), eval.jsonl (100 rows)
  images/{train,eval}/<id>/
    partial_grid_input.png   # the 1 input image (referenced by "image" in the jsonl)
    gt_grid.png               # the target image, REAL ground truth (referenced by "target_image")
    intermediate_1.png, intermediate_2.png   # debug/inspection only, not referenced by the jsonl

data/v1/meta/2d_geo_trans_Composed_intermediate_meta.json
```

Each jsonl row's prompt is the same fixed wrapper used by the probe,
instantiated with that sample's own rotation rule as the trailing `Task:`
line — see `build_2d_geo_trans_composed_intermediate.py`'s
`PARTIAL_GRID_PROMPT_WRAPPER`.
