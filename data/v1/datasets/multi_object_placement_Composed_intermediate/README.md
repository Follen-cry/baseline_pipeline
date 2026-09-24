# multi_object_placement_Composed_intermediate

500 train / 100 eval, adopting the same composed-single-image-input pipeline
as `2d_geo_trans_Composed_intermediate` (see that dataset's README.md and
`data/v1/sources/vbvr/processing/build_2d_geo_trans_composed_intermediate.py`).

- **Input (1 image):** a partial 2x2 grid — top-left = first frame (N colored
  shapes + color-matched star markers at their starting positions),
  bottom-right = final frame (every shape moved onto its marker),
  top-right/bottom-left = blank.
- **Target:** the same canvas with the two blank cells filled in with the
  **real** rendered intermediate frames (shapes 1/3 and 2/3 of the way to
  their markers).

## Task mechanics

N shapes (2–5, random per sample) each move toward a same-colored
four-pointed-star marker. All N objects move **simultaneously** under one
shared progress fraction — confirmed by reading the generator's own
`_generate_animation_frames`: each object's position is linearly
interpolated `start + (target - start) * progress`, then clamped to stay
`size+10` px from the canvas edge.

## Provenance

600 fresh samples generated with the vendored generator at
`data/v1/sources/vbvr/generators/multi_object_placement/`, `seed=20260916` —
disjoint by construction, and checked here (`param_hash` comparison), from
the original 10-sample eval_samples probe's `seed=777`. The 2 intermediate
frames per sample are rendered by re-running the interpolation+clamp formula
at progress 1/3 and 2/3, using the generator's own `_draw_marker`/
`_draw_object_with_gradient_border` draw calls. Fidelity checked **both
ways** (unlike the rotation task's one-sided check): progress=0 must equal
the real `first_frame.png` AND progress=1 must equal the real
`final_frame.png` — all 600 passed (see
`data/v1/sources/vbvr/processing/build_multi_object_placement_composed_intermediate.py`).

600 samples were shuffled (`seed=20260916`) and split 500/100.

## Layout

```
data/v1/sources/vbvr/raw/multi_object_placement/   # raw generator output, untouched
  multi_object_placement_task/
    multi_object_placement_{00000000..00000599}/
      first_frame.png, final_frame.png, prompt.txt, metadata.json

data/v1/datasets/multi_object_placement_Composed_intermediate/
  train.jsonl (500 rows), eval.jsonl (100 rows)
  images/{train,eval}/<id>/
    partial_grid_input.png   # the 1 input image
    gt_grid.png               # the target image, REAL ground truth
    intermediate_1.png, intermediate_2.png   # debug/inspection only

data/v1/meta/multi_object_placement_Composed_intermediate_meta.json
data/v1/meta/multi_object_placement_Composed_intermediate_train_meta.json  # train-only (see note below)
```

**Note:** the training entrypoint (`internvl_chat_finetune_u_full.py`) mixes
*every* key present in a `META_PATH` file into one concatenated training
set, so a train-only meta file is kept separate from the combined
train+eval meta to avoid leaking eval rows into training.
