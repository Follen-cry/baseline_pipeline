# rotation_puzzle_Composed_intermediate

500 train / 100 eval, adopting the same composed-single-image-input pipeline
as `2d_geo_trans_Composed_intermediate` (see that dataset's README.md and
`data/sources/vbvr/processing/build_2d_geo_trans_composed_intermediate.py`).

- **Input (1 image):** a partial 2x2 grid — top-left = first frame (a fixed
  2x2 grid of 4 squares, each with an L-shaped pipe pattern, at randomized
  starting rotations), bottom-right = final frame (all 4 squares solved at
  0°, pipes forming a continuous loop), top-right/bottom-left = blank.
- **Target:** the same canvas with the two blank cells filled in with the
  **real** rendered intermediate frames (squares 1/3 and 2/3 of the way
  rotated toward solved).

## Task mechanics

All 4 squares rotate **simultaneously** under one shared progress fraction —
confirmed by reading the generator's own ground-truth video
(`_create_rotation_animation_frames`). `metadata.json` already stores each
square's `initial_angle`, `target_angle` (always 0), and `rotation_angle`
(the exact shortest-path signed diff), so no wrap-around recomputation is
needed.

## Provenance

600 fresh samples generated with the vendored generator at
`data/sources/vbvr/generators/rotation_puzzle/`, `seed=20260916` — disjoint
by construction, and checked here (`param_hash` comparison), from the
original 10-sample eval_samples probe's `seed=777`. The 2 intermediate
frames per sample are rendered by reconstructing a `puzzle_data` dict from
metadata and calling the generator's own
`_render_frame(step_data, use_initial_angles=True)` at progress 1/3 and 2/3
— **linear** interpolation (not the ground-truth video's ease-in-out), for
consistency with every other task in this pipeline. Fidelity checked both
ways: progress=0 against `first_frame.png`, progress=1 against
`final_frame.png` — all 600 passed.

600 samples were shuffled (`seed=20260916`) and split 500/100.

## Layout

```
data/sources/vbvr/raw/rotation_puzzle/   # raw generator output, untouched
  rotation_puzzle_task/
    rotation_puzzle_{00000000..00000599}/
      first_frame.png, final_frame.png, prompt.txt, metadata.json

data/datasets/rotation_puzzle_Composed_intermediate/
  train.jsonl (500 rows), eval.jsonl (100 rows)
  images/{train,eval}/<id>/
    partial_grid_input.png   # the 1 input image
    gt_grid.png               # the target image, REAL ground truth
    intermediate_1.png, intermediate_2.png   # debug/inspection only

data/meta/rotation_puzzle_Composed_intermediate_meta.json
data/meta/rotation_puzzle_Composed_intermediate_train_meta.json  # train-only (see note below)
```

**Note:** the training entrypoint (`internvl_chat_finetune_u_full.py`) mixes
*every* key present in a `META_PATH` file into one concatenated training
set, so a train-only meta file is kept separate from the combined
train+eval meta to avoid leaking eval rows into training.
