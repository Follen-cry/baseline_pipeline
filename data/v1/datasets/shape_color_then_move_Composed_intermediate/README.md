# shape_color_then_move_Composed_intermediate

500 train / 100 eval, adopting the same composed-single-image-input pipeline
as `2d_geo_trans_Composed_intermediate` (see that dataset's README.md and
`data/v1/sources/vbvr/processing/build_2d_geo_trans_composed_intermediate.py`).

- **Input (1 image):** a partial 2x2 grid — top-left = first frame (an
  A→B→C :: D→?→? visual analogy: a static solved example row A/B/C, plus a
  question row with D shown and two "?" marks), bottom-right = final frame
  (both "?" revealed: E = D recolored, F = D recolored **and** moved),
  top-right/bottom-left = blank.
- **Target:** the same canvas with the two blank cells filled in with the
  **real** rendered intermediate frames.

## Task mechanics

This is a genuinely **two-phase sequential** transformation, not one
continuous sweep — confirmed by reading the generator's own ground-truth
video code (`_create_sequential_morph_frames`): phase 1 lerps E's color
`color_from → color_to` while F stays hidden; phase 2 holds E's color fixed
and lerps F's y-position `center → move_offset` (a fixed named pixel offset,
e.g. `down_large` = +80px). The top scaffold row (A, B, C) is static and
identical in every cell — it carries no signal about the transformation and
is copied unchanged.

Chosen intermediates (reusing the generator's own draw calls exactly, no
custom interpolation math for the color step since it's a discrete state):
- **intermediate_1** = phase-1 END: E fully recolored to `color_to`, F still `?`.
- **intermediate_2** = phase-2 MIDPOINT (progress=0.5): E unchanged, F drawn
  at `color_to` with y lerped 50% toward `move_offset` (same clamped linear
  formula as the generator's own `_create_sequential_morph_frames`).

## Provenance

600 fresh samples generated with the vendored generator at
`data/v1/sources/vbvr/generators/shape_color_then_move/`, `seed=20260916` —
disjoint by construction, and checked here (`param_hash` comparison), from
the original 10-sample eval_samples probe's `seed=777`. This generator's
`TaskGenerator.__init__` **requires** `generate_videos=True` (raises
otherwise) and writes a real ground-truth video per sample to a temp dir —
unavoidable overhead, discarded after each build run. Fidelity checked:
each sample's re-derived first_frame (D + two "?", built from metadata) is
asserted byte-identical to the real `first_frame.png` before its
intermediates are trusted — all 600 passed.

600 samples were shuffled (`seed=20260916`) and split 500/100.

## Layout

```
data/v1/sources/vbvr/raw/shape_color_then_move/   # raw generator output, untouched (includes real GT videos)
  shape_color_then_move_task/
    shape_color_then_move_{00000000..00000599}/
      first_frame.png, final_frame.png, prompt.txt, metadata.json, ground_truth.mp4

data/v1/datasets/shape_color_then_move_Composed_intermediate/
  train.jsonl (500 rows), eval.jsonl (100 rows)
  images/{train,eval}/<id>/
    partial_grid_input.png   # the 1 input image
    gt_grid.png               # the target image, REAL ground truth
    intermediate_1.png, intermediate_2.png   # debug/inspection only

data/v1/meta/shape_color_then_move_Composed_intermediate_meta.json
data/v1/meta/shape_color_then_move_Composed_intermediate_train_meta.json  # train-only (see note below)
```

**Note:** the training entrypoint (`internvl_chat_finetune_u_full.py`) mixes
*every* key present in a `META_PATH` file into one concatenated training
set, so a train-only meta file is kept separate from the combined
train+eval meta to avoid leaking eval rows into training.
