# vae_cond: how many input frames should the VAE see? (InternVL-U base, PhysicTran38K)

A small qualitative experiment on InternVL-U's hybrid ViT + VAE conditioning. Given 3 consecutive frames F0, F1, F2 of a
PhysicTran38K clip (gap 1.0 s) and its caption, the **base** model generates the next frame F3. Only the **VAE
condition** changes; the ViT always sees all three frames.

| condition | ViT sees | VAE latents fed to the decoder |
|---|---|---|
| `none`  | F0 F1 F2 | none (`conditional_input=None`; decoder conditions on VLM hidden states only) |
| `first` | F0 F1 F2 | F0 |
| `all`   | F0 F1 F2 | F0 F1 F2 (stock pipeline behaviour) |

5 datapoints = 15 generations, same noise seed (42) for every condition, so differences come from conditioning alone.
No training and no checkpoints other than the base model.

## Run

```
bash run.sh [gpu_index]        # env: internvlu; ~15 generations on one A40
```
`run.sh` does `check_inputs.py` (CPU wiring check) -> `gen.py` (resumable) -> `make_grid.py`.
Output: `outputs/{none,first,all}/<id>.png`, `outputs/comparison.png` (rows = datapoints; columns F0 | F1 | F2 | GT F3 |
none | first | all), `outputs/rows/`, `outputs/pixel_diff.csv`, `outputs/_meta/*.json`.

## Files

| file | role |
|---|---|
| `config.json` | the only place paths/settings live (base model, package, source jsonl, inference settings) |
| `select_datapoints.py` -> `datapoints.jsonl` | one-off: 5 rows from T5 evalmini (`phystran_g1`, 125 rows), one per distinct category, seed 20260930. The jsonl is committed because the T5 jsonl is gitignored |
| `gen.py` | generation for the 3 conditions |
| `patches.py` | runtime patch for one stock-processor assert (below) |
| `check_inputs.py` | CPU-only check that the ViT/VAE counts per condition are what the table says |
| `make_grid.py` | comparison sheet + a crude pixel-difference table |
| `run.sh` | everything above, in order |

## How the three conditions are implemented

The prompt is T5's exact user turn (caption, `F0/F1/F2: <image>` slots, `GAP`), and the pipeline gets `image=[[F0, F1, F2]]`
(one entry per prompt = its 3 frames), so every frame gets an `<image>` slot and goes through the ViT.

In the stock `InternVLUPipeline._prepare_diffusion_inputs` every image before the target is also VAE-encoded
(`conditional_image` <- `pixel_values_gen`). `gen.py` wraps that method and slices the **VAE-only** tensors
`conditional_image` and `image_grid_thw_gen_cond` to the first 0 / 1 / 3 frames (kept index-aligned, since the decoder
concatenates both). The ViT-side tensors (`pixel_values`, `encoder_image_token_mask`, `image_fhw_cond`) are never touched.
`check_inputs.py` confirms on real inputs: ViT gets 3 frames (768 `<IMG_CONTEXT>` tokens) in the conditioned CFG rows in
all three conditions; VAE images per row are `[0,0,0]` / `[1,1,0]` / `[3,3,0]` (rows = full-cond, text-dropped,
fully-dropped; the fully-dropped unconditional row never has images).

Notes:
- **Stock processor assert.** With 3 frames, `InternVLUProcessor._insert_media_placeholders` asserts ("only 1 fake
  picture in pure text input") on the unconditional CFG row, whose prompt has no `<image>` placeholders. That row's image
  data is discarded anyway, so `patches.py` skips the assert for it. The package under `Model_Related/` is not edited.
- **Resolution.** Frames are resized with the same area-512 / multiple-of-16 rule as `eval/v2_suite_1_sub/scripts/gen.py`
  (832x480 -> 672x384), and the pipeline's own gen-processor may round the VAE condition slightly differently, as it does
  for every stock edit call.
- **Frame order.** `first` means F0, the earliest frame (the same F0 that T5 always used as its VAE hint).
- **Same-image ambiguity.** `none` vs `first` vs `all` also changes the decoder's token count (more condition tokens), not
  only the information in it; this experiment does not separate those.
