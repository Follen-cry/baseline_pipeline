# vae_cond: how many input frames should the VAE see? (InternVL-U base, PhysicTran38K)

A qualitative + light-quantitative experiment on InternVL-U's hybrid ViT + VAE conditioning. Given 3 consecutive frames
F0, F1, F2 of a PhysicTran38K clip (gap 1.0 s) and its caption, the **base** model generates the next frame F3. Only the
**VAE condition** changes; the ViT always sees all three frames.

| condition | ViT sees | VAE latents fed to the decoder |
|---|---|---|
| `none`  | F0 F1 F2 | none (`conditional_input=None`; decoder conditions on VLM hidden states only) |
| `first` | F0 F1 F2 | F0 |
| `all`   | F0 F1 F2 | F0 F1 F2 (stock pipeline behaviour) |

**40 datapoints** (rows of the T5 eval set `phystran_g1`, one per distinct category; the first 5 are the original small
run) x 3 conditions x **3 output sizes** (832x480 native, 688x400, 672x384) = 360 generations, same noise seed (42) for all.
No training; base checkpoint only. Results page (private artifact): https://claude.ai/artifact/XyGoyjRFNgSBL3WvMue6Rd

## Headline findings (details, tables and gallery on the page)

- `none` never reproduces the input scene (0/40 at every size).
- `first` vs `all`: no semantic difference (CLIP to GT differs by <= 0.013, s.e. ~0.01); `all` is closer in pixels (it is given F2).
- **Output size must equal the VAE condition size.** 672x384 (area-512 rule) is just under the processor's `min_pixels`, so the
  condition path upscales it to 688x400 while the output stays 672x384: 10/40 layout artifacts (panels, caption strips, shifted
  content). 688x400 / 832x480 give 1-3. Native 832x480 trades them for ~5 duplicated objects.
- No setting beats "copy F2" on similarity to the ground truth (CLIP 0.974 / MAD 10.2 vs 0.83-0.89 / 25-47).

## Run

```
cd experiment/vae_cond                  # env: internvlu (python path in config.json)
python select_datapoints.py             # picks 40, copies images to data/images/, writes datapoints.jsonl (one-off, already run)
bash run_all.sh 0 1 2 3 4 5             # all sizes x conditions, one shard per GPU, resumable (~11 min on 6 A40s)
python metrics.py                       # outputs/metrics.csv (MAD + CLIP similarity to GT and F2, plus a copy-F2 reference)
python review_sheet.py                  # outputs/review_<size>_<k>.png contact sheets for eyeballing
python inspect_preproc.py tension       # what the ViT and the VAE branch receive (CPU)
python build_artifact.py                # artifact/index.html (self-contained)
```
`bash run.sh 0` is the original single-GPU 5-datapoint flow (check_inputs -> gen -> make_grid) and still works at 672x384 only
if you pass `--sizes 672x384` to `gen.py` (the default now runs all three sizes).

## Files

| file | role |
|---|---|
| `config.json` | the only place paths/settings live (base model, package, source jsonl, sizes, inference settings) |
| `select_datapoints.py` -> `datapoints.jsonl`, `data/images/` | selection + local copies of the 4 images per datapoint (committed, ~13 MB) |
| `gen.py` | generation; `--sizes`, `--shard_idx/--num_shards`, resumable. Outputs `outputs/<cond>[_<WxH>]/<id>.png` (672x384 keeps the untagged folders) |
| `patches.py` | runtime patches for two stock-package bugs (below) |
| `run_all.sh` | multi-GPU launcher |
| `check_inputs.py` | CPU wiring check: ViT/VAE image counts per condition |
| `inspect_preproc.py` | CPU: renders the ViT and VAE-branch inputs to check for cropping |
| `metrics.py`, `review_sheet.py`, `make_grid.py`, `size_sheet.py` | metrics, contact sheets (`make_grid`/`size_sheet` are the 5-datapoint originals) |
| `make_tags.py` -> `tags.json` | hand tags (coherent / layout / duplicate / object lost) per datapoint, size, condition |
| `build_artifact.py`, `artifact_template.html` | the results page |

## How the conditions are implemented

The prompt is T5's exact user turn (caption, `F0/F1/F2: <image>` slots, `GAP`), and the pipeline gets `image=[[F0, F1, F2]]`
(one entry per prompt = its 3 frames), so every frame gets an `<image>` slot and goes through the ViT.

In the stock `InternVLUPipeline._prepare_diffusion_inputs` every image before the target is also VAE-encoded
(`conditional_image` <- `pixel_values_gen`). `gen.py` wraps that method and slices the **VAE-only** tensors
`conditional_image` and `image_grid_thw_gen_cond` to the first 0 / 1 / 3 frames (kept index-aligned, since the decoder
concatenates both). The ViT-side tensors (`pixel_values`, `encoder_image_token_mask`, `image_fhw_cond`) are never touched.
Per CFG row (full / text-dropped / unconditional) the VAE image counts are `[0,0,0]` / `[1,1,0]` / `[3,3,0]`.

Two stock-package problems are patched at runtime in `patches.py` (the package under `Model_Related/` is not edited):
- **Processor assert.** With 3 frames, `InternVLUProcessor._insert_media_placeholders` asserts ("only 1 fake picture in pure
  text input") on the unconditional CFG row, whose prompt has no `<image>` placeholders. That row's image data is discarded
  anyway, so the assert is skipped for it.
- **Decoder crash for >1 VAE condition image.** `_prepare_hidden_inputs_anyres` adds a vector of per-image token counts where
  it needs their sum (`TypeError: only integer tensors of a single element can be converted to an index` at 3 images). The
  patch rewrites the function source with `.sum()` at the two slice sites (asserts exactly 2 matches). Only `all` is affected.

## Caveats

- 40 datapoints, one source, one seed. The matched sizes were found after the first run (post hoc).
- The PhysicTran38K caption states the outcome, so text alone can describe F3 in every condition.
- `none/first/all` also differ in the number of condition tokens, not only the information they carry.
- `tags.json` is one unblinded reviewer's reading of low-resolution contact sheets; counts are approximate.
