# `eval/suites/v2_temporal_ssl/` — v2 eval suite

Held-out eval for T0-T4 on the v2 window pool's eval split (750 windows, 125 per source).
Existing suites (vbvr, worldprediction, ...) stay v1 code; v2 checkpoints are added to them as
extra rows. See `docs/v2.md`.

## Eval sets

| File (`data/v2/datasets/temporal_ssl/settings/<run>/`) | Rows | Use |
|---|---|---|
| `T{0..4}_eval.jsonl` | 750 | final eval |
| `T{0..4}_evalmini.jsonl` | 78 (13 per source, stratified by GAP) | in-training validation |

Both use the training row format unchanged (same `conversations`, `image`, `cond_image`,
`target_image`), plus `answer` / `layout` for scoring. The evalmini windows (`evalmini_ids.json`,
built by `data/v2/recipes/temporal_ssl/make_eval_mini.py`) are the same in every setting, so the
T0-T4 curves are comparable. T4 eval rows are all T4-B; a T4 model's T4-A validation is
`T0_evalmini`.

## Inference (`training/models/internvl-u/internvl_chat/internvl/train/v2_validation.py`)

Uses the training ("imgen") path on the unified model's own modules, so a checkpoint is
evaluated exactly as it was trained:

1. T1 / T3 / T4-B: greedy text from the row's human turn (training tokenization) until `<img>`.
2. The row is re-tokenized with GPT turn `<text>\n<img>` (`<img>` for T0 / T2) by
   `MultimodalImgenLazyDataset.eval_item` (no CFG dropout, errors raise instead of silently
   swapping rows).
3. The decoder is conditioned like `_compute_gen_loss` (VLM hidden states + VAE latent of the
   oracle `cond_image`) and sampled with the pipeline scheduler, CFG 1, 20 steps, seed =
   crc32(row id), at the size training resizes that row's target to (`--gen-resize-mode area`,
   default: its aspect ratio at area ~`--gen-image-size`² = 512², sides ×16; `keep_aspect` = long
   side ≤ S; `square` = v1's S×S squash).

Predictions: `<out>/<T>/preds.jsonl` lines `{"id", "response", "image"}` + `images/`,
`per_row.jsonl`, `summary.json`.

Standalone (base model or any pipeline dir), run from `internvl_chat/` with
`PYTHONPATH=$(pwd)` and `INTERNVLU_PKG_PATH` set:

```bash
torchrun --nproc_per_node=4 -m internvl.train.v2_validation --model <snapshot> \
    --eval <settings/main/T0_evalmini.jsonl> --eval <...T4_evalmini.jsonl> --out <dir> \
    [--wandb-project internvlu-v2-val --wandb-run-name base_zeroshot] [--limit 3]
```

In training (`engine/internvlu_4b_sft_full_unified.sh`): set `VAL_EVAL_JSONL` (comma list of
evalmini files; a T4 run should pass both `T4_evalmini` and `T0_evalmini`) and optionally
`VAL_STEPS` (500), `VAL_AT_START` (1), `VAL_DIFFUSION_STEPS` (20), `VAL_LOG_IMAGES` (6).
Metrics go through `Trainer.log` as `val/<T>/<group>/<metric>` (tensorboard + wandb with
`REPORT_TO="tensorboard wandb"`), sample grids (context | GT | prediction) to wandb.
Outputs: `$OUTPUT_DIR/val/stepNNNNNN/<T>/`.

## Scoring (`score.py`, rule-based)

```bash
python score.py --eval <T3_evalmini.jsonl> --pred preds.jsonl --out results/T3_step500
python score.py --eval <T3_evalmini.jsonl> --oracle copy --out results/T3_copy   # baseline / sanity
```

- **Text** (grammar from `data/v2/common/prompts.py` `parse_answer`): `parse_ok`, `gap_acc`
  (T1, T4-B), `order_exact` (swaps of `answer.order_equiv` pairs accepted) and `order_pair`
  (T3, T4-B), `missing_acc` and `all_correct` (T4-B). A parse failure scores 0 everywhere.
  Chance: order 1/6, missing 1/3, gap = majority-class share (stored in `summary.json`).
- **Image**, at each row's training target size (`--resize area --size 512`, the defaults,
  = `gen_resize_mode` / `gen_image_size`; `--resize square --size 512` = the v1 squash; results
  under `results/base_zeroshot/` predate the switch and were scored at 512² square):
  `psnr`, `ssim`, `mae` vs target; the same for the copy-`cond_image` baseline (`*_copy`);
  `dpsnr` / `dssim` (gain over copy); `beat_copy` (share of rows with lower MSE than copy);
  `motion_psnr` / `dmotion_psnr` on pixels where target and `cond_image` differ by > 0.1.
  Raw PSNR/SSIM are dominated by static background (copy-baseline SSIM on evalmini is 0.86,
  0.98 on VBVR), so read the curves from `dpsnr`, `dmotion_psnr` and `beat_copy`.
- **Groups**: `all`, `src=`, `gap=`, `target=` (F1/F2/F3 for T2 / T4-B), `moving` (non-stalled).
  `summary.json["metrics"]` is a flat `{"<group>/<metric>": value}` dict for `wandb.log`.
