# MagicBrush — single-turn instruction editing (test split)

[OSU-NLP-Group/MagicBrush](https://github.com/OSU-NLP-Group/MagicBrush), CVPR'23.

**Scope of this migration: InternVL-U only.** The original suite (still at
`ssl_mllm/Evaluation/editing_benchmarks/MagicBrush/`) also has generation
drivers for BAGEL-7B-MoT, SenseNova-U1, Emu3.5 and Cosmos3, used to build a
5-model comparison. Those were deliberately **not** brought over here (user
decision, 2026-09-07) — this suite currently only reproduces the InternVL-U
side. Port the others the same way (see "Adding another model" below) if
cross-model comparison is wanted again.

## Data

The official test split is withheld behind a manual-request form (train/dev
only ship on `osunlp/MagicBrush`). We use the public mirror
`akshayg08/MagicBrushTest` (535 sessions / 1053 turns, same
`edit_sessions.json` / `global_descriptions.json` / `images/` layout as the
official release).

**Known-good copy** (verified complete, 535/535 sessions, 2.4G):
```
/scratch/local/ssd/junlin/data/MagicBrush/test/
  edit_sessions.json        # img_id -> [{input, mask, output, instruction}, ...]
  edit_turns.json           # flat list of the same turns
  global_descriptions.json  # img_id -> {filename: caption}, used for CLIP-T
  images/<img_id>/<img_id>-{input,output1,mask1,output2,mask2,...}.png
```
Not vendored into this repo (third-party licensed data, 2.4G of images) —
point `--data_root` at the path above, or re-download `akshayg08/MagicBrushTest`
if working from a different machine.

There is also a **partial, stale copy** at
`ssl_mllm/data/datasets/MagicBrush/test/` (only 133/535 sessions, 582M) —
do not use it, it's an incomplete leftover from an earlier attempt to move
the data onto network scratch.

## Protocol

Each of the 1053 turns is evaluated **independently**: turn *i*'s `input`
field is already the *ground-truth* output of turn *i-1* (verified directly
against the raw JSON), so reading each turn's own `input`/`instruction`/
`output` gives single-turn edits with no iterative chaining through a model's
own prior generations — this is MagicBrush's "independent editing" (`inde`)
setting, not the "iterative" (`iter`) one.

`dataset.py` builds one `MagicBrushItem` per turn and assigns the official
generated-filename convention (`<img_id>_1.png` for turn 1,
`<img_id>_inde_{n}.png` for turn n>1) so the metrics script's pairing logic
can consume outputs from any model without restructuring.

## Generation

```bash
CUDA_VISIBLE_DEVICES=<g> python inference/gen_magicbrush_internvlu.py \
    --data_root /scratch/local/ssd/junlin/data/MagicBrush/test \
    --out /scratch/local/ssd/junlin/results/MagicBrush/<Model> --limit 0
```

`--model_path` defaults to the base InternVL-U checkpoint snapshot; point it
at a merged fine-tuned checkpoint dir to evaluate a variant instead. Output
layout matches what `evaluators/eval_magicbrush.py` expects directly:
`<out>/images/<img_id>/<gen_filename>.png`.

**Known dependency gap**: the generation script imports `internvlu`
(`InternVLUPipeline`) from `Model_Related/InternVLU/InternVL-U` in the old
`ssl_mllm` tree (hardcoded `PKG` path in the script) — this directory is
**not** part of this repo's `training/models/internvl-u` git submodule (that
submodule only mirrors `Model_Related/InternVLU/InternVL`, the internvl_chat
*training* fork; the separate InternVL-U *inference* package was never
migrated in). Same gap exists in `eval/suites/vbvr/inference/run_target_pred_eval.py`.
Not fixed by this migration — this repo isn't yet fully self-contained for
InternVL-U inference, only training.

## Metrics

`evaluators/eval_magicbrush.py` is a close port of the official
`evaluation/image_eval.py` (v3): **L1, L2, CLIP-I, DINO, CLIP-T**, computed
separately for `final_turn` (last edit of each session) and `all_turn` (every
turn). Since we only do single-turn inference, both splits reflect the
"independent editing" setting and are directly comparable to published
MagicBrush independent-setting numbers.

```bash
python evaluators/eval_magicbrush.py \
    --data_root /scratch/local/ssd/junlin/data/MagicBrush/test \
    --generated /scratch/local/ssd/junlin/results/MagicBrush/<Model>/images \
    --save_path /scratch/local/ssd/junlin/results/MagicBrush/<Model>
```

Run in env `geneval-eval-env` (has `openai-clip` + `scipy` + CUDA torch/
torchvision already installed). CLIP/DINO checkpoints auto-download on first
run.

After scoring, copy the small `evaluation_metrics.json` into this suite's
`results/<Model>/` (git-tracked — the generated images themselves stay
external, only the metric summary is committed), then regenerate the
comparison table:

```bash
cp /scratch/local/ssd/junlin/results/MagicBrush/<Model>/evaluation_metrics.json results/<Model>/evaluation_metrics.json
python artifacts/gen_magicbrush_report.py   # writes artifacts/magicbrush_comparison.md
```

## Results already computed (9 InternVL-U checkpoints)

`results/<checkpoint>/evaluation_metrics.json`, table in
`artifacts/magicbrush_comparison.md`:

| checkpoint | family |
|---|---|
| `InternVL-U` | base |
| `InternVL-U-ffs-sft` | IntPhys2 FFS-MCQ SFT |
| `InternVL-U-gen-sft-vae` | IntPhys2 VAE-conditioned gen SFT |
| `InternVL-U-dualssl-leak` | IntPhys2 dual-SSL, leak variant |
| `InternVL-U-nwm-gen-action` / `-noaction` | NWM gen SSL |
| `InternVL-U-nwm-dual-action` / `-noaction` | NWM dual SSL |
| `InternVL-U-nwm-edit-action` | NWM edit-conditioned |

## Adding another model

1. Write `inference/gen_magicbrush_<model>.py` following
   `gen_magicbrush_internvlu.py`'s manifest/resume/shard conventions (input:
   `dataset.py`'s `MagicBrushItem`; output: `<out>/images/<img_id>/<gen_filename>.png`
   + `<out>/gen_manifest*.json`).
2. Score with the existing `evaluators/eval_magicbrush.py` unchanged — it's
   model-agnostic.
3. Copy `evaluation_metrics.json` into `results/<Model>/`, add the name to
   `MODEL_ORDER` in `artifacts/gen_magicbrush_report.py`, regenerate the report.

## Validate-then-run

Use `--limit N` on the generation script to smoke-test a handful of turns
end-to-end (generation + metrics) before committing to the full 1053-turn run.
