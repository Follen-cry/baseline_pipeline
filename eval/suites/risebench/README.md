# RISEBench — reasoning-informed visual editing

[PhoenixZ810/RISEBench](https://github.com/PhoenixZ810/RISEBench) ("Envisioning
Beyond the Pixels: Benchmarking Reasoning-Informed Visual Editing", NeurIPS'25
Datasets & Benchmarks Oral).

Mirrors `eval/suites/magicbrush/`: `dataset.py` + `inference/gen_risebench_internvlu.py`
for generation, `evaluators/eval_risebench.py` for scoring — a port of RISEBench's
own GPT-based LMM-as-judge (`gpt_eval.py`/`utils.py`), not a metric of our own.

## Data

360 human-annotated single-input-image items across 4 reasoning categories —
temporal (85), causal (90), spatial (100), logical (85) — each with an
editing `instruction` and a free-text `reference` describing the expected
outcome (a judge hint; RISEBench has **no ground-truth image**, unlike
MagicBrush).

Source: HF `PhoenixZ/RISEBench` (`datav2_total_w_subtask.json` + `data.zip`).
**Note**: this HF dataset is Xet-backed — `huggingface_hub.snapshot_download`
stalled indefinitely (0 bytes transferred) fetching `data.zip` when this
suite was set up. Worked around by resolving the plain HTTPS redirect and
`curl`-ing it directly:
```bash
curl -L -o data.zip https://huggingface.co/datasets/PhoenixZ/RISEBench/resolve/main/data.zip
```
then unzip `data.zip` (contains `data/<category>_images/`) alongside the json.

**Known-good copy**, already staged (284M):
```
/scratch/local/ssd/junlin/data/RISEBench/
  datav2_total_w_subtask.json   # [{index, category, subtask, instruction, image, reference}, ...]
  data/temporal_reasoning_images/*.png   (and causal_/spatial_/logical_reasoning_images/)
```
Not vendored into this repo (third-party licensed data, 284M) — point
`--data_root` at the path above, or re-download from `PhoenixZ/RISEBench` if
working from a different machine.

## Protocol

Each of the 360 items is independent — one input image, one instruction, one
generated output. `dataset.py` builds one `RiseBenchItem` per row and assigns
`<category>/<index>.png` as the generated-image filename (e.g.
`temporal_reasoning/temporal_reasoning_1.png`).

## Generation

```bash
CUDA_VISIBLE_DEVICES=<g> python inference/gen_risebench_internvlu.py \
    --data_root /scratch/local/ssd/junlin/data/RISEBench \
    --out /scratch/local/ssd/junlin/results/RISEBench/<Model> --limit 0
```

`--model_path` defaults to the base InternVL-U checkpoint snapshot; point it
at a merged fine-tuned checkpoint dir to evaluate a variant instead. Output:
`<out>/images/<category>/<index>.png` + resumable `<out>/gen_manifest*.json`.
Same `--shard_idx`/`--num_shards` sharding convention as
`suites/magicbrush/inference/gen_magicbrush_internvlu.py`.

**Known dependency gap** (same as `suites/magicbrush/`): the generation
script imports `internvlu` (`InternVLUPipeline`) from
`Model_Related/InternVLU/InternVL-U` in the old `ssl_mllm` tree via a
hardcoded `PKG` path — not part of this repo's `training/models/internvl-u`
submodule (training fork only). Not fixed by this migration.

## Scoring

`evaluators/eval_risebench.py` — GPT-as-judge, scoring Reasoning / Appearance
Consistency / Visual Plausibility, close-ported from upstream's `gpt_eval.py` +
`utils.py` (prompts, per-category branching, and score/completion formulas
carried over verbatim; only the API plumbing differs — any OpenAI-compatible
endpoint, JSON resume cache instead of pickle). See the module docstring for
the exact mapping. Needs a working `OPENAI_API_KEY` (or `--api_key`) — the
keys found in this user's other project `.env` files
(`~/apex/.env`, `~/apex_backup/.env`, `~/red-team/.env`) were all invalid/
revoked as of 2026-09-09 (401 from the API), so a fresh key is needed before
running this for real.

```bash
OPENAI_API_KEY=... python evaluators/eval_risebench.py \
    --data_root /scratch/local/ssd/junlin/data/RISEBench \
    --generated /scratch/local/ssd/junlin/results/RISEBench/<Model>/images \
    --save_path /scratch/local/ssd/junlin/results/RISEBench/<Model> \
    --limit 10   # smoke-test first
```

Default judge model is `gpt-5` (matching this repo's other GPT-judge scripts,
e.g. `Evaluation/editing_benchmarks/PICABench/PicaEval_gpt.py`); pass
`--model gpt-4.1-2025-04-14` to reproduce the paper's exact judge. Writes
`<save_path>/evaluation_metrics.json` (overall + per-category + per-subtask
means, matching `eval_magicbrush.py`'s output convention) and a per-item
`<save_path>/<model>_risebench_judged.json`. Judge calls are cached by item
index in `<save_path>/<model>_judge_cache.json` — safe to re-run/resume.

## Validate-then-run

Use `--limit N` on the generation script to smoke-test a handful of items
end-to-end before committing to the full 360-item run.
