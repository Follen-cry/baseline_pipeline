# AURORA-Bench — action & reasoning-centric image editing

[McGill-NLP/AURORA](https://github.com/McGill-NLP/AURORA) ("Learning Action
and Reasoning-Centric Image Editing from Videos and Simulations", NeurIPS'24
Datasets & Benchmarks).

**Scope of this suite: inference only, InternVL-U.** Mirrors
`eval/suites/magicbrush/` — only `dataset.py` + `inference/gen_aurorabench_internvlu.py`
are ported here (no evaluator/judge yet; the official protocol is an LMM
judge / human raters on the generated edit, not a metric we've ported).

Not to be confused with the repo's other, custom EPIC-derived AURORA-style
evals already documented at `results/04_unified_benchmarks/editing/AURORA_EPIC200_V2.md`
and `data/scripts/build_aurora_epic_ext150.py` — those apply AURORA's
0/50/100 rubric to a hand-built EPIC-Kitchens set, not the official
benchmark. This suite is the official 400-item `test` split.

## Data

400 items, 8 source datasets × 50 each (magicbrush, ag [Action-Genome],
something [Something-Something], clevr, whatsup, emu [Emu-Edit], epic
[Epic-Kitchens], kubric) — an `input` image and an editing `instruction`
each. **No ground-truth output image** in this split (AURORA-Bench is scored
by LMM judge / human raters on the generated edit, like RISEBench).

Source: HF `McGill-NLP/aurora-bench` (`data/test-00000-of-00001.parquet`,
400 rows, columns `input`/`instruction`/`source`). The parquet embeds images
as raw bytes with **no stable per-row id** — `input.path` (e.g.
`368667-output1.png`) is reused across rows since AURORA chains edits (an
earlier pair's *output* becomes a later pair's *input*) — so
`extract_from_hf.py` in this suite materializes the parquet to a json +
`images/` layout with a fresh `<source>_<running-index>` key per row (this
mirrors what the official GitHub repo ships directly, per its README: "if
you simply want a json and image folders (I personally find parquet
confusing)"). Re-run it if the data needs regenerating:
```bash
python extract_from_hf.py --out /scratch/local/ssd/junlin/data/AuroraBench
```
(This dataset is also Xet-backed on HF — see `suites/risebench/README.md`'s
note on `snapshot_download` stalling; if `hf_hub_download` inside this script
hangs, `curl -L -o test.parquet https://huggingface.co/datasets/McGill-NLP/aurora-bench/resolve/main/data/test-00000-of-00001.parquet`
and point `pd.read_parquet` at the local file instead.)

**Known-good copy**, already staged (83M):
```
/scratch/local/ssd/junlin/data/AuroraBench/
  test.json                        # [{key, source, instruction, input, orig_filename}, ...]
  images/<source>/<key>.png        # e.g. images/magicbrush/magicbrush_000.png
```
Not vendored into this repo (third-party licensed data, 83M) — point
`--data_root` at the path above, or re-run `extract_from_hf.py` if working
from a different machine.

## Protocol

Each of the 400 items is independent. `dataset.py` builds one
`AuroraBenchItem` per row and assigns `<source>/<key>.png` as the
generated-image filename.

## Generation

```bash
CUDA_VISIBLE_DEVICES=<g> python inference/gen_aurorabench_internvlu.py \
    --data_root /scratch/local/ssd/junlin/data/AuroraBench \
    --out /scratch/local/ssd/junlin/results/AuroraBench/<Model> --limit 0
```

`--model_path` defaults to the base InternVL-U checkpoint snapshot; point it
at a merged fine-tuned checkpoint dir to evaluate a variant instead. Output:
`<out>/images/<source>/<key>.png` + resumable `<out>/gen_manifest*.json`.
Same `--shard_idx`/`--num_shards` sharding convention as
`suites/magicbrush/inference/gen_magicbrush_internvlu.py`.

**Known dependency gap** (same as `suites/magicbrush/`): the generation
script imports `internvlu` (`InternVLUPipeline`) from
`Model_Related/InternVLU/InternVL-U` in the old `ssl_mllm` tree via a
hardcoded `PKG` path — not part of this repo's `training/models/internvl-u`
submodule (training fork only). Not fixed by this migration.

## Scoring (not ported here)

Official protocol: automatic metrics (`disc_edit.py` / `eval_disc_edit.py`
in the upstream repo) plus primary human evaluation on four edit types
(object/attribute, action-centric, reasoning-centric, global). This suite
currently only reproduces generation; add an `evaluators/eval_aurorabench.py`
here if/when scoring gets ported, following
`suites/magicbrush/evaluators/eval_magicbrush.py`'s shape — likely an LMM
judge given AURORA-Bench ships no GT image, closer to how `suites/vbvr/`'s
`judge_eval/` is structured than to MagicBrush's pixel metrics.

## Validate-then-run

Use `--limit N` on the generation script to smoke-test a handful of items
end-to-end before committing to the full 400-item run.
