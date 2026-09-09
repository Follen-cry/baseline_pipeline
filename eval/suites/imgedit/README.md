# ImgEdit-Bench — unified instruction-based image editing

[PKU-YuanGroup/ImgEdit](https://github.com/PKU-YuanGroup/ImgEdit) ("ImgEdit:
A Unified Image Editing Dataset and Benchmark", arXiv:2505.20275). ImgEdit is
both a 1.2M-pair training dataset and a 3-part benchmark; this suite covers
**Basic-Bench** only — the subset most papers cite as "ImgEdit-Bench".

**Scope of this suite: InternVL-U generation + local judge scoring.** Unlike
`risebench`/`aurorabench` (generation only, official scoring never ported),
this suite also ports ImgEdit's evaluation — but via the local
`ImgEdit_Judge` checkpoint (`evaluators/imgedit_judge.py`), not the official
GPT-4o API path, since the judge model is a redistributable ~8.3B-param
checkpoint upstream ships specifically as a non-API alternative.

## Data

737 single-input-image items across 9 edit-type categories — action, add,
adjust, background, compose, extract, remove, replace, style — each with an
editing `prompt`. **No ground-truth output image** (like RISEBench/
AURORA-Bench), scored instead by comparing input vs. generated output
against a per-edit_type GPT-4o rubric.

Source: HF `sysuyy/ImgEdit` (`Benchmark.tar`, 48M — small enough to vendor
whole, unlike RISEBench/AURORA-Bench's much larger third-party sets).
**Note**: this HF dataset is Xet-backed like those two — `snapshot_download`
may stall; the direct-HTTPS workaround (see `suites/risebench/README.md`)
works fine here too and is how this copy was fetched:
```bash
curl -L -o Benchmark.tar https://huggingface.co/datasets/sysuyy/ImgEdit/resolve/main/Benchmark.tar
tar -xf Benchmark.tar   # -> Benchmark/{singleturn,hard,multiturn}/
```

**Known-good copy**, already staged (49M):
```
/scratch/local/ssd/junlin/data/ImgEdit/Benchmark/singleturn/
  singleturn.json      # {key: {id, prompt, edit_type}, ...} (737 items) — upstream calls this basic_edit.json
  judge_prompt.json     # {edit_type: gpt_rubric_prompt} — upstream calls this prompts.json
  <edit_type_or_category>/*.jpg   # e.g. animal/, architecture/, style/, compose/{human,objects,animal}/
```
Not vendored into this repo (third-party licensed data) — point `--data_root`
at the path above (specifically the `singleturn/` subdir), or re-download
from `sysuyy/ImgEdit` if working from a different machine.

`Benchmark.tar` also contains `Benchmark/hard/` (UGE-Bench source images,
47 items — metadata is `Benchmark/UGE/UGE_edit.json` on GitHub, not bundled
in the tar) and `Benchmark/multiturn/{content_memory,content_understand,
version_backtrace}/` (Multiturn-Bench, each with a bundled `annotation.json`)
— both already staged alongside `singleturn/` above but **not wired up by
this suite**. Add sibling `dataset.py`/`inference/` pairs following this
one's shape if/when those get ported.

## Protocol

Each of the 737 items is independent — one input image, one instruction, one
generated output. `dataset.py` builds one `ImgEditItem` per row and assigns
`<edit_type>/<key>.png` as the generated-image filename (e.g.
`adjust/1082.png`) — upstream's own scripts only require the filename
*prefix* to match the dict key (flat `<key>.png`), so this nesting is our
convention, not a compatibility requirement.

`compose` items reference a subject/reference image pair via the raw `id`
path (e.g. `compose/human/6.jpg`); this loader treats it like any other
single `input_path`, same as upstream.

## Generation

```bash
CUDA_VISIBLE_DEVICES=<g> python inference/gen_imgedit_internvlu.py \
    --data_root /scratch/local/ssd/junlin/data/ImgEdit/Benchmark/singleturn \
    --out /scratch/local/ssd/junlin/results/ImgEdit/<Model> --limit 0
```

`--model_path` defaults to the base InternVL-U checkpoint snapshot; point it
at a merged fine-tuned checkpoint dir to evaluate a variant instead. Output:
`<out>/images/<edit_type>/<key>.png` + resumable `<out>/gen_manifest*.json`.
Same `--shard_idx`/`--num_shards` sharding convention as
`suites/magicbrush/inference/gen_magicbrush_internvlu.py`.

ImgEdit ships some very large source images (up to 8.29MP,
`compose/objects/2.jpg`) that exceed the InternVL-U pipeline's VAE-decode
capacity — the generation script reuses `suites/risebench/`'s measured
1.8MP-safe downscale cap, and it's load-bearing here (not just precautionary
like in `suites/aurorabench/`).

**Known dependency gap** (same as `suites/magicbrush/`): the generation
script imports `internvlu` (`InternVLUPipeline`) from
`Model_Related/InternVLU/InternVL-U` in the old `ssl_mllm` tree via a
hardcoded `PKG` path — not part of this repo's `training/models/internvl-u`
submodule (training fork only). Not fixed by this migration.

## Scoring

The official evaluation (`Benchmark/Basic/basic_bench.py` +
`step1_get_avgscore.py` + `step2_typescore.py` in the upstream repo) calls
GPT-4o to rate each edit on 3 aspects (label names vary by edit_type, e.g.
Prompt Compliance / Visual Seamlessness / Physical & Detail Fidelity for
`adjust`) using the per-edit_type rubric in `judge_prompt.json`, 1-5 each,
averages the three scores per item, then averages per edit_type.

This suite ports that via the **local** judge upstream also publishes
instead of calling GPT-4o: `ImgEdit_Judge` (HF `sysuyy/ImgEdit`,
`ImgEdit_Judge/` — a Qwen2.5-VL-7B-Instruct fine-tune, ~8.3B params, 16.6GB
bf16 across 4 safetensors shards). Staged at
`/scratch/network/ssd2/junlin/models/ImgEdit_Judge/` (network storage, not
node-local `/scratch/local/ssd` — that volume was at 99% free space when this
was set up). `evaluators/imgedit_judge.py` follows the exact chat-template +
`qwen_vl_utils.process_vision_info` + `min_pixels`/`max_pixels` invocation
from the upstream README's "Setups for ImgEdit-Judge" demo (deviating risks
a train/inference preprocessing mismatch), with `attn_implementation` left at
transformers' sdpa default rather than upstream's `flash_attention_2` — same
reason `suites/vbvr/evaluators/llm_judge_transformers.py` made that call:
this cluster's driver can't run the separately-compiled flash-attn extension.
Response parsing (`extract_scores_and_average`) is `step1_get_avgscore.py`
ported verbatim — the judge's output is free text ending in "<label>: <digit>"
lines, not JSON, same format GPT-4o would produce.

`evaluators/score_imgedit.py` folds upstream's 3-stage
basic_bench.py/step1/step2 pipeline into one script (no ThreadPoolExecutor —
one local GPU-bound model, called sequentially, not an API):

```bash
CUDA_VISIBLE_DEVICES=<g> python evaluators/score_imgedit.py \
    --data_root /scratch/local/ssd/junlin/data/ImgEdit/Benchmark/singleturn \
    --run_dir /scratch/local/ssd/junlin/results/ImgEdit/<Model> --limit 0
```

Env: `imgedit-eval-env` or `internvlu` (both have `qwen_vl_utils` + a
`transformers` new enough for `Qwen2_5_VLForConditionalGeneration`). Reads
`<run_dir>/gen_manifest.json` from a completed generation run; writes
`<run_dir>/judge_raw.json` (per-item raw response + parsed score, resumable
like the generation script) and `<run_dir>/judge_typescore.json`
(per-edit_type averages + an `"overall"` key).

## Validate-then-run

Use `--limit N` on the generation script to smoke-test a handful of items
end-to-end before committing to the full 737-item run.
