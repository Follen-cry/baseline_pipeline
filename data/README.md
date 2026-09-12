# data/

```
sources/<name>/    one dir per data source (code + a raw/ pointer), e.g. vbvr, epic_kitchens,
                   nwm, panda70m_epic, panda70m_v1, worldprediction — see each one's README
common/            cross-source processing code (used by more than one source)
recipes/<name>/    assembly code that mixes pre-split source outputs into one final dataset
meta/              InternVL-trainer-format meta jsons pointing at datasets/
datasets/<name>/   final output jsonl of a recipe — what training scripts actually read
```

Adding a new data source later = a new `sources/<name>/` dir, nothing else changes. Adding a
new training-data mix = a new `recipes/<name>/` dir producing a new `datasets/<name>/`, without
touching any existing source's code.

Full ground-truth provenance for the current `s0s3_baseline` recipe (exact files, config
values, row counts, what's confirmed vs. still-open): `../PROVENANCE.md`.

**Raw video/frame data is not migrated into this repo yet** — every source's `raw/` is
currently just a README pointer to where it lives in the original `ssl_mllm` working tree
(some of it node-pinned, only reachable from specific compute nodes). That's a deliberate,
separate follow-up pass; see each `sources/<name>/README.md`.
