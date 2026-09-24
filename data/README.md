# data/

Split by version (see `../docs/VERSIONS.md`). Both versions use the same five sub-layers:

```
v1/   S0-S4 series + experiments built on it       (see ../docs/v1.md)
v2/   T0-T4 temporal SSL                            (see ../docs/v2.md)

<version>/
├── sources/<name>/    one dir per data source: processing code, README (raw-data location,
│                      split unit, seed)
├── common/            cross-source code (used by more than one source)
├── recipes/<name>/    assembly code mixing pre-split source outputs into one final dataset
├── meta/              InternVL-trainer meta jsons pointing at datasets/
└── datasets/<name>/   final output jsonl of a recipe, which is what training scripts read
```

Adding a data source = a new `<version>/sources/<name>/`; adding a training-data mix = a new
`<version>/recipes/<name>/` producing a new `<version>/datasets/<name>/`. Neither touches an
existing source's code.

**Most frames are not in this repo.** v1's S0-S4 rows point at frame directories under
`/scratch/network/ssd/junlin/...`, and v2 frames will go to `/scratch/network/ssd/junlin/v2_frames/`.
The exceptions are v1's `datasets/worldprediction/frames*/` and `datasets/*_Composed_intermediate*/images/`,
which live in the repo directory but are gitignored. Full v1 provenance: `../PROVENANCE.md`.
