# baseline_pipeline

Standalone, reproducible repo for the S0-S3 InternVL-U baseline: the full data
generation/processing → train/eval split → training → eval pipeline, migrated out of the
larger `ssl_mllm` working tree so it's self-contained and easy to build on for future
experiments (new models, new eval suites, new training data) without re-scattering everything
again.

**Ground truth for exactly which files/configs were used in the real S0-S3 run**:
[`PROVENANCE.md`](PROVENANCE.md). Everything below is a map of *where things live now*;
PROVENANCE.md is the record of *what actually happened* and *why it's organized this way*.
When the two disagree, trust PROVENANCE.md.

## Versions

- **v1** — S0-S4 series and everything built on it: `data/v1/`, launchers in the submodule's
  `sft/v1/` + `orchestration/v1/`. [`docs/v1.md`](docs/v1.md)
- **v2** — T0-T4 temporal SSL: `data/v2/`, launchers in `sft/v2/` + `orchestration/v2/`.
  [`docs/v2.md`](docs/v2.md)

Overview and path map: [`docs/VERSIONS.md`](docs/VERSIONS.md). Git tag `v1` pins the layout
from before the v1/v2 split (when `data/v1/*` sat directly under `data/`).

## Layout

```
data/
├── v1/                  S0-S4 and its follow-up experiments
│   ├── sources/<name>/  one dir per data source (vbvr, epic_kitchens, nwm, panda70m_epic,
│   │                    panda70m_v1, worldprediction): generators/processing code + README
│   ├── common/          cross-source processing code (used by 2+ sources)
│   ├── recipes/<name>/  assembly code mixing pre-split source outputs into one final dataset
│   ├── meta/            InternVL-trainer meta jsons (annotation path + loader flags)
│   └── datasets/<name>/ final training-ready jsonl, which is what training scripts actually read
└── v2/                  T0-T4 temporal SSL (same five sub-layers; scaffolding for now)

training/
└── models/internvl-u/   git submodule, the InternVL-U training fork (upstream: OpenGVLab/InternVL).
                         Launchers live inside it under shell/internvlu/{sft,orchestration}/{v1,v2}/;
                         see training/README.md.

eval/
├── inference/           shared inference (vlmevalkit)
└── suites/<name>/       one dir per eval type; shared across versions (they compare v1 and
                         v2 checkpoints side by side), plus v2_temporal_ssl/ for v2's own eval
```

Adding something new never requires touching this top-level shape:
- new data source → new `data/<version>/sources/<name>/`
- new training-data mix → new `data/<version>/recipes/<name>/`
- new model family → new `training/models/<name>/` submodule
- new eval type → new `eval/suites/<name>/`
- new training generation → new `data/v<N>/` + `sft/v<N>/` + `docs/v<N>.md`

## What's migrated vs. not (this pass)

**In**: all code (generators, processing scripts, recipe assembly, training submodule, eval
suite) + meta jsons + the final `data/v1/datasets/{final_s0s3,vbvr_target_pred_4task}/` jsonl
(~247M total).

**Not yet**: raw video/frame data for any of the 5 sources (large; some node-pinned to
specific compute nodes — see each source's README under `data/v1/sources/`), and the actual
model checkpoints (stay wherever they were trained, paths documented in `PROVENANCE.md` §4-§5).
Both are planned as a separate follow-up migration pass.

## Two structural notes from migration (deviations worth knowing about)

- **`data/v1/recipes/`**, not `data/splits/`. The original plan sketched a generic
  `data/splits/` tier for train/eval separation. Tracing the real pipeline showed every source
  decides its own split internally (video/trajectory/clip-level, inside its own
  `data/v1/sources/<name>/processing/` code) — there's no central "splits" logic to factor out.
  What *is* central is assembling several already-split sources into one mix, which is what
  `data/v1/recipes/` actually holds. See `data/v1/recipes/README.md`.
- **No `training/recipes/` or `training/orchestration/`** at the top level. Both turned out to
  already live inside the `training/models/internvl-u` submodule (that's how the original
  codebase organized it) — duplicating them here would just be a second copy of the same
  files. See `training/README.md`.
