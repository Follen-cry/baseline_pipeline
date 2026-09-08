# eval/inference/

Generic model-inference runners shared across 2+ suites (the bucket `../README.md` flagged as
missing before this pass — everything under `eval/suites/` used to be fully self-contained).

## `vlmevalkit/`

A **trimmed vendored snapshot** of `Evaluation/VLMEvalKit/` from the old `ssl_mllm` tree, not a
straight copy: that source dir is 193M; this is ~21M. Per this repo's migration rule, raw
copies get reorganized to actual scope rather than carried over wholesale. No submodule (unlike
`../suites/worldprediction/`) — the source dir has no real upstream git history of its own to
preserve (it's untracked loose files sitting in the old tree, not a separate repo), so it's
vendored as a plain snapshot, same treatment given to the VBVR-DataFactory generators during
the original migration pass.

**Kept**: `vlmeval/` (the toolkit package, incl. the custom `vlmeval/vlm/internvlu.py` InternVLU
adapter), `run.py`, `scripts/run_internvlu.sh`, `requirements*`, `setup.py`/`setup.cfg`,
`LICENSE`, `README.md` (upstream docs), and `reference_results/internvlu_fixed/` (renamed from
`outputs_internvlu_fixed/` — a known-good prior run: 11 InternVL-U checkpoints × CV-Bench-2D/3D
+ RealWorldQA, using the *current* prompt/answer-extraction treatment; `outputs_internvlu/`,
its stale pre-fix predecessor per `run_internvlu.sh`'s own comment, was **not** carried over).

**Dropped** (present in the old tree, not here): every other model family's result dumps —
`outputs_sensenova/` (SenseNova-U1, a different model), `outputs_cosmos3_spatial/`,
`outputs_spatial_xmodel/` + `logs_xmodel/`, `outputs_nwm_spatial/`, `outputs_vaecond/`,
`outputs_ffsmcq/`, `outputs_spatial_multigpu*/`, `outputs_reasoning_edit*/`,
`outputs_intphys2gen*/`, `outputs_newmodels_spatial/`, assorted root `*.log` files, `docs/`,
`.github/`, `assets/`, `.pre-commit-config.yaml`, `_salvage_from_old_fork/` (superseded scratch
from when `vlmeval/`+`run.py` were originally merged back together), and every `scripts/*`
other than `run_internvlu.sh` (`run_cosmos3.sh`, notebooks, misc one-off scripts) — none of
these are in scope for this repo's InternVL-U-only suites.

### Known gap fixed during vendoring: stale checkpoint registry

`vlmeval/config.py`'s pre-existing `internvlu_series` dict (the bulk of its entries) points at
**exploratory-era checkpoints** (`internvlu-clevrer-ssl-*`, `internvlu-ffs-1ep-pipeline`,
`internvlu-invdyn-*`, etc.) — none of these appear in `../../Model_Related/InternVLU/
CHECKPOINTS.md` (not vendored into this repo's `training/` yet — see that file directly in the
old tree) and none are the S0-S3 baseline's actual object of comparison. **Added** a new block
of 6 entries at the end of that dict — `InternVL-U-{base,s0,s1,s2,s3,s4}-4task500sft` —
pointing at the real stage-2 `target_pred` checkpoints
(`/scratch/network/ssd/junlin/models/internvlu-{name}-4task500sft-merged`, verified to exist on
disk 2026-09-08). Use *these* six for any RealWorldQA/SeedBench run meant to compare against
`../suites/vbvr/`'s results — the pre-existing entries above them in the dict are unrelated
leftovers from earlier exploratory work, left in place (not deleted) since other things may
still reference them.

### Import gap fixed: `paths.py`

`vlmeval/vlm/internvlu.py` (and `sensenova_u1.py`, pulled in transitively by `vlmeval/vlm/
__init__.py`'s unconditional imports) does `from paths import add_internvlu_to_path`, resolving
3 directories up from its own location — in the old tree that landed on `Evaluation/paths.py`;
here it resolves to `eval/inference/paths.py`, which didn't exist. Added a trimmed shim there
(just `add_internvlu_to_path`/`add_sensenova_to_path`, hardcoding the same absolute
`Model_Related/InternVLU/InternVL-U` path as `suites/magicbrush`/`suites/risebench`/
`suites/aurorabench`'s inference scripts — same known gap, not fixed generically, just made
consistent here too). Without it, `import vlmeval` fails outright.

**Verified working end-to-end** (2026-09-08, GPU 7): both `RealWorldQA` (n=765) and
`SEEDBench_IMG` (n=14,232) build their dataset (auto-downloading the TSV to
`~/LMUData/`), load `InternVL-U-base-4task500sft` (the real merged S0-S3 baseline checkpoint,
not a placeholder), build a correctly-formatted MCQ prompt, and generate a coherent answer —
SeedBench's option format did **not** trip up the prompt adapter (the risk flagged in
`suites/seedbench/README.md`).

### SeedBench needed no dataset-side changes

`SEEDBench_IMG` was already a recognized `--data` value (`vlmeval/dataset/image_mcq.py`'s
`DATASET_URL`, the same generic MCQ dataset class RealWorldQA/CV-Bench use) — it self-downloads
its TSV on first run. Only the checkpoint-registry fix above was needed, and it already covers
both `suites/realworldqa/` and `suites/seedbench/`.

### Usage

```bash
cd eval/inference/vlmevalkit
MODELS="InternVL-U-base-4task500sft InternVL-U-s0-4task500sft InternVL-U-s1-4task500sft \
        InternVL-U-s2-4task500sft InternVL-U-s3-4task500sft InternVL-U-s4-4task500sft" \
DATA="RealWorldQA" \
bash scripts/run_internvlu.sh
```

`DATA` is space-separated and accepts multiple benchmark names in one run (e.g.
`"RealWorldQA SEEDBench_IMG"`). See `suites/realworldqa/README.md` / `suites/seedbench/
README.md` for the per-suite invocation and output-file layout.
