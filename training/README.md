# training/

## `models/`

One git submodule per model family. Currently one entry:

- **`models/internvl-u/`** — the InternVL-U training fork (upstream:
  [OpenGVLab/InternVL](https://github.com/OpenGVLab/InternVL)). Its local git history starts
  fresh from a single "initial snapshot" commit made during this repo's migration — the
  original fork was never `git init`'d locally, so no literal upstream history could be
  carried over (see that commit's message for details).

  **Caveat**: the submodule's `.gitmodules` URL is currently an absolute local filesystem path
  (`/scratch/network/ssd2/junlin/ssl_mllm/Model_Related/InternVLU/InternVL`), since both repos
  live on the same scratch volume. If this repo is ever cloned elsewhere, that submodule URL
  will need to be re-pointed (e.g. to a pushed remote) before `git submodule update` will work.

## No separate `recipes/` or `orchestration/` directories

The originally-planned `training/recipes/` and `training/orchestration/` top-level dirs turned
out to be redundant once the submodule was set up: the actual S0-S3 recipe scripts and the
orchestration script that runs them **already live inside the submodule**, at
`models/internvl-u/internvl_chat/shell/internvlu/{sft,orchestration,engine}/`. Duplicating them
at this repo's top level would just be two copies of the same files. If a *second* model family
is added later with its own recipes, the same pattern applies — recipes stay inside that
model's own submodule, under whatever training-launcher convention that codebase uses.

## Where the S0-S3 baseline recipe actually is

See `../PROVENANCE.md` §4 for the full, log-verified detail. Summary of what to run and in what
order:

1. **Stage 1 (independent, not chained)** — one script per setting, all defaulting
   `INTERNVLU_CKPT` to the same base snapshot:
   - `models/internvl-u/internvl_chat/shell/internvlu/sft/run_s0_gen_sft.sh`
   - `models/internvl-u/internvl_chat/shell/internvlu/sft/run_s1_gen_sft.sh`
   - `models/internvl-u/internvl_chat/shell/internvlu/sft/run_s2_gen_sft.sh`
   - `models/internvl-u/internvl_chat/shell/internvlu/sft/run_s3_gen_sft.sh`
   - Orchestrated (optionally) by
     `models/internvl-u/internvl_chat/shell/internvlu/orchestration/orchestrate_s0s3_torrnode8.sh`
     — but per PROVENANCE.md §4, the real historical run needed some manual intervention when
     the orchestrator's automated launches failed; don't assume a clean unattended run.
   - Real training logs for the actual run (including the failed attempts) are committed inside
     the submodule at `internvl_chat/logs/s0s3_torrnode8/` (force-added past the submodule's own
     `*.log` gitignore rule, since they're cited provenance evidence).
2. **Stage 2 (continuation SFT, target_pred)** —
   `models/internvl-u/internvl_chat/shell/internvlu/sft/run_target_pred_mop_sft.sh`, run once
   per stage-1 checkpoint with `INTERNVLU_CKPT` overridden to each `internvlu-s{N}-gen-merged`.
   Exact confirmed invocations: PROVENANCE.md §5.

Data paths (`META_PATH` in each script) point at `data/meta/final_S{0,1,2,3}_meta.json` and
`data/meta/vbvr_target_pred_4task_meta.json` — see `../data/README.md` for how those are
produced.

Checkpoints themselves are **not** part of this repo (too large, and this repo intentionally
stays code+small-data only) — they stay wherever they were trained
(`/scratch/network/ssd2/junlin/models/` for stage 1, `/scratch/network/ssd/junlin/models/` for
stage 2 — note the two different volumes, see PROVENANCE.md §5).
