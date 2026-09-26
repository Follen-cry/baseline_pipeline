# `data/v2/recipes/temporal_ssl/` — v2 recipe (T0-T4)

Turns the v2 sources into one shared window pool, then into the five v2 settings.
Full, reproducible description of the pool: `docs/v2.md`, "Window pool: how to reproduce".

- `clip_sources.py` — per-source loaders: frozen selection manifest → uniform clip rows
  (`source, clip_id, video, caption, caption_source, category, source_meta`). Captions are each
  source's own text, used verbatim; MiT uses the Qwen3-VL caption from its VLM QC pass (module
  docstring has the field per source).
- `merge_pools.py` — probe → score → plan → extract → finalize, seeded (42), per source. Ready to
  run; dry-run numbers below. Every build is a named **run** (`--name`, default `main`):
  - `main`: 60,000 train windows over 6 sources, PhysicTran38K 23.5% (14,118), PhysInOne /
    VBVR / BAAI 17.6% each (10,588), MiT / SSv2 11.8% each (7,059); eval = 125 whole clips per
    source, one window each. PhyCo-Sim and Panda-70M were dropped (2026-09-26).
    Full clips, global window rule (`common/window_sampler.py`, Δt ∈ {0.5, 1, 2} s).
  - Static-window filter (`score` stage, `common/window_motion.py`): windows where none of F1..F3
    differs from F0 in ≥ 0.1% of pixels are dropped before planning, so quotas stay exact.
    Windows that stop part-way are kept and flagged (`motion.stalled`).
  - Δt balance: within each source the quota is split evenly over the Δt it supports (water-fill);
    dry run of `main` gives 44.9 / 44.0 / 11.1 % for Δt 0.5 / 1 / 2 (2.0 needs ≥ 6 s clips, only
    BAAI, VBVR, PhysInOne have them); 11.1% of windows are `stalled`.
  - Train quota is spread over clips in rounds (every clip one window before any gets a second).
  - VBVR draws only from `split == "train"` of its manifest, task-stratified.
  - Shared per source: probe/score caches (`<work>/{probe,score}_<src>.jsonl`) and frames
    (`v2_frames/<src>/<clip>/`, native resolution capped to fit 1024×1024, never written twice). Per run: `<work>/runs/<name>/` and
    `datasets/temporal_ssl/pools/<name>/`. `<work>` = `/scratch/network/ssd/junlin/v2_frames/_work/temporal_ssl/`.
  - Extra single-source set, same eval as `main`, disjoint from `main`'s train windows:
    `python merge_pools.py --stage all --name physinone_extra --sources physinone --quota physinone=10000 --eval-from main --exclude-from main`
- `derive_settings.py` (planned) — seeded pool → `T{0..4}_{train,eval}.jsonl` (ordering, masking,
  shuffling, T4-A/B assignment, `cond_image` = observed frame nearest the target).
- `validate.py` (planned) — label-balance checks (GAP, the 6 ORDER permutations, MISSING k).

Output: `../../datasets/temporal_ssl/pools/<run>/<src>_{train,eval}.jsonl` + `pools/<run>/summary.json`.
Setting definitions and decisions: `docs/v2.md`.
