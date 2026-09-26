# `data/v2/datasets/temporal_ssl/` — v2 datasets

- `pools/<run>/<source>_{train,eval}.jsonl` — 4-frame window pools, built by
  `recipes/temporal_ssl/merge_pools.py` (`main` = the 60K mix; other runs e.g. extra single-source
  sets); `pools/<run>/summary.json` has the run config, per-source counts and Δt mix.
  Only `summary.json` is committed: the jsonl files are gitignored (absolute paths to frames
  outside the repo); rebuild them with `recipes/temporal_ssl/merge_pools.py --stage all [--name <run>]`
  (deterministic, seed 42) and compare against the committed `summary.json`.
- `settings/<run>/T{0..4}_{train,eval}.jsonl` — trainer rows derived from the pool by
  `recipes/temporal_ssl/derive_settings.py` (gitignored like the pools; `settings/<run>/summary.json`
  holds counts, balance and a sha256 prefix per file). Row fields:

| Field | Meaning |
|---|---|
| `id` | `<variant>__<pool window id>` |
| `setting`, `variant` | `T0`..`T4`; variant `T4A` / `T4B` for T4 rows |
| `task_type` | `imgen` (multimodal imgen reader in the trainer) |
| `source`, `window_id`, `clip_id`, `split`, `caption`, `gap_s`, `stalled` | copied from the pool window |
| `image` | the 3 input frame paths, in prompt order (F-order or A, B, C) |
| `target_image`, `cond_image` | generated frame; VAE condition = nearest shown frame (tie → earlier) |
| `layout` | ordered: `shown` F-names; shuffled: `labels` {A/B/C: F-name}; plus `target`, `cond` |
| `answer` | ground truth: `gap`, `order`, `missing` as applicable; `order_equiv` = label pairs whose order is indistinguishable (identical adjacent frames) |
| `conversations` | human prompt (`common/prompts.py`) and gpt answer (`{json}\n<img>` or `<img>`) |

Pool row (one per window; `frames`, `frame_idx`, `timestamps` are chronological and aligned with
`order` = `["F0","F1","F2","F3"]`):

| Field | Meaning |
|---|---|
| `id` | `<source>__<clip_id>__dt<Δt>__k<k>` (the same window has the same id in every run) |
| `source`, `clip_id`, `split` | split is `train` / `eval`, disjoint by clip |
| `caption`, `caption_source` | the source's own caption, verbatim |
| `gap_s` | Δt in seconds (0.5 / 1.0 / 2.0) |
| `gap_frames` | real frame spacing (3 ints; rounding and the last-frame clamp make it differ by ≤ 2 frames from Δt·fps) |
| `order`, `frames`, `frame_idx`, `timestamps` | F0..F3: jpg paths, frame indices, exact seconds |
| `window` | `k` (window index at this Δt), `start_s`, `end_s` |
| `clip` | `video`, `fps`, `num_frames`, `duration`, `category` |
| `motion` | `change_vs_f0` [F0-F1, F0-F2, F0-F3], `change_adjacent` [F0-F1, F1-F2, F2-F3] (fraction of changed pixels), `stalled` = an adjacent pair < 0.1% (ambiguous for ORDER / MISSING) |
| `source_meta` | source-specific fields (phenomenon, physical params, template, task, ...) |

Frame images live outside the repo at `/scratch/network/ssd/junlin/v2_frames/<source>/<clip_id>/`.
See `docs/v2.md`.
