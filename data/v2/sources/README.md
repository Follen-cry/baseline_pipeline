# `data/v2/sources/` — v2 data sources

One dir per source used by v2, whether it is new or also used by v1 (e.g. EPIC-Kitchens).
v1's builders in `data/v1/sources/` are not reused, because v2 re-samples 4-frame windows by
timestamp at variable gaps. Each `<name>/` holds:

- `README.md` — license, download route, raw-data location
  (`/scratch/network/ssd/junlin/raw/<name>/`), caption field, split unit + seed, feasible gaps
- `download_<name>.py` — if the source needs downloading
- `build_windows.py` — raw video → window-pool rows (see `../common/window_schema.py`),
  frames to `/scratch/network/ssd/junlin/v2_frames/<name>/`

## Source overview (2026-09-25)

Full rules: the "Filtering & sampling summary" section at the top of each source's README.
Raw data under `/scratch/network/ssd/junlin/raw/<name>/` (VBVR videos stay in `vbvr_next_frame/raw`).

| Source | Kind | Filtering / sampling | Final list | Size (list / on disk) |
|---|---|---|---|---|
| `physinone` | synthetic (UE) | category-stratified over 71 phenomena; 1 static camera/scene | `selection_5k` 5,000 scenes | ~130 GB / 130 GB |
| `phyco_sim` | synthetic (sim) | motion ≥ 18 f; per-scene quota × physical-parameter bins | `selection_10k` (→ 5K todo) | 0.8 / 69 GB |
| `baai_physics` | real, fixed camera | caption-keyword groups + ≤ 20/query folder; CV QC | `selection_5k` 5,000 | 23.7 / 51 GB |
| `vbvr` | synthetic (generators) | whole pool minus eval-reserved (S3 eval + target_pred eval leaks) and duplicates | `split=train` 134,095 (subsample todo) | 39.8 / 42.9 GB |
| `mit_physics` | real, YouTube/stock | 24 object-physics classes; CV QC + VLM QC; class-balanced | `selection_5k` 5,000 | 1.9 / 20 GB |
| `panda70m` | real, YouTube | metadata gates + caption regex + LLM caption judge; category round-robin | `selection_5k` 2,009 (blocked) | 1.6 / 1.7 GB |
| `physictran38k` | model-generated | 46 transition types; authors' final_filter list + water-fill | `selection_10k` (→ 5K todo) | 1.5 / 1.5 GB |
| `ssv2` | real, handheld | 24 physics templates, water-filled | `selection_10k` (→ 5K todo) | 0.86 / 0.87 GB |

Shared QC tools in `../common/`: `clip_qc.py` (cuts, camera motion, object motion),
`vlm_check.py` (Qwen3-VL label/real-footage/text check), `trim_selection.py` (thresholds + water-fill to N).
