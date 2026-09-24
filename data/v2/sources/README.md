# `data/v2/sources/` — v2 data sources

One dir per source used by v2, whether it is new or also used by v1 (e.g. EPIC-Kitchens).
v1's builders in `data/v1/sources/` are not reused, because v2 re-samples 4-frame windows by
timestamp at variable gaps. Each `<name>/` holds:

- `README.md` — license, download route, raw-data location
  (`/scratch/network/ssd/junlin/raw/<name>/`), caption field, split unit + seed, feasible gaps
- `download_<name>.py` — if the source needs downloading
- `build_windows.py` — raw video → window-pool rows (see `../common/window_schema.py`),
  frames to `/scratch/network/ssd/junlin/v2_frames/<name>/`
