# `data/common/v2/` — v2 shared library

Code shared by every v2 source and the v2 recipe (planned modules):

- `paths.py` — raw / frame / dataset roots (`/scratch/network/ssd/junlin/raw/`, `.../v2_frames/`);
  v2 code reads paths from here instead of hardcoding them.
- `window_schema.py` — canonical window-pool row: source, clip_id, split, Δt, F0..F3 paths, caption.
- `window_sampler.py` — windows from (fps, duration, GAP set, stride); skips infeasible gaps.
- `extract_frames.py` — timestamp-based frame extraction, deduplicated across windows.
- `prompts.py` — T0-T4 prompt templates and the GAP / ORDER / MISSING answer grammar (the v2
  eval parser uses the same file).
- `caption_vlm.py` — clip-level VLM captioning (generalized from v1's
  `recipes/s4_progression/label_with_vlm.py`).

v1 shared code (`../*.py`) is not modified. See `docs/v2.md`.
