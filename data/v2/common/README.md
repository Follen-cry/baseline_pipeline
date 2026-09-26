# `data/v2/common/` — v2 shared library

Code shared by every v2 source and the v2 recipe:

- `paths.py` — raw / frame roots (`/scratch/network/ssd/junlin/raw/`, `.../v2_frames/`);
  v2 code reads paths from here instead of hardcoding them.
- `window_sampler.py` — **the global window rule** (all sources), from clip length only
  (num_frames, fps), no content signal:
  - keep clips with T = num_frames / fps ≤ 20 s (+0.25 s tolerance: nominal 20 s re-encodes
    measure 20.05–20.10 s);
  - Δt ∈ {0.5, 1.0, 2.0} s, span 3Δt; per scale N = floor(T / 3Δt) windows at t_k = k·3Δt;
  - frame index = round(t·fps); an index equal to num_frames (window ends exactly at T) uses
    the last frame;
  - ≤ 10 windows per clip: over the cap, water-fill one slot per scale per round, larger Δt
    first; within a scale keep evenly spaced windows (quota 1 = middle one).
  `python window_sampler.py --num-frames 600 --fps 30` prints the windows for a clip length.
- `extract_frames.py` — applies the rule to a clip list (`source, clip_id, video`, + `fps` when
  `video` is a frame directory; other fields are copied to every window row) and writes frames
  to `v2_frames/<source>/<clip_id>/<idx:05d>.jpg`: two sequential OpenCV passes (count, then
  read), a frame shared by windows written once, native resolution capped to fit 1024×1024
  (never upscaled), JPEG q95. Outputs one row per
  window (`window_id, dt, k, start_s, frame_idx, timestamps, frames, fps, num_frames, duration`)
  and a resumable per-clip status log `<out>.clips.jsonl`.
  Optional per-clip `windows: [[dt, k], ...]` keeps only a planned subset, and
  `num_frames_expected` guards against the plan and the decode disagreeing (`clip_length()` is the
  shared frame counter).
- `window_motion.py` — static-window filter: per-window pixel-change scores (256 px gray, blur,
  |Δ| > 25) vs F0 and between adjacent frames; `is_static` = max change vs F0 < 0.1%.
- `window_schema.py` (planned) — canonical window-pool row: source, clip_id, split, Δt, F0..F3
  paths, caption.
- `prompts.py` (planned) — T0-T4 prompt templates and the GAP / ORDER / MISSING answer grammar (the v2
  eval parser uses the same file).

Captions come with each source's raw data, so there is no captioning stage. See `docs/v2.md`.
