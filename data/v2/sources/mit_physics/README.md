# `data/v2/sources/mit_physics/` — Moments in Time, physics classes (v2 source)

- **Data:** Moments in Time v2 raw release (`Moments_in_Time_Raw_v2.zip`), fetched from the
  ungated HF mirror [`Pai3dot14/Moments_in_Time_Raw_v2_hf`](https://huggingface.co/datasets/Pai3dot14/Moments_in_Time_Raw_v2_hf)
  (7 byte-parts, 294.7 GB, zip64).
- **License:** the official route is the request form on <http://moments.csail.mit.edu/>. The
  bundled `license.txt` (copied to `raw/mit_physics/metadata/`) allows **non-commercial research
  and education only** and forbids redistribution. The HF mirror itself is a third-party
  redistribution; we use it for technical access (decision 2026-09-24).
- **What it is:** 305 action classes, 727,305 train + 30,500 val clips (100 per class), all
  **3.0 s**, 480–640 px wide (some 720p), 24–30 fps, ~0.39 MB each. YouTube / Getty / stock
  footage, so camera motion, cuts, watermarks and label noise are common. **No captions** —
  only the class label.
- **Raw location:** `/scratch/network/ssd/junlin/raw/mit_physics/`.

## Filtering & sampling summary

**Category selection + CV QC + VLM QC.**

1. Category filter: 24 hand-picked object-physics classes (falling, bouncing, fluids, breaking, burning, …); human-action classes excluded. Training split only.
2. Pool: every clip of those classes (52,390); each source video (filename stem) used once across classes.
3. CV QC (`common/clip_qc.py`): 0 scene cuts, camera motion ≤ 0.05, `obj_motion_max` ≥ 0.02, duration ≥ 2.9 s → 16,254.
4. VLM QC (`common/vlm_check.py`, Qwen3-VL-8B): require `shows_label` + `real_footage`; prefer no `heavy_text` → 12,040 pass.
5. Final: `common/trim_selection.py --group-key class` → `selection_5k.jsonl` = 5,000 (1.9 GB), ~212 per class (rocking 119).

Details below.

## Why range reads instead of downloading the zip

Every zip member is compressed on its own, and the parts' signed CDN URLs accept HTTP Range.
`download_mit_physics.py` reads the zip64 central directory (125 MB, 758,492 entries) into
`zip_index.jsonl`, then fetches each selected member's compressed bytes, inflates them and
checks the CRC.

## Selection

24 object-physics classes (`CLASSES` in the script): falling, dropping, bouncing, rolling,
sliding, breaking, cracking, crushing, erupting, spinning, swinging, rocking, floating,
splashing, spilling, pouring, flowing, dripping, overflowing, draining, leaking, bubbling,
boiling, burning. Human-action classes (kicking, throwing, landing, …) are left out. Only the
training split is used; the validation split is kept as a candidate v2 eval pool. A source
video (filename stem) is used at most once across classes.

1. `selection_15k.jsonl`: first pass, 600 random clips per class (14,400).
   `clip_qc.py` on those showed only ~4.5K pass camera motion ≤ 0.05, and contact sheets showed
   heavy content noise (talking heads labelled "breaking", motion graphics, time-lapse, watermarks),
   so the pool was widened.
2. `selection_pool.jsonl`: **every** clip of the 24 classes (52,390, 19.9 GB).
3. `../../common/clip_qc.py` → `qc.jsonl`; hard filters: 0 scene cuts, camera motion ≤ 0.05
   frame-widths/s, `obj_motion_max` ≥ 0.02.
4. `../../common/vlm_check.py` (Qwen3-VL-8B-Instruct, 4 frames + class name) → `vlm.jsonl`;
   require `shows_label` and `real_footage`; clips without `heavy_text` are preferred.
   It also writes a candidate `caption` (the MiT caption policy is still open).
5. `../../common/trim_selection.py` → **`selection_5k.jsonl`**, water-filled over the 24 classes
   (seed 42). The chain is `raw/mit_physics/run_pool_pipeline.sh`.

## Caveats for building v2 windows

- **3 s clips:** 4 frames span 3Δt, so Δt ≤ ~0.95 s (Δt = 1.0 s needs ≥ 91 frames at 30 fps).
- **Caption:** none in the raw data. Options: class-name template, the VLM candidate caption
  from `vlm.jsonl`, or Spoken Moments in Time (separate request form).
- Getty/videoblocks watermarks are common even among `heavy_text = false` clips.
