# `data/v2/sources/baai_physics/` — BAAI Physics-aware videos (v2 source)

- **Repo:** [`BAAI-DataCube/Physics-aware-videos`](https://huggingface.co/datasets/BAAI-DataCube/Physics-aware-videos)
  (org is **BAAI-DataCube**, not `BAAI`), CC-BY-NC-4.0, ungated.
- **What it is:** **real-world**, **fixed-camera** clips of physical processes (falling, rolling,
  collision, shattering, fluids, smoke/dust), retrieved from BAAI Datacube by keyword queries and
  filtered by an MLLM. Clips are 5–20 s, often 1080p at 30/60 fps (some 240p).
- **Layout:** 83,223 clips in 961 query folders `videos/aNNNNN/NNNNN.mp4` (the query text is not
  released). `metadata/dataset.jsonl` = `{video, caption}`, captions by Qwen-VL-72B (~26 words).
  The videos are one **uncompressed** `videos.tar` split into 11 byte-parts
  (10 × 42.9 GB + 16.9 GB = **446 GB**). Tar order ≠ jsonl order.
- **Raw location:** `/scratch/network/ssd/junlin/raw/baai_physics/`.

## Filtering & sampling summary

**Caption-keyword filter + CV QC.**

1. Pool: all 83,223 clips (authors already kept fixed-camera, 5–20 s clips via MLLM filtering).
2. Caption filter: regex keyword groups (topple, bounce, slide, swing/spin, break, collide, fall, smoke/particle, roll, fluid); clips with no keyword dropped → 57,963.
3. Sampling (10K oversample): round-robin water-fill over keyword groups (each clip counted under its rarest group), random within a group, ≤ 20 clips per query folder, members > 150 MB skipped, seed 42.
4. CV QC (`common/clip_qc.py`): 0 scene cuts, camera motion ≤ 0.08 frame-widths/s, `obj_motion_max` ≥ 0.02, duration ≥ 5 s → 5,106 pass.
5. Final: `common/trim_selection.py --group-key kw_group` → `selection_5k.jsonl` = 5,000 (23.7 GB). No VLM check.

Details below.

## Why range reads instead of downloading the tar

The parts' signed CDN URLs (one HF `resolve` per part, valid ~1 year) accept HTTP Range, and the
tar is uncompressed, so `download_baai_physics.py` walks the 512-byte tar headers remotely to
build `tar_index.jsonl` (path, header offset, size) and then fetches only the selected members.
The tar is cut into 64 segments; each segment's first header is found by scanning for the
ustar magic (checksum-verified), and the segments are walked in parallel (~0.6 s per serial
header read). Full download would need 446 GB of the ~800 GB free on `/scratch/network/ssd`.

## Selection (`download_baai_physics.py`, seed 42) → `selection_10k.jsonl`

Target is a **10K oversample**, later trimmed to 5K for v2 (same as PhysicTran38K / PhyCo-Sim / SSv2).

1. Caption keyword groups (`KW_GROUPS`, rarest first): topple, bounce, slide, swing_spin, break,
   collide, fall, smoke_particle, roll, fluid. A clip is counted under the first group it matches;
   clips with no physics keyword are dropped (57,963 of 83,223 match).
2. Water-fill 10K across the groups (round-robin), random within a group, **≤ 20 clips per query
   folder** (folders range from 1 to 2,242 clips), members > 150 MB skipped.

Manifest fields: `id, source, query_folder, kw_group, kw_groups, caption, tar_path, tar_offset,
bytes, video`. Videos: `raw/baai_physics/videos/<folder>/<file>.mp4`.

## Status (2026-09-24)

1. **Index:** `tar_index.jsonl`, all 83,223 members (64 parallel segment walks, ~27 min); every
   jsonl caption row matches a tar member.
2. **10K downloaded** (`selection_10k.jsonl`, 51.3 GB, ~77 MB/s, 0 missing / size mismatches).
3. **Trimmed to 5K** → **`selection_5k.jsonl`** with `../../common/clip_qc.py` +
   `../../common/trim_selection.py --group-key kw_group --min-duration 5 --max-cam 0.08 --min-obj 0.02`
   (0 scene cuts). Of 10K: 5,106 pass (camera 3,944, cuts 843, static 106, short 1).
   Per group: roll 781, break 761, fluid 612, swing_spin 602, smoke_particle 584, fall 529,
   collide 501, bounce 261, slide 193, topple 176.

Despite "fixed camera only" in the dataset card, median `cam_motion` is 0.06 frame-widths/s.
Contact sheets: ≤ 0.02 is truly locked-off, 0.05–0.08 is mild handheld jitter with good
physics content, ≥ 0.1 is real pans / handheld; hence the 0.08 cut. Content quality is much
higher than MiT, so no VLM check was needed.

## Caveats for building v2 windows

- Long enough for every planned Δt (5–20 s), including 2.0 s — the only new source where that holds.
- **Caption:** ~9% of captions narrate the outcome ("…causing it to shatter"); same leak concern
  as the other sources.
- Keyword matches are noisy (e.g. "water" matches boating scenes); the authors already filtered
  camera motion, but a VLM / motion spot-check before trimming to 5K is still advisable.
