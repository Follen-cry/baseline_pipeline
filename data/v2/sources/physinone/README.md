# `data/v2/sources/physinone/` — PhysInOne (v2 source)

> **v2 pool (2026-09-26):** all 5,000 scenes are candidates; frames read from `rgb/` at 30 fps.
> Caption = `caption.txt`, verbatim (outcome descriptions are wanted).

- **Source:** [vLAR-group/PhysInOne](https://github.com/vLAR-group/PhysInOne) (CVPR 2026),
  CC BY-NC-SA 4.0. Synthetic UE-rendered scenes covering 71 physical phenomena.
- **Hosting:** 16 HF dataset repos `PhysInOneP01/PhysInOneP01` … `P16` (~111 TB). One zip per
  scene (~0.25–0.8 GB): 13 cameras (`CineCamera_0..10,12` static + `CineCamera_Moving`) ×
  {rgb, depth, seg} jpg/png frames, plus `caption.txt`, the trajectory json (`frame_rate: 30`),
  per-camera blender json, and point clouds.
- **Case index:** `repo_assignment.txt` + `repo_map.json` from the GitHub repo, copied to
  `/scratch/network/ssd/junlin/raw/physinone/_meta/` (commit recorded next to them). All
  122,988 indexed scenes are Train: 5,554 single-, 20,860 double-, 96,574 triple-physics.

## Filtering & sampling summary

**Category-stratified (71 physical phenomena).**

1. Pool: official Train split, SinglePhysics + DoublePhysics scenes only (TriplePhysics skipped).
2. Sampling: water-fill over phenomena (least-covered phenomenon served first; prefer already-downloaded, then SinglePhysics, then random), seed 42 → 4,350 single + 650 double; 7–95 scenes per phenomenon.
3. Camera: one static camera per scene, drawn at random (deterministic fallback if missing). No camera-motion filter needed (fixed cameras).
4. No pixel-level QC, no caption filter.
5. Final: `selection_5k.jsonl` = 5,000 scenes (~130 GB of rgb frames).

Details below.

## Why range reads instead of the official downloader

The official `download_selected.py` fetches whole scene zips, which would be ~4.5 TB for 5K
scenes. `download_physinone.py` opens each zip over HTTP Range instead: it reads the central
directory (~0.7 MB, ~4 requests), then fetches **one static camera's rgb frames**, which are
stored contiguously (one request, ~8–27 MB), plus `caption.txt` and the small json. Each
scene costs one HF `resolve` call (throttled to 2/s, 5-min pause on 429); the byte ranges go
straight to the CDN.

## Selection (seed 42) → `selection_5k.jsonl`

Target is **5K scenes** (unlike PhysicTran38K / PhyCo-Sim, no 10K oversample: a synthetic source
with fixed cameras needs little post-filtering).

1. Water-fill over Single+DoublePhysics Train scenes: each step serves the least-covered
   phenomenon, preferring (a) scenes already downloaded, (b) SinglePhysics, (c) random.
2. Result: **4,350 single + 650 double**; per-phenomenon coverage min / median / max =
   7 / 85 / 95 over 70 phenomena (the min-7 phenomenon only has 7 scenes in the whole
   single+double pool).
3. TriplePhysics skipped.
4. Camera: one static camera per scene drawn at random. If a scene lacks it, a deterministic
   fallback picks another static camera the scene has; the camera actually used is written to
   the scene's `DONE` file.

History: a first run targeted 10K (all singles + 4,446 doubles) and finished 707 scenes before
the target was cut to 5K. That list is kept as `selection_10k_superseded.jsonl`; 695 of those
707 are reused by the 5K selection, and the other 12 stay on disk unused.

Output: `raw/physinone/<case_id>/{rgb/NNNN.jpg, caption.txt, recorder_stats.json,
<case>_trajectory.json, blender_<cam>.json, static_camera_list.txt, DONE}`.

## Status (2026-09-24): 5,000 / 5,000 downloaded

130 GB on disk (134 GB of rgb jpgs by byte count). Every scene has `caption.txt`; 200 sampled
scenes read back fine. Frames per scene: 150 (2,156), 90 (732), 120 (479), 300 (475),
180 (248), i.e. 3–10 s at 30 fps. The planned camera was absent in 516 scenes, which used the
deterministic fallback. 12 scenes from the superseded 10K run remain on disk outside the 5K.

Bug fixed during the run: the static-camera filter first excluded any path containing
"Moving", which also matched the 370 case_ids named e.g. `MovingHitsFixed…` or
`LiquidCarryMovingObj…`. It now compares the camera folder name to `CineCamera_Moving`, and a
second pass fetched those 370.

## Caveats for building v2 windows

- **Frame count varies per scene** (90–300 frames, i.e. 3–10 s at 30 fps). Read it from `DONE`
  or count jpgs. Frames are 1120×1120 jpgs; there's no mp4, which v2 doesn't need since
  windows are sampled from frames.
- **Caption:** `caption.txt` is a rich per-scene description, but it narrates the outcome
  ("…the second ball undergoes free fall"). Same leak concern as PhyCo / PhysicTran.
- Frames are rendered from fixed cameras, so no camera-motion filter is needed.
