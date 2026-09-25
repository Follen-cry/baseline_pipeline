# `data/v2/sources/panda70m/` — Panda-70M physics clips (v2 source)

- **Source:** [snap-research/Panda-70M](https://github.com/snap-research/Panda-70M) (CVPR 2024),
  YouTube clips with auto-generated captions. Clips are fetched from YouTube; the dataset
  only ships metadata.
- **Pool:** `panda70m_training_2m.csv` (800K videos / 2.4M clips), which is a subset of
  the official 10M quality tier (`matching_score > 0.43`, ≤ 3 clips per video). The CSV lives at
  `ssl_mllm/data/datasets/panda70m/metadata/` (see the README there for the schema).
- **Raw location:** `/scratch/network/ssd/junlin/raw/panda70m/`
- **Caption field:** Panda-70M's per-clip `caption` (one sentence describing the whole clip).
- **Relation to v1:** none. v1's two Panda-70M pools (3DSRBench-style and EPIC-kitchen-style,
  both from the 10M split) had their raw mp4s deleted on 2026-09-24. This is a fresh selection.

## Filtering & sampling summary

**Caption filter (regex + LLM judge).**

1. Pool: `panda70m_training_2m.csv` (2.4M clips).
2. Metadata gates: `desirable_filtering` + longest single shot ≥ 4 s (tier A) / 3–4 s or tiny-camera-movement (tier B fallback).
3. Caption regex: physics-word `PHYS_RE` hit and `HARD_REJECT_RE` (games, screens, animation, interviews) miss → 100,584 candidates.
4. LLM caption judge (Qwen3-VL-30B-A3B-FP8, text only): physical process AND motion-rich AND NOT manipulation, confidence ≥ 4 → 9,188 passed.
5. Sampling: tier A before tier B; within a tier, round-robin over 12 judge categories (seed 42) until 5,000 downloads. Download = longest single shot (≤ 20 s), re-encoded, decode-verified.
6. Optional visual VLM score (`visual`, 61% pass) — soft only. No CV QC yet.
7. Final: `selection_5k.jsonl` = **2,009 so far** (YouTube bot-check blocked the download).

Details below.

## Selection goal

Motion-rich clips whose caption describes a **physical process** (falling, collisions, fluids,
fire/smoke/explosions, breaking/melting, projectiles, spinning, wind-driven motion, and so on),
**not manipulation**. Rules agreed with the user (2026-09-24):

- Manipulation captions ("a person is cutting / stirring / pouring / boiling ...", including
  passive phrasings like "noodles are being poured") are out, even if physics is involved. A
  person may *trigger* the process when the caption is about the resulting motion ("a rock is
  thrown into the lake and splashes").
- Also out: sports and human locomotion, animal behavior, vehicles that are only driving, flying
  or sailing, camera-only motion (aerial or drone shots), screens, games, CGI and fantasy.
- In: vehicle crashes, rocket launches, boats tossed by waves, demolitions, and similar.

## Pipeline

| Step | Script | Output (under the raw dir) |
|---|---|---|
| 1. Prefilter: mechanical gates + physics-word regex | `filter_panda70m.py --stage prefilter [--tier B]` | `_filter/stage1_candidates{,_tierB}.jsonl` |
| 2. LLM caption judge | `filter_panda70m.py --stage judge [--tier B]` | `_filter/stage2_judged{,_tierB}.jsonl` |
| 3. Export the passed pool | `filter_panda70m.py --stage export` | `pool_passed.jsonl` |
| 4. Download + cut + verify, stop at 5,000 | `download_panda70m.py --target 5000` | `videos/<clip_key>.mp4`, `download_log.jsonl`, `selection_5k.jsonl` |

`run_pipeline.sh` is the driver that ran all four steps, overlapping downloads with judging.

### 1. Prefilter

- **Tier A:** `desirable_filtering == "desirable"` and a longest single shot (Panda's TransNetV2
  `shot_boundary_detection`) of at least 4 s.
- **Tier B (fallback):** `desirable` clips whose longest shot is 3–4 s, plus
  `2_tiny_camera_movement` clips (near-static camera, which is fine for physics) with a shot of
  at least 3 s. Tier B is judged and downloaded only after tier A. It exists because tier A alone
  yields only ~7K passed captions, which is too few to be sure of reaching 5,000 downloads.
- **Download segment:** the clip's longest single shot, capped at 20 s, so a v2 4-frame window
  can never cross a cut.
- **Caption:** must hit `PHYS_RE`, a loose list of physics words, and miss `HARD_REJECT_RE`
  (games, screens, animation, interviews).

**Funnel over the 2.4M clips:**

| Tier | Passed shot + desirability gates | After hard-reject | Candidates |
|---|---|---|---|
| A | 1.20M | 59,907 dropped | **80,449** |
| B | 265K | 14,802 dropped | **20,135** |

**Regex recall check:** the LLM passed 1 of 400 randomly sampled desirable captions that miss
`PHYS_RE`, so the regex loses almost nothing.

### 2. Judge

- **Model:** `Qwen3-VL-30B-A3B-Instruct-FP8`, served by vLLM on torrnode11 (`:8010`, served name
  `qwen3-vl-30b-fp8`). Text only, temperature 0.
- **Prompt:** `JUDGE_SYSTEM_PROMPT` in `filter_panda70m.py`. It returns JSON with
  `physical_process`, `motion_rich`, `manipulation_focused`, `category` (12 classes) and
  `confidence` (1–5).
- **Pass rule:** `physical_process AND motion_rich AND NOT manipulation_focused AND confidence >= 4`.
- **Order:** candidates are visited in a seeded random order (seed 42). The whole tier A and
  tier B pools were judged, at about 5.6 captions/s.
- **Prompt iteration:**
  - The first version let through animals swimming, surfers, kids on swings, "a person is
    boiling ...", and cars trailing smoke. Version 2 requires the physical process itself to be
    the subject of the caption.
  - Version 3 adds: people using tools or machines that throw sparks, passive-voice
    manipulation, and fantasy/CGI.
  - On a 400-caption pilot, about 24 of the 31 passes were clean.

### 3–4. Download

- **Fetch:** yt-dlp downloads the full video at ≤ 360p (itag 18; `android` client, then
  `tv_simply`). `--download-sections` makes ffmpeg segfault (exit −11) on this cluster, so the
  cut is done locally.
- **Cut:** the segment is cut and **re-encoded** (libx264, CRF 20, no audio). A stream-copy cut
  starts on a non-keyframe and leaves frames that won't decode; that is how v1's panda70m_epic
  clips ended up with only about half their frames decodable.
- **Verify:** each output is decoded end to end with OpenCV. It counts only if
  decoded ≥ 95% of `fps × segment_dur` and the clip is at least 3 s long.
- **Order:** all of tier A first, then tier B. Within a tier, categories are interleaved
  round-robin (seed 42), so stopping at 5,000 keeps the category mix as even as the pool allows.
- **Rate limiting:** 4 workers with random jitter. A circuit breaker pauses 15 min when ≥ 40% of
  the last 30 attempts look like a YouTube block (429 / bot check), and exits after 4 pauses in
  a row. Rerun to resume.
- **Pilot (30 clips):** 28 succeeded and 2 failed (private videos). About 1.3 MB and 19 s per
  clip, roughly 830 attempts/hour, with no rate limiting.

## Output: `selection_5k.jsonl`

One row per downloaded clip, in download order:

`id` (= `<videoID>_<clip_idx>`), `tier`, `desirable_filtering`, `video_id`, `url`,
`panda_timestamp` (the original Panda clip), `segment` (the part actually downloaded, as absolute
video time), `caption`, `matching_score`, `category`, `judge` (full judge JSON), `fps`,
`num_frames` (decoded), `duration_s`, `width`, `height`, `video` (absolute mp4 path).

## Status (2026-09-25): judging done, download paused at 1,204 / 5,000 (YouTube bot-check)

- **Judge:** complete for both tiers. Tier A: 6,787 / 80,449 passed (8.4%). Tier B: 2,401 /
  20,135 passed (11.9%). **Total pool: 9,188** (`pool_passed.jsonl`). By category:
  fire_smoke_explosion 3,114, fluid_flow 2,604, collision_impact 743, projectile 620,
  wind_driven 559, splash_pour 498, deformation_breaking 288, rotation_oscillation 280,
  falling 242, rolling_sliding 147, other_physics 93.
- **Download:**
  - **Result:** 1,204 clips succeeded, all tier A, 943 MB in `videos/`. Failures were 51
    unavailable, 43 errors and 42 cut failures. Clips are mostly 640×360 at 30 fps, with a
    median length of 6.6 s.
  - **Block:** YouTube's bot-check ("Sign in to confirm you're not a bot") began intermittently
    around attempt 390 and became a full block around 1,200 successes. The circuit breaker
    stopped the run on 2026-09-25 00:20.
  - **Retry routes:** every yt-dlp client (android, tv_simply, ios, web_safari, mweb, tv) was
    blocked, on torrnode11 and on torrnode7. torrnode7 has a different egress IP
    (129.67.94.121 vs 129.67.94.83, shared by torrnode8/12).
  - **To resume:** `python download_panda70m.py --target 5000 --workers 2 --cookies <cookies.txt>`.
    Use a cookies.txt exported from a logged-in YouTube browser session, or rerun without
    cookies once the block lifts. The download is resumable; blocked attempts are retried.
- **Visual check** (`verify_panda70m.py`, stored as `visual` in `selection_5k.jsonl`): 735 of
  1,204 clips pass (61%), and 1 errored. Use it as a soft score, since it agrees with a manual
  label on only 17/24 clips.

## Notes for building v2 windows

- **Caption leak:** captions describe the whole clip, sometimes including the outcome ("... falls
  and shatters"). Apply the same leak review as the other v2 sources.
- **Duration:** tier A is at least 4 s, so Δt ≤ 1.0 s is always feasible. Tier B can be as short as
  3 s. Δt = 2.0 s needs a 6 s segment.
- **Split unit:** Panda video (`video_id`). Up to 3 clips share one YouTube video, so split by
  video, not by clip.
