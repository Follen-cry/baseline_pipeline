# `data/sources/worldprediction/`

Train/eval data derived from the WorldPrediction benchmark (`facebookresearch/WorldPrediction`,
vendored at `../../../eval/suites/worldprediction/`), split so the benchmark itself stays a clean
held-out eval set while the rest of its data sources become training data for the same task.

## Split
- **Eval = COIN, in full.** COIN is the WorldPrediction eval benchmark's primary source; none of
  it is used for training.
- **Train = CrossTask + EPIC-KITCHENS-100 + IKEAASM.** `EgoExo4D` is excluded (gated dataset, no
  local video root — see `eval/suites/worldprediction/data/video_roots.py`).
- Split is **source-level** (COIN vs. non-COIN), not row- or video-level, so there is no
  train/eval leakage risk from sharing source videos across the split.
- **Known overlap to be aware of**: `data/sources/epic_kitchens/` already trains InternVL-U's
  S0-S3 SSL stages on the same 116 EPIC-KITCHENS-100 videos referenced here (same physical files
  at `/scratch/local/ssd/junlin/data/worldprediction/epic-kitchen/`). This recipe's EPIC-100 rows
  are a *different task* (WM 4-way MCQ vs. epic_kitchens' next-frame SSL) built from the *same*
  raw videos — not a train/eval leak within this recipe (this recipe has no EPIC-100 eval rows),
  but worth knowing if training on both recipes at once.

## Task
Only **WorldPrediction-WM (World Modeling)** is built so far: single-step "which candidate action
caused this world-state change" 4-way MCQ. **WorldPrediction-PP (Procedural Planning)**, the
multi-step plan-ordering task, uses the same underlying action-clip/4-frame mechanism and is a
natural follow-up but is not built yet.

## Action-clip frame extraction
Every candidate action's video segment `[start_time, end_time]` is turned into exactly 4 frames
via `eval/suites/worldprediction/data/load_frames.py::collect_frames(model_max_frames=4,
desired_fps=5.0)` — the **same function and same parameters** (from
`configs/vlm/internvlu/InternVL-U-4f.json`) the InternVL-U WorldPrediction adapter calls at
inference time (`model/vlms/internvlu.py::InternVLU._materialize_videos`). No new sampling logic
was written; offline (train) and on-the-fly (eval) frame extraction are provably identical.

`collect_frames` only guarantees exactly 4 frames when clip duration >= `4/5.0 = 0.8s` (shorter
clips fall back to a variable-length fps-based path). Any sample with a candidate clip shorter
than 0.8s is dropped entirely (both train and eval), so "always exactly 4 frames per action clip"
is a hard invariant of the output data, not a best-effort.

## Prompt format
The MCQ prompt (initial/final world-state images, 4 lettered candidate actions, JSON answer
format `{"action": "X"}`) is a near-verbatim port of
`eval/suites/worldprediction/model/vlm.py::VLM.select_action`'s message construction, flattened
to text with `eval/suites/worldprediction/model/vlms/internvlu.py::InternVLU._build_prompt`'s
exact `Image{i}: <image>` / `Frame{i}: <image>` numbering and `"User: "` prefix convention. This
is a **port, not a shared import** (the original logic is inlined inside the model-calling method,
not factored into a standalone function) — see the MAINTENANCE note in
`processing/build_worldprediction_wm.py` if `vlm.py`'s prompt template ever changes.

`max_dynamic_patch: 1` in the meta jsons is intentional, not a placeholder: WM samples already
carry up to 18 images (2 state frames + 4 candidates x 4 frames), and InternVL-U's own inference
config (`InternVL-U-4f.json`) does not apply InternVL's classic per-image dynamic-tiling to video
frames — matching that at training time avoids both a token-budget blowup and an inference/train
resolution mismatch.

## Processing
`processing/build_worldprediction_wm.py` — single script, no stages. Reads
`WorldPrediction-WM.json`, extracts/saves frames to `../../datasets/worldprediction/frames/`,
writes `../../datasets/worldprediction/{wm_train,wm_eval}.jsonl`. Must run in the `internvlu`
conda env (has `av`/PyAV; the `WorldPrediction` conda env's `requirements.txt` lists `av==13.1.0`
but it isn't actually installed there) and on `torrnode11` (all 4 sources' raw videos are
node-pinned local-SSD paths, not network-visible).

## Row counts (2026-09-11 run)
| source | kept | skipped | split |
|---|---|---|---|
| COIN | 171 / 236 | 65 (48 missing state-frame video, 17 missing candidate video) | eval |
| CrossTask | 81 / 109 | 28 (missing candidate video — CrossTask's local video pool is incomplete) | train |
| EPIC-KITCHENS-100 | 193 / 193 | 0 | train |
| IKEAASM | 159 / 159 | 0 | train |

Total: **433 train rows, 171 eval rows**. Frame PNGs: 11G at
`../../datasets/worldprediction/frames/`. Skip reasons breakdown:
`{"missing_state_frame": 48, "missing_video": 37, "incomplete_frames": 8}` (the last is candidate
clips that decoded fewer than 4 frames despite passing the >=0.8s duration filter, e.g. one
`P30_08.MP4` EPIC clip with a corrupt packet).

## Not done yet
- **PP (Procedural Planning)** task — same 4-frame action-clip mechanism, different (multi-step)
  prompt template from `model/vlm.py::VLM.select_plan`.
- Missing COIN/CrossTask videos (93 candidate/state videos referenced but not present in the
  locally staged video pools) are not re-fetched — would need re-running each source's own
  download step to close the gap and recover the ~93 skipped samples.
- Training config (freeze pattern, LoRA rank, loss weights) is not yet decided/wired up — see the
  main conversation for the discriminative-MCQ (S0-S3-style, `lm_loss` only, no generation-decoder
  branch) recommendation agreed with the user.
