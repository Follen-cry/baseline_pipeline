"""Sample duration-adaptive 3-frame-in -> 1-frame-out training windows from
the raw VBVR ground_truth.mp4 videos produced by vbvr_next_frame_generate.py,
and assemble them into the InternVL-U imgen-SFT jsonl + meta schema (matching
data/datasets/maze_dataset/maze_gen/maze_gen_train.jsonl + data/meta/maze_gen_meta.json).

Production copy of data/scripts/vbvr_sample_windows.py. The duration-adaptive
delta / window-start logic (pick_delta, window_starts) is unchanged -- it was
already correct and matches the task spec exactly. What's different:

- Collect-all-then-subsample, not stop-early. The smoke-test version's
  sample_task() breaks out of its video loop the instant it hits
  TARGET_PER_TASK windows, so at production scale (7,000-11,000 videos/task)
  it would only ever touch the first fraction of them -- the rest silently
  unused. This version runs in two passes instead: Pass A opens every video
  just far enough to read fps/frame_count (no frame decoding) and records
  every eligible window as lightweight metadata; Pass B randomly subsamples
  that full candidate pool down to TARGET_PER_TASK with a seeded RNG; Pass C
  extracts and saves frames for ONLY the windows that survived the subsample,
  grouped by source video so a video contributing multiple selected windows
  is only opened once. This avoids decoding/writing frames for windows that
  would just be thrown away, which matters here since some tasks produce far
  more eligible windows than the target (see the data-construction spec
  artifact's Task 1 risk table -- e.g. key_door_matching projects ~17,640
  candidates against a 10,000 target).
- TARGET_PER_TASK = 10,000 (was 20 for the smoke test).
- split is hardcoded to "ID" for every row: vbvr_next_frame_generate.py's
  TASKS only contains the 10 canonical in-domain tasks, so there is no OOD
  case here (unlike the smoke test, which sampled all 15 tasks).
- Output paths moved off this repo's ssd2 volume (36GB free) onto
  /scratch/network/ssd/junlin (1.7TB free) for images; the small jsonl/meta
  files stay in-repo, matching data/meta/maze_gen_meta.json's convention.

Sampling logic (per the task spec) -- unchanged from the smoke test
---------------------------------------------------------------------
Objective per window: [I_t, I_{t+delta}, I_{t+2*delta}] -> I_{t+3*delta}

For every video we read the ACTUAL fps/frame_count/duration (never assume a
fixed fps or duration -- generators differ, e.g. most tasks render at 16fps
but 2d_geometric_transformation renders at 15fps; ground-truth clip length
also varies per task, from ~2.2s for mirror_reflection/grid_shift up to
~6s+ for stable_sort/rotation_puzzle/key_door_matching).

Duration-adaptive delta (a fixed 1s gap needs >=3s of span, which excludes
the shortest clips entirely):

    duration <  3.0s -> delta = 0.5s
    3.0 <= duration < 4.0s -> delta = 0.75s
    duration >= 4.0s -> delta = 1.0s

delta_frames = round(delta_seconds * fps)

Eligibility: a video must contain a valid t, t+delta, t+2*delta, t+3*delta
window, i.e. frame_count - 1 >= 3 * delta_frames. If the initially selected
delta tier doesn't fit, we step DOWN through the tiers (1.0 -> 0.75 -> 0.5)
and retry; if even delta=0.5s doesn't fit we skip the video entirely.

Multiple windows per video use a start-time stride of ~2*delta, capped at
MAX_WINDOWS_PER_VIDEO so long clips don't dominate the candidate pool before
the final subsample.

None of the 10 generators expose frame-indexed semantic/action-state
metadata (their metadata.json only carries generation *parameters* --
angles, grid layouts, velocities -- not a timeline of state-change frames),
so there is nothing to snap to and we fall back to uniform adaptive-stride
sampling for all tasks.

Prompt text: each generator's own prompt.txt describes the FULL video's
end-to-end outcome, which is the wrong framing once that same prompt is
reused verbatim for every window sampled from the clip. vbvr_next_frame_
prompt_adapt.adapt_prompt() rebuilds a task-specific "predict the next
frame, not the final state" instruction per window from the generator's
structured metadata.json parameters instead; the original free-text prompt
is kept alongside it in window_meta.json's "original_generator_prompt".

--no-ce mode
------------
VBVR's per-task video count is large and none of it carries a real text GT
(the "prompt" is a synthesized instruction describing the transform, not an
annotated caption) -- see the training-setup discussion this script's --no-ce
flag was added for. The default mode still trains a text-CE loss on the fixed
template GPT_TEMPLATE ("The next frame should look like this: <img>"), which
is identical across every row in the dataset and therefore contributes near-
zero learning signal once the model saturates on it after a few hundred
steps -- but it's still a full sentence's worth of tokenized/supervised
target computed every step for nothing.

``--no-ce`` swaps in GPT_TEMPLATE_NO_CE = "<img>" (bare marker, no leading
sentence) instead. This still goes through the same MultimodalImgenLazyDataset
path as the default mode (needed because the 3 context frames MUST stay as
real conditioning images -- the schema's other option, caption-only imgen,
takes no input images at all and would silently turn this into unconditional
T2I, which is not the task). Bare "<img>" reduces the supervised span to a
single, always-identical token, which is as close to zero text-CE as the
data side can get without a training-code change; it is not a literal
hard-zero (that needs a small trainer-side labels-masking hook, deliberately
NOT added here because an all--no_ce batch risks a NaN lm_loss if the loss
reduction divides by zero valid label positions -- ask for it separately if
you still want a hard guarantee once VBVR-only training is otherwise working).

Output paths are suffixed with "_no_ce" so the old jsonl/meta stay untouched
and both modes can coexist:
    default : data/datasets/vbvr_next_frame/next_frame_train.jsonl
    --no-ce : data/datasets/vbvr_next_frame/next_frame_train_no_ce.jsonl

--direction {forward,backward,mixed}
-------------------------------------
Built for the S1 "variable temporal target" training setting: instead of the
target always being the frame right after the 3 shown context frames
(forward, the default), the target can instead be the frame right before
them (backward). This needs NO new frame extraction -- each window already
decodes and saves all 4 raw frames [f0,f1,f2,f3] (forward: ctx=[f0,f1,f2],
target=f3); backward just re-points ctx=[f1,f2,f3], target=f0 over the SAME
4 files (see _direction_paths / FRAME_FILES). The prompt's closing
instruction (vbvr_next_frame_prompt_adapt.adapt_prompt's `direction` arg)
and the GPT-turn template both switch to match ("predict the frame that
occurred immediately before..." / "The preceding frame should look like
this: <img>"). `cond_image` (the VAE-conditioning frame) is written
explicitly per row and flips too (closest context frame to the target: last
for forward, first for backward) -- this must NOT be left to
MultimodalImgenLazyDataset's default fallback, which always takes the LAST
input image and would silently condition on the wrong frame for backward
rows. `mixed` picks forward/backward independently per window (~50/50,
reproducible per --seed). Non-forward directions write to `_backward` /
`_mixed` sibling jsonl/meta files, same pattern as --no-ce.
--direction backward/mixed also works with --reuse-samples at (near) zero
extra cost, INCLUDING against samples originally extracted in forward-only
runs -- see rebuild_rows_from_samples's docstring.

``--reuse-samples`` skips the expensive video-decode passes (A/B/C) entirely
and rebuilds the jsonl from the ``window_meta.json`` files a prior run already
wrote under SAMPLES_ROOT -- use this to flip between the two GPT templates
without re-extracting frames (the PNGs are identical in both modes; only the
jsonl's conversations text differs).

Usage:
    python VBVR-DataGeneration/next_frame/vbvr_next_frame_sample.py
    python VBVR-DataGeneration/next_frame/vbvr_next_frame_sample.py --tasks grid_shift mirror_reflection
    python VBVR-DataGeneration/next_frame/vbvr_next_frame_sample.py --no-ce
    python VBVR-DataGeneration/next_frame/vbvr_next_frame_sample.py --no-ce --reuse-samples  # cheap: no re-decoding
    python VBVR-DataGeneration/next_frame/vbvr_next_frame_sample.py --direction backward
    python VBVR-DataGeneration/next_frame/vbvr_next_frame_sample.py --direction mixed --reuse-samples  # S1, cheap
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
from PIL import Image

from vbvr_next_frame_generate import TASKS, ALL_TASKS, RAW_ROOT
from vbvr_next_frame_prompt_adapt import adapt_prompt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vbvr_task_presets import TASK_PRESETS, resolve_tasks

REPO_ROOT = Path("/scratch/network/ssd2/junlin/ssl_mllm")
SAMPLES_ROOT = Path("/scratch/network/ssd/junlin/vbvr_next_frame/samples")
DATA_DIR = REPO_ROOT / "data" / "vbvr_next_frame"
META_DIR = REPO_ROOT / "data" / "meta"

TARGET_PER_TASK = 10000
MAX_WINDOWS_PER_VIDEO = 7
MIN_DELTA = 0.5
DELTA_TIERS = [1.0, 0.75, 0.5]  # descending, used for step-down retry

HUMAN_TEMPLATE = "<image>\n<image>\n<image>\n{prompt}"
GPT_TEMPLATE_FORWARD = "The next frame should look like this: <img>"
GPT_TEMPLATE_BACKWARD = "The preceding frame should look like this: <img>"
# --no-ce mode: bare marker, no supervised sentence -- see module docstring.
GPT_TEMPLATE_NO_CE = "<img>"

DIRECTIONS = ("forward", "backward")
# Physical frame files saved per window by extract_selected(), in raw temporal
# order [t, t+delta, t+2*delta, t+3*delta] -- ALWAYS saved this way regardless
# of --direction. "backward" windows are built by re-pointing which 3 of these
# 4 files are context vs. target, not by extracting anything new (see S1
# variable-target design discussion): the physical frames for a forward window
# already cover exactly what a backward window needs.
FRAME_FILES = ["frame_0.png", "frame_1.png", "frame_2.png", "target.png"]


def _direction_paths(win_dir, direction):
    """(context_paths, target_path, cond_path) for a window dir, given direction.
    cond_path is the context frame closest in time to the target -- last
    context frame for forward, first for backward (see MultimodalImgenLazyDataset's
    cond_image fallback: it defaults to the LAST input image, which is only
    correct for forward, hence we set cond_image explicitly for both)."""
    f0, f1, f2, f3 = (win_dir / n for n in FRAME_FILES)
    if direction == "backward":
        return [f1, f2, f3], f0, f1
    return [f0, f1, f2], f3, f2


def _pick_direction(direction_arg, rng):
    if direction_arg == "mixed":
        return "backward" if rng.random() < 0.5 else "forward"
    if direction_arg not in DIRECTIONS:
        raise ValueError(f"unknown direction {direction_arg!r}")
    return direction_arg


def output_paths(no_ce, direction="forward", no_caption=False, out_dir=None, stem="next_frame"):
    """(jsonl_path, meta_path, summary_path) for the given mode. --no-ce,
    non-forward --direction, and --no-caption each write to their own
    suffixed siblings so no combination overwrites another. out_dir/stem let
    a task subset (e.g. the 2026-08-29 OOD-ext 4-task run) write to its own
    jsonl instead of clobbering the 10-ID-task file."""
    suffix = "_no_ce" if no_ce else ""
    suffix += "" if direction == "forward" else f"_{direction}"
    suffix += "_nocap" if no_caption else ""
    d = out_dir if out_dir is not None else DATA_DIR
    return (
        d / f"{stem}_train{suffix}.jsonl",
        META_DIR / f"vbvr_{stem}{suffix}_meta.json",
        d / f"sampling_summary{suffix}.json",
    )


def initial_delta(duration):
    if duration < 3.0:
        return 0.5
    if duration < 4.0:
        return 0.75
    return 1.0


def pick_delta(duration, fps, frame_count):
    """Return (delta_seconds, delta_frames) or None if no tier fits."""
    start_tier = initial_delta(duration)
    tiers = [d for d in DELTA_TIERS if d <= start_tier] or [MIN_DELTA]
    if start_tier not in tiers:
        tiers = [start_tier] + tiers
    for delta in tiers:
        delta_frames = max(1, round(delta * fps))
        if frame_count - 1 >= 3 * delta_frames:
            return delta, delta_frames
    return None


def window_starts(frame_count, delta_frames, cap):
    last_valid_start = frame_count - 1 - 3 * delta_frames
    if last_valid_start < 0:
        return []
    stride = max(1, 2 * delta_frames)
    starts = list(range(0, last_valid_start + 1, stride))
    return starts[:cap]


def extract_frame(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    if not ok:
        return None
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


# ---------------------------------------------------------------------------
# Pass A: enumerate every eligible window across every video, cheaply
# ---------------------------------------------------------------------------
def plan_candidates(task_name, max_windows_per_video):
    task_raw_dir = RAW_ROOT / task_name
    video_paths = sorted(task_raw_dir.rglob("ground_truth.*"))

    candidates = []
    stats = {"videos_available": len(video_paths), "videos_eligible": 0,
             "videos_skipped_too_short": 0, "videos_skipped_missing_files": 0,
             "windows_per_eligible_video": [], "deltas_used": []}

    for video_path in video_paths:
        sample_dir = video_path.parent
        prompt_path = sample_dir / "prompt.txt"
        meta_path = sample_dir / "metadata.json"
        if not prompt_path.exists() or not meta_path.exists():
            stats["videos_skipped_missing_files"] += 1
            continue

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if not fps or frame_count < 4:
            stats["videos_skipped_too_short"] += 1
            continue
        duration = frame_count / fps

        picked = pick_delta(duration, fps, frame_count)
        if picked is None:
            stats["videos_skipped_too_short"] += 1
            continue
        delta_seconds, delta_frames = picked

        starts = window_starts(frame_count, delta_frames, cap=max_windows_per_video)
        if not starts:
            stats["videos_skipped_too_short"] += 1
            continue

        stats["videos_eligible"] += 1
        stats["windows_per_eligible_video"].append(len(starts))
        stats["deltas_used"].append(delta_seconds)

        for start in starts:
            idxs = [start, start + delta_frames, start + 2 * delta_frames, start + 3 * delta_frames]
            candidates.append({
                "video_path": video_path, "sample_dir": sample_dir,
                "fps": fps, "frame_count": frame_count, "duration": duration,
                "delta_seconds": delta_seconds, "delta_frames": delta_frames,
                "start": start, "indices": idxs,
            })

    return candidates, stats


def _build_row(task_name, category, global_idx, image_paths, target_path, cond_path,
                prompt, no_ce, direction, has_caption=True, setting_label=None):
    if no_ce:
        gpt_value = GPT_TEMPLATE_NO_CE
    else:
        gpt_value = GPT_TEMPLATE_BACKWARD if direction == "backward" else GPT_TEMPLATE_FORWARD
    row = {
        "id": f"vbvr_{task_name}_{global_idx:05d}",
        "task_type": "imgen",
        "task_name": task_name,
        "category": category,
        "split": "ID",
        "direction": direction,
        "has_caption": bool(has_caption),
        "image": image_paths,
        "target_image": str(target_path),
        "cond_image": str(cond_path),
        "no_ce": bool(no_ce),
        "conversations": [
            {"from": "human", "value": HUMAN_TEMPLATE.format(prompt=prompt)},
            {"from": "gpt", "value": gpt_value},
        ],
    }
    if setting_label:
        row["setting"] = setting_label
    return row


# ---------------------------------------------------------------------------
# Pass C: extract frames for the subsampled candidates, grouped by video
# ---------------------------------------------------------------------------
def extract_selected(task_name, category, selected, out_dir, rng, direction_arg="forward", no_ce=False,
                      no_caption=False, setting_label=None):
    by_video = defaultdict(list)
    for c in selected:
        by_video[c["video_path"]].append(c)

    rows = []
    global_idx = 0
    for video_path, cands in by_video.items():
        sample_dir = cands[0]["sample_dir"]
        original_prompt = (sample_dir / "prompt.txt").read_text().strip()
        raw_parameters = json.loads((sample_dir / "metadata.json").read_text()).get("parameters", {})

        cap = cv2.VideoCapture(str(video_path))
        for c in sorted(cands, key=lambda x: x["start"]):
            frames = [extract_frame(cap, i) for i in c["indices"]]
            if any(f is None for f in frames):
                continue

            # direction decided BEFORE saving anything: physical frame files are
            # always saved in raw temporal order (frame_0/1/2/target.png), so the
            # only thing direction affects here is (a) which 3 of the 4 saved
            # files become ctx/target/cond in the jsonl row, and (b) the prompt.
            direction = _pick_direction(direction_arg, rng)
            prompt = adapt_prompt(task_name, raw_parameters, direction=direction,
                                   include_scene=not no_caption)

            win_dir = out_dir / f"{global_idx:05d}"
            win_dir.mkdir(parents=True, exist_ok=True)
            for k, frame in enumerate(frames[:3]):
                frame.save(win_dir / f"frame_{k}.png")
            frames[3].save(win_dir / "target.png")
            ctx_paths, target_path, cond_path = _direction_paths(win_dir, direction)

            window_meta = {
                "task_name": task_name, "category": category, "split": "ID",
                "source_video": str(video_path), "source_task_id": sample_dir.name,
                "fps": c["fps"], "frame_count": c["frame_count"], "duration_s": round(c["duration"], 3),
                "delta_s": c["delta_seconds"], "delta_frames": c["delta_frames"],
                "frame_indices": c["indices"], "frame_times_s": [round(i / c["fps"], 3) for i in c["indices"]],
                "direction": direction,
                "has_caption": not no_caption,
                "prompt": prompt,
                "raw_parameters": raw_parameters,
                "original_generator_prompt": original_prompt,
            }
            (win_dir / "window_meta.json").write_text(json.dumps(window_meta, indent=2))

            rows.append(_build_row(
                task_name, category, global_idx,
                [str(p) for p in ctx_paths], target_path, cond_path,
                prompt, no_ce, direction, has_caption=not no_caption, setting_label=setting_label,
            ))
            global_idx += 1
        cap.release()

    return rows


def rebuild_rows_from_samples(task_name, category, out_dir, seed, direction_arg="forward", no_ce=False,
                               no_caption=False, setting_label=None, limit=None):
    """Fast path for --reuse-samples: rebuild rows from window_meta.json files a
    prior run already wrote, with no video decoding at all -- this is also how
    --direction backward/mixed AND --no-caption get built cheaply: the 4
    physical frame files a forward+captioned run already saved
    (frame_0/1/2/target.png, raw temporal order) are exactly what a
    backward/no-caption window needs too -- direction only changes which 3 of
    the 4 files are ctx/target/cond (see _direction_paths), and no_caption
    only changes which prompt text adapt_prompt() returns for those same
    images. No new PNGs are ever written here.

    Requires window_meta.json to carry "raw_parameters" (added when this
    script's extract_selected() wrote it) so the prompt can be regenerated
    for whatever --direction/--no-caption this rebuild asks for, independent
    of whatever the ORIGINAL extraction run happened to pick for that window.
    Older window_meta.json files without "raw_parameters" can only be reused
    at direction=forward, no_caption=False (falls back to the stored "prompt",
    which is exactly that combination in every pre-existing run).

    limit: stop after this many rows (for cheap smoke tests -- skips nothing,
    just takes the first N window dirs in sorted order).
    """
    rows = []
    for win_dir in sorted(out_dir.glob("[0-9]" * 5)):
        if limit is not None and len(rows) >= limit:
            break
        meta_path = win_dir / "window_meta.json"
        target_path = win_dir / "target.png"
        frame_paths = [win_dir / f"frame_{k}.png" for k in range(3)]
        if not meta_path.exists() or not target_path.exists() or not all(p.exists() for p in frame_paths):
            continue
        meta = json.loads(meta_path.read_text())
        global_idx = int(win_dir.name)

        win_rng = random.Random(f"{seed}:{task_name}:{win_dir.name}:direction")
        direction = _pick_direction(direction_arg, win_rng)

        raw_parameters = meta.get("raw_parameters")
        if raw_parameters is not None:
            prompt = adapt_prompt(task_name, raw_parameters, direction=direction,
                                   include_scene=not no_caption)
        elif direction == "forward" and not no_caption:
            prompt = meta["prompt"]
        else:
            raise ValueError(
                f"{meta_path} has no 'raw_parameters' (pre-dates --direction/"
                f"--no-caption support) so it cannot be rebuilt at "
                f"direction={direction!r}, no_caption={no_caption!r}; "
                "re-extract this task without --reuse-samples first."
            )

        ctx_paths, tgt_path, cond_path = _direction_paths(win_dir, direction)
        rows.append(_build_row(
            task_name, category, global_idx,
            [str(p) for p in ctx_paths], tgt_path, cond_path,
            prompt, no_ce, direction, has_caption=not no_caption, setting_label=setting_label,
        ))
    return rows


def sample_task(task_name, category, target_per_task, max_windows_per_video, rng,
                 direction_arg="forward", no_ce=False, no_caption=False, setting_label=None,
                 samples_root=None):
    candidates, stats = plan_candidates(task_name, max_windows_per_video)
    stats["candidates_total"] = len(candidates)

    if len(candidates) > target_per_task:
        selected = rng.sample(candidates, target_per_task)
    else:
        selected = candidates
    stats["candidates_selected"] = len(selected)
    stats["shortfall"] = len(selected) < target_per_task

    out_dir = (samples_root or SAMPLES_ROOT) / task_name
    # Same rng continues to drive per-window direction picks (--direction mixed)
    # deterministically, then the final shuffle -- both downstream of the seeded
    # candidate subsample above, so the whole run stays reproducible per --seed.
    rows = extract_selected(task_name, category, selected, out_dir, rng,
                             direction_arg=direction_arg, no_ce=no_ce,
                             no_caption=no_caption, setting_label=setting_label)
    rng.shuffle(rows)
    return rows, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--task-preset", default=None,
                     help=f"named --tasks shortcut from vbvr_task_presets.py (valid: {list(TASK_PRESETS)}); "
                          "mutually exclusive with --tasks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-per-task", type=int, default=TARGET_PER_TASK)
    ap.add_argument("--max-windows-per-video", type=int, default=MAX_WINDOWS_PER_VIDEO)
    ap.add_argument("--no-ce", action="store_true",
                     help="Drop the text-CE sentence: GPT turn becomes bare '<img>' "
                          "instead of 'The next frame should look like this: <img>'. "
                          "Writes to *_no_ce sibling jsonl/meta files; default-mode "
                          "outputs are untouched. See module docstring.")
    ap.add_argument("--reuse-samples", action="store_true",
                     help="Skip video decoding (passes A/B/C) and rebuild the jsonl "
                          "from an existing run's window_meta.json files under "
                          "SAMPLES_ROOT -- use to flip --no-ce or --direction cheaply "
                          "once frames have already been extracted once (a forward "
                          "extraction already saves all 4 raw frames a backward "
                          "window needs -- see _direction_paths).")
    ap.add_argument("--direction", choices=["forward", "backward", "mixed"], default="forward",
                     help="forward (default, unchanged): ctx=[f0,f1,f2] -> target=f3 "
                          "(predict the next frame). backward: ctx=[f1,f2,f3] -> "
                          "target=f0 (predict the frame immediately before them) -- "
                          "the S1 variable-target training setting. mixed: each "
                          "window independently forward or backward (~50/50, "
                          "reproducible per --seed). Writes to *_backward / *_mixed "
                          "sibling jsonl/meta files; forward-mode outputs are never "
                          "overwritten.")
    ap.add_argument("--out-dir", type=str, default=None,
                     help="Override the jsonl/summary output directory (default: "
                          "DATA_DIR). Use for a task subset that must not overwrite "
                          "the default next_frame_train[_no_ce].jsonl.")
    ap.add_argument("--out-stem", type=str, default="next_frame",
                     help="Override the output filename stem (default: 'next_frame' "
                          "-> next_frame_train[_no_ce].jsonl / vbvr_next_frame[_no_ce]_meta.json).")
    ap.add_argument("--no-caption", action="store_true",
                     help="Drop the per-task scene description from the human prompt, "
                          "leaving only the generic direction closing -- this is the "
                          "'no C' half of the S0-S3 training-setting design (S0/S1 vs. "
                          "S2/S3). Orthogonal to --no-ce (that's about the GPT turn's "
                          "supervised text, this is about the human turn's task-specific "
                          "content). Writes to *_nocap sibling jsonl/meta files.")
    ap.add_argument("--setting-label", type=str, default=None,
                     help="Free-text label (e.g. 'S0', 'S1', 'S2') stamped verbatim into "
                          "every row's \"setting\" field. Pure bookkeeping -- not "
                          "validated against --direction/--no-caption, just makes the "
                          "output jsonl self-describing when mixing multiple settings.")
    ap.add_argument("--limit", type=int, default=None,
                     help="With --reuse-samples: stop after this many rows PER TASK "
                          "(first N window dirs in sorted order). For cheap smoke tests; "
                          "ignored on the fresh-extraction path.")
    ap.add_argument("--samples-root", type=str, default=None,
                     help="Override SAMPLES_ROOT (default: the shared production "
                          "location under /scratch/network/ssd/junlin/vbvr_next_frame/"
                          "samples/). Use a scratch dir for smoke tests so a small "
                          "fresh extraction can never collide with / overwrite the "
                          "existing 10k-window production samples for a task (fresh "
                          "extraction always starts numbering window dirs at 00000).")
    args = ap.parse_args()

    task_names = resolve_tasks(args.tasks, args.task_preset, ALL_TASKS, list(TASKS.keys()))
    unknown = [t for t in task_names if t not in ALL_TASKS]
    if unknown:
        raise SystemExit(f"unknown task names: {unknown} (valid: {list(ALL_TASKS.keys())})")

    samples_root = Path(args.samples_root).resolve() if args.samples_root else SAMPLES_ROOT
    out_dir_override = Path(args.out_dir).resolve() if args.out_dir else None
    jsonl_path, meta_path, summary_path = output_paths(
        args.no_ce, args.direction, args.no_caption, out_dir_override, args.out_stem)

    all_rows = []
    summary = {}
    for task_name in task_names:
        _, category = ALL_TASKS[task_name]
        out_dir = samples_root / task_name
        if args.reuse_samples:
            rows = rebuild_rows_from_samples(
                task_name, category, out_dir, args.seed,
                direction_arg=args.direction, no_ce=args.no_ce,
                no_caption=args.no_caption, setting_label=args.setting_label,
                limit=args.limit,
            )
            stats = {"reused_from": str(out_dir)}
        else:
            rng = random.Random(f"{args.seed}:{task_name}")  # per-task RNG, still fully reproducible
            rows, stats = sample_task(
                task_name, category, args.target_per_task, args.max_windows_per_video,
                rng, direction_arg=args.direction, no_ce=args.no_ce,
                no_caption=args.no_caption, setting_label=args.setting_label,
                samples_root=samples_root,
            )
        all_rows.extend(rows)
        summary[task_name] = {**stats, "windows_collected": len(rows)}
        status = "OK" if len(rows) == args.target_per_task else "SHORT"
        extra = (f"candidates={stats['candidates_total']:6d} "
                 f"videos_eligible={stats['videos_eligible']:6d}/{stats['videos_available']:6d} "
                 f"deltas={sorted(set(stats['deltas_used']))}") if not args.reuse_samples else f"(reused from {out_dir})"
        print(f"[{status}] {task_name:28s} windows={len(rows):6d}/{args.target_per_task} {extra}", flush=True)

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    meta_suffix = ("_no_ce" if args.no_ce else "") + ("" if args.direction == "forward" else f"_{args.direction}") \
        + ("_nocap" if args.no_caption else "")
    meta_name = f"vbvr_{args.out_stem}{meta_suffix}"
    meta = {
        meta_name: {
            "root": "/",
            "annotation": str(jsonl_path),
            "data_augment": False,
            "max_dynamic_patch": 1,
            "repeat_time": 1,
            "length": len(all_rows),
            "task_type": "imgen",
        }
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))

    short = [t for t, s in summary.items() if s["windows_collected"] < args.target_per_task]
    print(f"\nTotal windows: {len(all_rows)} -> {jsonl_path}")
    print(f"Meta -> {meta_path}")
    if short:
        print(f"Tasks short of {args.target_per_task}: {short}")


if __name__ == "__main__":
    main()
