"""Builder for WorldPrediction-WM (World Modeling) train/eval jsonl.

Splits the WorldPrediction benchmark (facebookresearch/WorldPrediction, vendored
at ../../../eval/suites/worldprediction/) by data source:
    eval  = COIN (all of it)
    train = CrossTask + EPIC-KITCHENS-100 + IKEAASM
    (EgoExo4D excluded: no local video root, gated dataset -- see
     eval/suites/worldprediction/data/video_roots.py)

Every action clip (a candidate's video segment) is turned into exactly
`--frames` frames (default 4) using the SAME function the InternVL-U
WorldPrediction adapter calls at inference time --
eval/suites/worldprediction/data/load_frames.py::collect_frames, called with
the same desired_fps=5.0 as eval/suites/worldprediction/configs/vlm/internvlu/
InternVL-U-{4,8}f.json (model_max_frames must match --frames on the eval-side
config too -- see MAINTENANCE note). No new sampling logic is written here;
this script only imports and calls the existing function so offline (train)
and on-the-fly (eval) frame extraction are provably identical.

collect_frames() only guarantees exactly `model_max_frames` frames when
duration >= model_max_frames/desired_fps; shorter clips fall back to an
fps-based path that can yield fewer frames. To make "always exactly N frames"
an invariant rather than a best-effort, any sample with a candidate clip
shorter than model_max_frames/desired_fps is dropped entirely (applied
identically to train and eval) -- this threshold scales with --frames (0.8s
at 4 frames, 1.6s at 8 frames), so a re-run at a different frame count is not
guaranteed to keep/drop the same set of samples.

The MCQ prompt (initial/final state images + 4 lettered candidate actions +
answer format) is a deliberate near-verbatim port of
eval/suites/worldprediction/model/vlm.py::VLM.select_action's message
construction and
eval/suites/worldprediction/model/vlms/internvlu.py::InternVLU._build_prompt's
flattening (Image{i}: <image> / Frame{i}: <image> numbering, "User: " prefix).
This is a port, not a shared import, because vlm.py's construction is inlined
inside the model-calling method rather than factored into a standalone
function -- if that prompt template ever changes, this port needs a manual
re-sync (see MAINTENANCE note below).

Usage:
    python build_worldprediction_wm.py --task WM               # 4 frames (default), writes wm_train.jsonl/wm_eval.jsonl/frames/
    python build_worldprediction_wm.py --task WM --frames 8     # writes wm_train_8f.jsonl/wm_eval_8f.jsonl/frames_8f/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = "/scratch/network/ssd2/junlin/ssl_mllm/baseline_pipeline"
SUITE_DIR = f"{REPO_ROOT}/eval/suites/worldprediction"
OUT_DIR = f"{REPO_ROOT}/data/datasets/worldprediction"

# MAINTENANCE: keep in sync with eval/suites/worldprediction/data/video_roots.py
VIDEO_ROOTS = {
    "COIN": "/scratch/local/ssd/junlin/data/coin/videos",
    "CrossTask": "/scratch/local/ssd/junlin/data/crosstask/videos",
    "EPIC-KITCHENS-100": "/scratch/local/ssd/junlin/data/worldprediction/epic-kitchen",
    "IKEAASM": "/scratch/local/ssd/junlin/data/worldprediction/ikea-asm",
}
EVAL_DATASETS = {"COIN"}
TRAIN_DATASETS = {"CrossTask", "EPIC-KITCHENS-100", "IKEAASM"}
# EgoExo4D intentionally excluded: gated dataset, no local video root.

DESIRED_FPS = 5.0  # matches configs/vlm/internvlu/InternVL-U-{4,8}f.json's desired_fps
OPTIONS_ID = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Set by main() from --frames. Module-level so process_sample() (called with
# no extra args by main()'s loop) can see it without threading it through
# every call -- this script is single-run-per-process, never imported.
MODEL_MAX_FRAMES = 4
MIN_CLIP_DURATION = MODEL_MAX_FRAMES / DESIRED_FPS  # see collect_frames branch

sys.path.insert(0, SUITE_DIR)
from data.load_frames import collect_frames, get_frames_from_video  # noqa: E402


def sanitize(uid: str) -> str:
    return uid.replace("|", "_").replace("/", "_")


def build_wm_messages(states, candidate_actions):
    """Port of model/vlm.py::VLM.select_action's message construction (prompt
    text only, up to but not including the model .generate() call / response
    parsing). `states` = {"initial_state": PIL, "final_state": PIL}.
    `candidate_actions` = [{"frames": [4 PIL frames], ...}, ...] in the same
    A/B/C/D order the ground-truth letter was computed against.
    """
    content = [
        {"type": "text", "text": "You are given an initial and final world state. "
                                  "Identify which candidate action most likely caused the change."},
        {"type": "text", "text": "Initial world state:"},
        {"type": "image", "image": states["initial_state"]},
        {"type": "text", "text": "Final world state:"},
        {"type": "image", "image": states["final_state"]},
        {"type": "text", "text": "Candidate actions:"},
    ]
    for i, action in enumerate(candidate_actions):
        label = OPTIONS_ID[i]
        content.append({"type": "text", "text": f"{label}. Candidate Action:"})
        content.append({"type": "video", "video": action["frames"]})
    content.append({"type": "text", "text": (
        "Which action most likely caused the world state changes?\n"
        "Select by choosing the corresponding letter in the format {\"action\": \"X\"}.\n"
    )})
    return [{"role": "user", "content": content}]


def flatten_prompt(messages):
    """Port of model/vlms/internvlu.py::InternVLU._build_prompt (text-flattening
    half only -- video items here are already-materialized PIL frame lists,
    matching what _materialize_videos would have produced)."""
    text_parts = []
    image_paths = []  # we keep PIL images out of the jsonl; caller saves them and
                       # passes back paths in the same order via `image_idx` bookkeeping
    image_idx = 0
    for msg in messages:
        role = msg["role"]
        buf = []
        for item in msg["content"]:
            t = item["type"]
            if t == "text":
                buf.append(item["text"])
            elif t == "image":
                image_idx += 1
                buf.append(f"Image{image_idx}: <image>")
            elif t == "video":
                for _ in item["video"]:
                    image_idx += 1
                    buf.append(f"Frame{image_idx}: <image>")
        if buf:
            prefix = "User: " if role == "user" else f"{role.capitalize()}: "
            text_parts.append(prefix + "\n".join(buf))
    return "\n\n".join(text_parts)


def process_sample(dataset_name, sample_uid, sample_info, video_root, frames_out_dir):
    abs_states_video = os.path.join(video_root, sample_info["states"]["video"])
    init_frame = get_frames_from_video(abs_states_video, sample_info["states"]["segment_start_time"])
    final_frame = get_frames_from_video(abs_states_video, sample_info["states"]["segment_end_time"])
    if init_frame is None or final_frame is None:
        return None, "missing_state_frame"

    ground_truth = sample_info["ground_truth"]
    candidates = sample_info["candidates"]
    gt_idx = next((i for i, c in enumerate(candidates) if c["segment_uid"] == ground_truth), None)
    if gt_idx is None:
        return None, "ground_truth_not_in_candidates"

    cand_frames = []
    for cand in candidates:
        abs_cand_video = os.path.join(video_root, cand["video"])
        if not os.path.isfile(abs_cand_video):
            return None, f"missing_video:{abs_cand_video}"
        duration = cand["segment_end_time"] - cand["segment_start_time"]
        if duration < MIN_CLIP_DURATION:
            return None, f"clip_too_short:{duration:.3f}s"
        frames, _ = collect_frames(
            video_path=abs_cand_video,
            start_time=cand["segment_start_time"],
            end_time=cand["segment_end_time"],
            model_max_frames=MODEL_MAX_FRAMES,
            desired_fps=DESIRED_FPS,
        )
        frames = [f for f in frames if f is not None]
        if len(frames) != MODEL_MAX_FRAMES:
            return None, f"incomplete_frames:{len(frames)}/{MODEL_MAX_FRAMES}"
        cand_frames.append({"frames": frames, "segment_uid": cand["segment_uid"], "action_label": cand.get("action_label")})

    # Save frames to disk, build the jsonl "image" path list in the exact
    # order flatten_prompt()'s image_idx numbering expects.
    sample_dir = os.path.join(frames_out_dir, sanitize(sample_uid))
    os.makedirs(sample_dir, exist_ok=True)
    image_paths = []

    init_path = os.path.join(sample_dir, "state_init.png")
    init_frame.convert("RGB").save(init_path)
    image_paths.append(init_path)

    final_path = os.path.join(sample_dir, "state_final.png")
    final_frame.convert("RGB").save(final_path)
    image_paths.append(final_path)

    for i, cand in enumerate(cand_frames):
        label = OPTIONS_ID[i]
        for j, frame in enumerate(cand["frames"]):
            p = os.path.join(sample_dir, f"cand_{label}_f{j}.png")
            frame.convert("RGB").save(p)
            image_paths.append(p)

    messages = build_wm_messages(
        {"initial_state": init_frame, "final_state": final_frame},
        cand_frames,
    )
    prompt_text = flatten_prompt(messages)
    answer_letter = OPTIONS_ID[gt_idx]
    answer_text = json.dumps({"action": answer_letter})

    row = {
        "id": f"wp_wm_{dataset_name}_{sanitize(sample_uid)}",
        "task_type": "understanding",
        "benchmark": "WorldPrediction-WM",
        "dataset_source": dataset_name,
        "sample_uid": sample_uid,
        "ground_truth_segment_uid": ground_truth,
        "answer_letter": answer_letter,
        "num_candidates": len(candidates),
        "image": image_paths,
        "conversations": [
            {"from": "human", "value": prompt_text},
            {"from": "gpt", "value": answer_text},
        ],
    }
    return row, None


def main():
    global MODEL_MAX_FRAMES, MIN_CLIP_DURATION

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="WM", choices=["WM"])
    parser.add_argument("--frames", type=int, default=4,
                         help="frames per action clip (model_max_frames). Default 4 writes the "
                              "original unsuffixed paths (wm_train.jsonl/frames/); any other "
                              "value writes frames_{n}f/ and wm_{train,eval}_{n}f.jsonl instead, "
                              "so a re-run never clobbers the 4-frame build.")
    parser.add_argument("--limit", type=int, default=None, help="cap samples per dataset, for smoke testing")
    args = parser.parse_args()

    MODEL_MAX_FRAMES = args.frames
    MIN_CLIP_DURATION = MODEL_MAX_FRAMES / DESIRED_FPS

    suffix = "" if args.frames == 4 else f"_{args.frames}f"
    frames_dir = os.path.join(OUT_DIR, f"frames{suffix}")
    train_path = os.path.join(OUT_DIR, f"wm_train{suffix}.jsonl")
    eval_path = os.path.join(OUT_DIR, f"wm_eval{suffix}.jsonl")

    ann_path = f"{SUITE_DIR}/data/WorldPrediction-{args.task}.json"
    data = json.load(open(ann_path))

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    train_rows, eval_rows = [], []
    skip_counts = {}
    for dataset_name, samples in data.items():
        if dataset_name not in EVAL_DATASETS and dataset_name not in TRAIN_DATASETS:
            continue  # EgoExo4D
        video_root = VIDEO_ROOTS[dataset_name]
        frames_out_dir = os.path.join(frames_dir, dataset_name)
        items = list(samples.items())
        if args.limit:
            items = items[: args.limit]
        n_ok, n_skip = 0, 0
        for sample_uid, sample_info in items:
            row, skip_reason = process_sample(dataset_name, sample_uid, sample_info, video_root, frames_out_dir)
            if row is None:
                n_skip += 1
                key = skip_reason.split(":")[0]
                skip_counts[key] = skip_counts.get(key, 0) + 1
                continue
            n_ok += 1
            if dataset_name in EVAL_DATASETS:
                eval_rows.append(row)
            else:
                train_rows.append(row)
        print(f"[{dataset_name}] kept {n_ok}/{len(items)}, skipped {n_skip}")

    with open(train_path, "w") as f:
        for row in train_rows:
            f.write(json.dumps(row) + "\n")
    with open(eval_path, "w") as f:
        for row in eval_rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nWrote {len(train_rows)} train rows -> {train_path}")
    print(f"Wrote {len(eval_rows)} eval rows -> {eval_path}")
    print(f"Skip reasons: {skip_counts}")


if __name__ == "__main__":
    main()
