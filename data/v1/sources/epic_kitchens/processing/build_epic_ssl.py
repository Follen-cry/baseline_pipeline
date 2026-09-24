"""Builder for EPIC-KITCHENS-100 next-frame SSL data (selection FFS + generation),
for InternVL-U SFT. Mirrors data/scripts/build_nwm_ssl.py's structure (see
data/specs/EPIC_SSL_DATA_SPEC.md) but replaces NWM/RECON's pose-derived actions and
hard negatives with EPIC-100's real narration annotations and temporal-distance
negatives, since egocentric kitchen video has no camera pose.

Source videos (116 MP4s, EPIC-KITCHENS-100 validation split) live ONLY on this
node's local SSD -- /scratch/local/ssd/junlin/data/worldprediction/epic-kitchen/
<P##>/<P##_##>.MP4 (downloaded by Evaluation/WorldPrediction/download_epic_kitchen.sh
for the WorldPrediction eval benchmark; the videos are NOT network-visible). This
script therefore MUST run on that node (hostname torrnode11) for the "anchors"
stage (the only stage that touches video files via decord). Extracted frame jpgs
are written to network storage (/scratch/network/ssd/junlin/epic_ssl_frames/,
same filesystem as NWM's RECON images) so downstream training can read them from
any node.

Stages (run with --stage, in order):
  splits   -- carve the 116 local videos into internal train/eval video pools
              (data/datasets/epic_ssl/splits/{train,eval}_videos.txt)
  anchors  -- compile transitions (context frames -> next frame) at a fixed
              sampling grid, match EPIC-100 narrations for action_text, mine
              temporal-distance hard negatives for 4-way MCQ candidates,
              subsample to 4000 train + 300 eval anchors, extract only the
              frames actually used via decord -> jpg
              (data/datasets/epic_ssl/anchors/anchors_{train,eval}.jsonl)
  ffs      -- build epic_ffs_{action,noaction}_{train,eval}.jsonl (task_type
              "understanding", 4-way MCQ)
  gen      -- build epic_gen_{action,noaction}_{train,eval}.jsonl (task_type
              "imgen", next-frame generation)
  dual     -- convert epic_ffs_{action,noaction}_{train,eval}.jsonl 1:1 into
              epic_dual_{action,noaction}_{train,eval}.jsonl (task_type
              "imgen", imgen_form "multimodal") via the SAME converter NWM
              uses (Model_Related/InternVLU/InternVL/internvl_chat/tools/
              make_nwm_dual_ssl.py -- fully dataset-agnostic: it only reads
              the FFS row's 8-image list + answer letter, so it's reused
              unmodified rather than ported)

Usage (must run on torrnode11 for splits/anchors; ffs/gen/dual only touch jsonls):
  python data/scripts/build_epic_ssl.py --stage splits
  python data/scripts/build_epic_ssl.py --stage anchors
  python data/scripts/build_epic_ssl.py --stage ffs
  python data/scripts/build_epic_ssl.py --stage gen
  python data/scripts/build_epic_ssl.py --stage dual
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import socket
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DATA = "/scratch/network/ssd2/junlin/ssl_mllm/data"
OUT = f"{DATA}/epic_ssl"
DUAL_TOOL_DIR = ("/scratch/network/ssd2/junlin/ssl_mllm/Model_Related/InternVLU/"
                  "InternVL/internvl_chat/tools")
LOCAL_VIDEO_ROOT = "/scratch/local/ssd/junlin/data/worldprediction/epic-kitchen"
FRAMES_ROOT = "/scratch/network/ssd/junlin/epic_ssl_frames"
NARRATION_CSV = f"{OUT}/EPIC_100_validation_local116.csv"
VIDEO_MANIFEST = f"{OUT}/video_manifest.json"

SAMPLE_FPS = 2.0          # our working "grid" -- context frames are 0.5s apart
CONTEXT_SIZE = 3          # 3 context frames, spanning 1.0s -- was 4; shrunk 2026-08-31
                           # to match VBVR/panda70m's 3-frame convention (shared real-
                           # world-mix grid: SAMPLE_FPS=2.0, CONTEXT_SIZE=3, HORIZON=1
                           # everywhere except NWM, whose native 4fps pre-extraction
                           # keeps its own units -- see build_nwm_ssl.py)
HORIZON = 1               # target = 1 grid-step (0.5s) after the last context frame
STRIDE = 8                # grid-steps between anchor windows (4s), keeps windows
                           # well-separated given only 116 source videos
NUM_CANDIDATES = 4
SPLIT_SEED = 42
ANCHOR_SEED = 0
EVAL_HOLDOUT_VIDEOS = 20  # of 116 videos, held out whole for our eval pool
TARGET_TRAIN = 7000       # was 4000; raised 2026-08-31 as part of the real-video-mix
                           # rebalance (measured ceiling at CONTEXT_SIZE=3: 7391)
TARGET_EVAL = 300
PER_VIDEO_CAP_TRAIN = 100  # was 60; raised so the higher TARGET_TRAIN can still be
                            # reached without falling back to full cap-relax
PER_VIDEO_CAP_EVAL = 25


def _assert_on_source_node():
    host = socket.gethostname()
    if host != "torrnode11":
        raise RuntimeError(
            f"EPIC-KITCHENS source MP4s only exist on torrnode11's local SSD "
            f"({LOCAL_VIDEO_ROOT}); this node is {host!r}. The 'anchors' stage "
            f"(video decoding) must run on torrnode11.")


@dataclass(frozen=True)
class TransitionRecord:
    sample_id: str
    video_id: str
    context_grid: list[int]
    target_grid: int
    context_native: list[int]
    target_native: int
    action_text: str
    narration_id: str

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


@dataclass(frozen=True)
class SelectionExample:
    example_id: str
    transition_id: str
    video_id: str
    context_paths: list[str]
    action_text: str
    narration_id: str
    candidate_paths: list[str]
    candidate_types: list[str]
    answer_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def answer_label(self):
        return chr(ord("A") + self.answer_index)

    def to_dict(self):
        value = asdict(self)
        value["answer_label"] = self.answer_label
        return value

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value.pop("answer_label", None)
        return cls(**value)


def read_jsonl(path, parser):
    records = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(parser(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return records


def write_jsonl(path, records):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            value = record.to_dict() if hasattr(record, "to_dict") else record
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _local_video_ids():
    ids = []
    for p in sorted(Path(LOCAL_VIDEO_ROOT).iterdir()):
        if p.is_dir() and p.name.startswith("P") and p.name != "EPIC-KITCHENS":
            for f in sorted(p.iterdir()):
                if f.suffix == ".MP4":
                    ids.append(f.stem)
    return ids


def build_video_manifest():
    """Probe fps/frame_count for all 116 local videos via cv2 (container-metadata
    read, near-instant -- decord's VideoReader() was observed to hang
    unpredictably for many minutes per file in this environment, cv2 does not).
    Run once, cached to disk."""
    _assert_on_source_node()
    import cv2
    manifest = {}
    ids = _local_video_ids()
    for i, vid in enumerate(ids):
        participant = vid.split("_")[0]
        path = f"{LOCAL_VIDEO_ROOT}/{participant}/{vid}.MP4"
        cap = cv2.VideoCapture(path)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        manifest[vid] = {"path": path, "fps": fps, "num_frames": n_frames,
                          "duration_sec": n_frames / fps if fps else 0}
        print(f"[manifest] {i + 1}/{len(ids)} {vid}: fps={fps:.2f} frames={n_frames}")
    Path(OUT).mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(VIDEO_MANIFEST, "w"), indent=2)
    print(f"[manifest] wrote {len(manifest)} videos -> {VIDEO_MANIFEST}")


def build_trajectory_splits():
    manifest = json.load(open(VIDEO_MANIFEST))
    ids = sorted(manifest.keys())
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(ids)
    eval_ids = sorted(ids[:EVAL_HOLDOUT_VIDEOS])
    train_ids = sorted(ids[EVAL_HOLDOUT_VIDEOS:])
    assert not (set(eval_ids) & set(train_ids))

    out_dir = Path(OUT) / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_videos.txt").write_text("\n".join(train_ids) + "\n")
    (out_dir / "eval_videos.txt").write_text("\n".join(eval_ids) + "\n")
    print(f"[splits] train={len(train_ids)} eval={len(eval_ids)} videos -> {out_dir}")


def _load_narrations():
    """video_id -> sorted list of (start_frame, stop_frame, narration_text, narration_id)."""
    by_video = defaultdict(list)
    with open(NARRATION_CSV) as f:
        for row in csv.DictReader(f):
            by_video[row["video_id"]].append(
                (int(row["start_frame"]), int(row["stop_frame"]),
                 row["narration"].strip(), row["narration_id"]))
    for vid in by_video:
        by_video[vid].sort(key=lambda x: x[0])
    return by_video


def _match_narration(segments, a, b):
    """Narration segment with the greatest overlap with native-frame interval
    [a, b] (the transition from the last context frame to the target frame).
    Returns (text, narration_id) or (None, None) if nothing overlaps."""
    best, best_overlap = None, 0
    for start, stop, text, nid in segments:
        overlap = min(b, stop) - max(a, start)
        if overlap > best_overlap:
            best_overlap = overlap
            best = (text, nid)
    return best if best is not None else (None, None)


def compile_transitions(video_ids, manifest, narrations_by_video):
    records = []
    for video_id in video_ids:
        info = manifest[video_id]
        fps, n_native = info["fps"], info["num_frames"]
        native_stride = max(1, round(fps / SAMPLE_FPS))
        n_grid = (n_native - 1) // native_stride + 1
        segments = narrations_by_video.get(video_id, [])
        if not segments:
            continue
        for current in range(CONTEXT_SIZE - 1, n_grid - HORIZON, STRIDE):
            context_grid = list(range(current - CONTEXT_SIZE + 1, current + 1))
            context_native = [g * native_stride for g in context_grid]
            target_grid = current + HORIZON
            target_native = target_grid * native_stride
            if target_native >= n_native or context_native[-1] >= n_native:
                continue
            action_text, narration_id = _match_narration(
                segments, context_native[-1], target_native)
            if action_text is None:
                continue
            records.append(TransitionRecord(
                f"epic:{video_id}:{current}:{target_grid}", video_id,
                context_grid, target_grid, context_native, target_native,
                action_text, narration_id))
    return records


def _negatives(target, pool, count):
    """Temporal-distance-based hard negatives (no pose available, unlike NWM):
    near_wrong_state = closest wrong target by grid distance (hardest, visually
    similar); mid_wrong_state = next closest; same_video_distractor = a distant
    target elsewhere in the same video (easiest). Mirrors NWM's _negatives
    tiering structure with the pose-specific wrong_yaw/wrong_distance tiers
    replaced by temporal-distance tiers, the only geometry available here."""
    seen, unused = {target.target_native}, []
    for x in pool:
        if x.target_native in seen:
            continue
        seen.add(x.target_native)
        unused.append(x)
    if len(unused) < count:
        return []
    unused.sort(key=lambda x: abs(x.target_grid - target.target_grid))
    selected = [(unused[0], "near_wrong_state")]
    if len(unused) > 1:
        selected.append((unused[1], "mid_wrong_state"))
    remaining = unused[2:]
    remaining.sort(key=lambda x: -abs(x.target_grid - target.target_grid))
    selected += [(x, "same_video_distractor") for x in remaining[:count - len(selected)]]
    return selected[:count]


def build_selection_examples(transitions, seed=0):
    by_video = defaultdict(list)
    for record in transitions:
        by_video[record.video_id].append(record)
    examples = []
    for record in transitions:
        negatives = _negatives(record, by_video[record.video_id], NUM_CANDIDATES - 1)
        if len(negatives) < NUM_CANDIDATES - 1:
            continue
        candidates = [(record.target_native, "ground_truth")] + \
                     [(x.target_native, kind) for x, kind in negatives]
        rng = random.Random(f"{seed}:{record.sample_id}")
        rng.shuffle(candidates)
        natives, kinds = map(list, zip(*candidates))
        examples.append(SelectionExample(
            f"selection:{record.sample_id}", record.sample_id, record.video_id,
            [_frame_path(record.video_id, n) for n in record.context_native],
            record.action_text, record.narration_id,
            [_frame_path(record.video_id, n) for n in natives], kinds,
            kinds.index("ground_truth"),
            {"context_native": record.context_native, "target_native": record.target_native,
             "candidate_natives": natives}))
    return examples


def _frame_path(video_id, native_idx):
    return f"{FRAMES_ROOT}/{video_id}/{native_idx}.jpg"


def _subsample_capped(examples, target_n, cap, seed):
    rng = random.Random(seed)
    order = list(examples)
    rng.shuffle(order)
    counts, selected, leftover = {}, [], []
    for ex in order:
        c = counts.get(ex.video_id, 0)
        if c < cap:
            selected.append(ex)
            counts[ex.video_id] = c + 1
            if len(selected) >= target_n:
                return selected
        else:
            leftover.append(ex)
    for ex in leftover:
        selected.append(ex)
        if len(selected) >= target_n:
            break
    if len(selected) < target_n:
        print(f"[anchors] WARNING: only found {len(selected)}/{target_n} examples after full relax")
    return selected


def _extract_frames(anchors):
    """Extract only the jpgs actually referenced by the final anchor set, via
    cv2 random-access seek+read (decord's VideoReader() was observed to hang
    unpredictably for many minutes per file in this environment; cv2's
    CAP_PROP_POS_FRAMES seek is fast and reliable here -- these are standard
    fixed-fps H.264 sources, not VFR, so seek accuracy is not a concern).
    Must run on torrnode11 (source MP4s)."""
    _assert_on_source_node()
    import cv2
    from PIL import Image

    needed = defaultdict(set)
    for ex in anchors:
        for p in ex.context_paths + ex.candidate_paths:
            video_id = Path(p).parent.name
            native_idx = int(Path(p).stem)
            needed[video_id].add(native_idx)

    manifest = json.load(open(VIDEO_MANIFEST))
    total_written, total_skipped = 0, 0
    for i, (video_id, indices) in enumerate(sorted(needed.items())):
        out_dir = Path(FRAMES_ROOT) / video_id
        out_dir.mkdir(parents=True, exist_ok=True)
        todo = sorted(idx for idx in indices if not (out_dir / f"{idx}.jpg").exists())
        if not todo:
            total_skipped += len(indices)
            continue
        path = manifest[video_id]["path"]
        cap = cv2.VideoCapture(path)
        for idx in todo:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                print(f"[frames] WARNING: failed to read {video_id}/{idx}, skipping")
                continue
            Image.fromarray(frame[:, :, ::-1]).save(out_dir / f"{idx}.jpg", quality=95)
        cap.release()
        total_written += len(todo)
        total_skipped += len(indices) - len(todo)
        print(f"[frames] {i + 1}/{len(needed)} {video_id}: wrote {len(todo)}, "
              f"skipped {len(indices) - len(todo)} existing")
    print(f"[frames] total: wrote {total_written}, skipped (already existed) {total_skipped}")


def build_anchors():
    manifest = json.load(open(VIDEO_MANIFEST))
    narrations = _load_narrations()
    splits_dir = Path(OUT) / "splits"
    anchors_dir = Path(OUT) / "anchors"
    anchors_dir.mkdir(parents=True, exist_ok=True)

    for split, video_file, target_n, cap in (
        ("train", splits_dir / "train_videos.txt", TARGET_TRAIN, PER_VIDEO_CAP_TRAIN),
        ("eval", splits_dir / "eval_videos.txt", TARGET_EVAL, PER_VIDEO_CAP_EVAL),
    ):
        video_ids = [x.strip() for x in video_file.read_text().splitlines() if x.strip()]
        transitions = compile_transitions(video_ids, manifest, narrations)
        print(f"[anchors] {split}: {len(transitions)} raw transitions "
              f"(narration-matched) across {len(video_ids)} videos")
        examples = build_selection_examples(transitions, seed=ANCHOR_SEED)
        print(f"[anchors] {split}: {len(examples)} selection examples "
              f"(candidate mining succeeded)")
        anchors = _subsample_capped(examples, target_n, cap, ANCHOR_SEED)
        write_jsonl(anchors_dir / f"anchors_{split}.jsonl", anchors)
        traj_used = len({a.video_id for a in anchors})
        print(f"[anchors] {split}: wrote {len(anchors)} anchors ({traj_used} distinct videos) "
              f"-> {anchors_dir}/anchors_{split}.jsonl")

    for split in ("train", "eval"):
        anchors = read_jsonl(anchors_dir / f"anchors_{split}.jsonl", SelectionExample.from_dict)
        bad = [a.example_id for a in anchors if a.candidate_types[a.answer_index] != "ground_truth"]
        assert not bad, f"oracle sanity failed for {len(bad)} anchors in {split}: {bad[:5]}"
    print("[anchors] oracle sanity check passed (answer_index always ground_truth)")

    all_anchors = (read_jsonl(anchors_dir / "anchors_train.jsonl", SelectionExample.from_dict) +
                   read_jsonl(anchors_dir / "anchors_eval.jsonl", SelectionExample.from_dict))
    print(f"[anchors] extracting frames for {len(all_anchors)} anchors...")
    _extract_frames(all_anchors)


def build_ffs_prompt_action(example: SelectionExample) -> str:
    n, k = len(example.context_paths), len(example.candidate_paths)
    labels = [chr(ord("A") + i) for i in range(k)]
    head = "".join(f"Frame {i + 1}: <image>\n" for i in range(n))
    body = (f"These are {n} consecutive first-person camera frames from an egocentric "
            f"kitchen video, in order.\n"
            f"Action being performed after the last frame: {example.action_text}\n")
    opts = "".join(f"Option {labels[i]}: <image>\n" for i in range(k))
    return (head + body + opts +
            f"Which option is the true next frame? Answer with a single letter ({', '.join(labels)}).")


def build_ffs_prompt_noaction(example: SelectionExample) -> str:
    n, k = len(example.context_paths), len(example.candidate_paths)
    labels = [chr(ord("A") + i) for i in range(k)]
    head = "".join(f"Frame {i + 1}: <image>\n" for i in range(n))
    body = f"These are {n} consecutive first-person camera frames from an egocentric kitchen video, in order.\n"
    opts = "".join(f"Option {labels[i]}: <image>\n" for i in range(k))
    return (head + body + opts +
            f"Which option is the true next frame? Answer with a single letter ({', '.join(labels)}).")


def _build_ffs_split(anchors, variant):
    build_prompt = build_ffs_prompt_action if variant == "action" else build_ffs_prompt_noaction
    rows, manifest = [], []
    for idx, ex in enumerate(anchors):
        images = list(ex.context_paths) + list(ex.candidate_paths)
        prompt = build_prompt(ex)
        rows.append({
            "id": idx, "task_type": "understanding", "image": images,
            "conversations": [{"from": "human", "value": prompt},
                               {"from": "gpt", "value": ex.answer_label}],
        })
        manifest.append(ex.to_dict())
    return rows, manifest


def build_ffs():
    anchors_dir = Path(OUT) / "anchors"
    for variant in ("action", "noaction"):
        task_dir = Path(OUT) / f"epic_ffs_{variant}"
        task_dir.mkdir(parents=True, exist_ok=True)
        lengths = {}
        for split in ("train", "eval"):
            anchors = read_jsonl(anchors_dir / f"anchors_{split}.jsonl", SelectionExample.from_dict)
            rows, manifest = _build_ffs_split(anchors, variant)
            with open(task_dir / f"epic_ffs_{variant}_{split}.jsonl", "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            write_jsonl(task_dir / f"manifest_{split}.jsonl", manifest)
            lengths[split] = len(rows)
            print(f"[ffs:{variant}] {split}: {len(rows)} rows -> {task_dir}/epic_ffs_{variant}_{split}.jsonl")

        meta_dir = Path(DATA) / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        json.dump({f"epic_ffs_{variant}": {
            "root": FRAMES_ROOT,
            "annotation": str(task_dir / f"epic_ffs_{variant}_train.jsonl"),
            "data_augment": False, "max_dynamic_patch": CONTEXT_SIZE + NUM_CANDIDATES,
            "repeat_time": 1, "length": lengths["train"], "task_type": "understanding"}},
            open(meta_dir / f"epic_ffs_{variant}_meta.json", "w"), indent=2)
        json.dump({f"epic_ffs_{variant}_eval": {
            "root": FRAMES_ROOT,
            "annotation": str(task_dir / f"epic_ffs_{variant}_eval.jsonl"),
            "data_augment": False, "max_dynamic_patch": CONTEXT_SIZE + NUM_CANDIDATES,
            "repeat_time": 1, "length": lengths["eval"], "task_type": "understanding"}},
            open(meta_dir / f"epic_ffs_{variant}_eval_meta.json", "w"), indent=2)

        with open(task_dir / "VERSION", "w") as f:
            f.write(
                "version: v1\n"
                "task: next-frame selection (understanding, 4-option MCQ)\n"
                f"action_variant: {variant}\n"
                "dataset: EPIC-KITCHENS-100 validation split, 116 videos downloaded for "
                "Evaluation/WorldPrediction (source MP4s live only on torrnode11 local SSD)\n"
                f"sample_fps: {SAMPLE_FPS}, context_size: {CONTEXT_SIZE}, horizon: {HORIZON} "
                f"grid-step ({HORIZON / SAMPLE_FPS:.2f}s), stride: {STRIDE} grid-steps "
                f"({STRIDE / SAMPLE_FPS:.2f}s) between anchor windows\n"
                f"split: {lengths['train']} train + {lengths['eval']} eval "
                f"(video-disjoint, held out {EVAL_HOLDOUT_VIDEOS} whole videos for eval)\n"
                f"split_seed: {SPLIT_SEED}, anchor_seed: {ANCHOR_SEED}\n"
                "candidates: temporal-distance hard negatives (near_wrong_state / "
                "mid_wrong_state / same_video_distractor) mined from OTHER anchor-eligible "
                "transitions in the same video (no pose available, unlike NWM/RECON)\n"
                "action_text: real EPIC-100 narration (verb+noun, e.g. 'open fridge') whose "
                "[start_frame, stop_frame] overlaps the transition window; anchors without "
                "an overlapping narration are excluded, so action/noaction share IDENTICAL "
                "context/candidates/answers (only the prompt text differs)\n"
                "created: 2026-08-02\n"
                "spec: ../EPIC_SSL_DATA_SPEC.md\n"
            )
    print("[ffs] done")


GEN_GPT_ANSWER = "The next frame should look like this: <img>"


def build_gen_prompt_action(n: int, action_text: str) -> str:
    head = "".join("<image>\n" for _ in range(n))
    return (head + f"These are {n} consecutive first-person camera frames from an "
            f"egocentric kitchen video. Action being performed after the last frame: "
            f"{action_text}\nObserve these frames and, given the action, produce the "
            f"next frame of the video.")


def build_gen_prompt_noaction(n: int) -> str:
    head = "".join("<image>\n" for _ in range(n))
    return (head + f"These are {n} consecutive first-person camera frames from an "
            f"egocentric kitchen video. Observe these frames and produce the next "
            f"frame of the video.")


def _build_gen_split(anchors, variant):
    rows, manifest = [], []
    for idx, ex in enumerate(anchors):
        n = len(ex.context_paths)
        target = ex.candidate_paths[ex.answer_index]
        prompt = (build_gen_prompt_action(n, ex.action_text) if variant == "action"
                  else build_gen_prompt_noaction(n))
        rows.append({
            "id": idx, "task_type": "imgen",
            "image": list(ex.context_paths), "target_image": target,
            "conversations": [{"from": "human", "value": prompt},
                               {"from": "gpt", "value": GEN_GPT_ANSWER}],
        })
        manifest.append(ex.to_dict())
    return rows, manifest


def build_gen():
    anchors_dir = Path(OUT) / "anchors"
    for variant in ("action", "noaction"):
        task_dir = Path(OUT) / f"epic_gen_{variant}"
        task_dir.mkdir(parents=True, exist_ok=True)
        lengths = {}
        for split in ("train", "eval"):
            anchors = read_jsonl(anchors_dir / f"anchors_{split}.jsonl", SelectionExample.from_dict)
            rows, manifest = _build_gen_split(anchors, variant)
            with open(task_dir / f"epic_gen_{variant}_{split}.jsonl", "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            write_jsonl(task_dir / f"manifest_{split}.jsonl", manifest)
            lengths[split] = len(rows)
            print(f"[gen:{variant}] {split}: {len(rows)} rows -> {task_dir}/epic_gen_{variant}_{split}.jsonl")

        meta_dir = Path(DATA) / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        json.dump({f"epic_gen_{variant}": {
            "root": FRAMES_ROOT,
            "annotation": str(task_dir / f"epic_gen_{variant}_train.jsonl"),
            "data_augment": False, "max_dynamic_patch": CONTEXT_SIZE,
            "repeat_time": 1, "length": lengths["train"], "task_type": "imgen"}},
            open(meta_dir / f"epic_gen_{variant}_meta.json", "w"), indent=2)
        json.dump({f"epic_gen_{variant}_eval": {
            "root": FRAMES_ROOT,
            "annotation": str(task_dir / f"epic_gen_{variant}_eval.jsonl"),
            "data_augment": False, "max_dynamic_patch": CONTEXT_SIZE,
            "repeat_time": 1, "length": lengths["eval"], "task_type": "imgen"}},
            open(meta_dir / f"epic_gen_{variant}_eval_meta.json", "w"), indent=2)

        with open(task_dir / "VERSION", "w") as f:
            f.write(
                "version: v1\n"
                "task: next-frame generation (imgen)\n"
                f"action_variant: {variant}\n"
                "dataset: EPIC-KITCHENS-100 validation split, same internal train/eval "
                "video carve-out and anchors as epic_ffs_* (gen target == FFS true frame)\n"
                f"sample_fps: {SAMPLE_FPS}, context_size: {CONTEXT_SIZE}, horizon: {HORIZON} "
                f"grid-step ({HORIZON / SAMPLE_FPS:.2f}s), stride: {STRIDE} grid-steps "
                f"({STRIDE / SAMPLE_FPS:.2f}s) between anchor windows\n"
                f"split: {lengths['train']} train + {lengths['eval']} eval "
                "(same anchors as epic_ffs_*, correct-action condition only)\n"
                f"split_seed: {SPLIT_SEED}, anchor_seed: {ANCHOR_SEED}\n"
                "note: epic_gen_action and epic_gen_noaction share IDENTICAL context/target "
                "images (same anchors file) -- only the human-turn prompt text differs\n"
                "created: 2026-08-02\n"
                "spec: ../EPIC_SSL_DATA_SPEC.md\n"
            )
    print("[gen] done")


def build_dual():
    import sys
    if DUAL_TOOL_DIR not in sys.path:
        sys.path.insert(0, DUAL_TOOL_DIR)
    import make_nwm_dual_ssl as dual_tool  # fully dataset-agnostic converter

    for variant in ("action", "noaction"):
        src_dir = Path(OUT) / f"epic_ffs_{variant}"
        dst_dir = Path(OUT) / f"epic_dual_{variant}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        lengths = {}
        for split in ("train", "eval"):
            src = src_dir / f"epic_ffs_{variant}_{split}.jsonl"
            dst = dst_dir / f"epic_dual_{variant}_{split}.jsonl"
            n = dual_tool.convert_split(src, dst, n_ctx=CONTEXT_SIZE)
            lengths[split] = n
            print(f"[dual:{variant}] {split}: {n} rows -> {dst}")

        meta_dir = Path(DATA) / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        json.dump({f"epic_dual_{variant}": {
            "root": FRAMES_ROOT,
            "annotation": str(dst_dir / f"epic_dual_{variant}_train.jsonl"),
            "data_augment": False, "max_dynamic_patch": CONTEXT_SIZE + NUM_CANDIDATES,
            "repeat_time": 1, "length": lengths["train"], "task_type": "imgen"}},
            open(meta_dir / f"epic_dual_{variant}_meta.json", "w"), indent=2)
        json.dump({f"epic_dual_{variant}_eval": {
            "root": FRAMES_ROOT,
            "annotation": str(dst_dir / f"epic_dual_{variant}_eval.jsonl"),
            "data_augment": False, "max_dynamic_patch": CONTEXT_SIZE + NUM_CANDIDATES,
            "repeat_time": 1, "length": lengths["eval"], "task_type": "imgen"}},
            open(meta_dir / f"epic_dual_{variant}_eval_meta.json", "w"), indent=2)

        with open(dst_dir / "VERSION", "w") as f:
            f.write(
                "version: v1\n"
                "task: dual FFS-selection + generation (imgen, imgen_form=multimodal)\n"
                f"action_variant: {variant}\n"
                f"converted_from: epic_ffs_{variant}_{{train,eval}}.jsonl (1:1, via "
                "Model_Related/InternVLU/InternVL/internvl_chat/tools/make_nwm_dual_ssl.py, "
                "reused UNMODIFIED from the NWM pipeline -- it's dataset-agnostic, only "
                "reads the FFS row's 8-image list + answer letter, N_CTX=4 matches both)\n"
                f"split: {lengths['train']} train + {lengths['eval']} eval\n"
                "cond_image: last context frame (ctx4). Unlike NWM (horizon==stride==8, "
                "which chains consecutive sliding windows so ~89% of rows pixel-duplicate "
                "an option), Epic's horizon=1 != stride=8 means anchor windows do NOT "
                "chain -- see EPIC_SSL_DATA_SPEC.md for the measured cond_image/option "
                "collision rate on this dataset.\n"
                "is_option: [0,0,0,0,1,1,1,1] -- read by the trainer to build "
                "option_context_mask; masking of option images out of the GENERATION "
                "conditioning only happens if --gen_no_leak is passed at train time "
                "(default False = leak ON). Do NOT pass --gen_no_leak to keep FFS option "
                "images unmasked for generation.\n"
                "created: 2026-08-03\n"
                "spec: ../EPIC_SSL_DATA_SPEC.md\n"
            )
    print("[dual] done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                     choices=["manifest", "splits", "anchors", "ffs", "gen", "dual"])
    args = ap.parse_args()
    if args.stage == "manifest":
        build_video_manifest()
    elif args.stage == "splits":
        build_trajectory_splits()
    elif args.stage == "anchors":
        build_anchors()
    elif args.stage == "ffs":
        build_ffs()
    elif args.stage == "gen":
        build_gen()
    elif args.stage == "dual":
        build_dual()


if __name__ == "__main__":
    main()
