"""Builder for Panda-70M next-frame SSL data (selection FFS + generation),
for InternVL-U SFT. Mirrors data/scripts/build_epic_ssl.py's structure (see
data/specs/EPIC_SSL_DATA_SPEC.md) but adapted for a fundamentally different source
shape: EPIC has 116 long raw videos with real narration annotations; Panda-70M
gives SHORT pre-clipped segments (already trimmed per-caption by the Panda-70M
curators, median ~6.5s, no per-frame action labels). A 5000-clip download was
attempted but YouTube bot-blocked this node's IP partway through (see the
TARGET_TRAIN/TARGET_EVAL note below) -- only 1,073 clips actually downloaded,
so TARGET_TRAIN/TARGET_EVAL/EVAL_HOLDOUT_CLIPS/PER_CLIP_CAP_* are scaled down
from the original 4000/300 design to what that pool can honestly support.
Consequences of the short-clip source shape (independent of the download
shortfall):

  - No narration matching / no action variant at all. Only "noaction" is
    built: no per-frame instruction text, pure visual next-frame prediction.
  - ~1 anchor per clip on average (occasionally more, capped), instead of
    EPIC's many-anchors-per-video -- so the clip pool itself (not
    anchors-per-video) is the thing subsampled/capped to reach the target.
  - Hard negatives can't be mined within a clip (usually only 1 transition
    per clip exists) -- replaced by CROSS-CLIP random negatives, drawn from
    other clips' anchor targets in the same split. The dual converter
    (make_nwm_dual_ssl.py) is negative-provenance-agnostic, confirmed by
    inspection, so this substitution is safe.
  - Downloaded clips live on THIS node's local SSD only (video2dataset output,
    /scratch/local/ssd/junlin/data/datasets/panda70m/<subset>/), analogous to EPIC's
    node-pinning gotcha. The "manifest"/"anchors" stages must run on the node
    that did the download. Extracted frame jpgs go to network storage
    (/scratch/network/ssd/junlin/panda70m_ssl_frames/) so training can happen
    from any node.

Same leak-free design choice as EPIC (not NWM): horizon=1 != stride=8, so
anchor windows within a clip never chain, keeping dual's cond_image
pixel-disjoint from all 4 MCQ options.

Stages (run with --stage, in order):
  manifest -- probe fps/frame_count for every downloaded clip via cv2 (mirrors
              EPIC's cv2-over-decord choice -- decord hung unpredictably on
              this cluster). Reads the video2dataset .json sidecars for
              status/caption/provenance. Writes data/datasets/panda70m_ssl/clip_manifest.json.
  splits   -- clip-level split (analogous to EPIC's video-level split):
              shuffle all clips with status=="success" and duration >= the
              minimum needed for 1 anchor, hold out EVAL_HOLDOUT_CLIPS whole
              clips for eval, rest for train.
              (data/datasets/panda70m_ssl/splits/{train,eval}_clips.txt)
  anchors  -- compile transitions (context frames -> next frame) at the same
              sampling grid as EPIC (sample_fps=2.0, context_size=4, horizon=1,
              stride=8 grid-steps WITHIN a clip), mine cross-clip random hard
              negatives for 4-way MCQ candidates, subsample to TARGET_TRAIN +
              TARGET_EVAL anchors, extract only the frames actually used via
              cv2 -> jpg (data/datasets/panda70m_ssl/anchors/anchors_{train,eval}.jsonl)
  ffs      -- build panda3dsr_ffs_noaction_{train,eval}.jsonl (task_type
              "understanding", 4-way MCQ). Built only as the internal source
              for "dual" (mirrors EPIC/NWM's pattern) -- not intended as a
              standalone trained task per the user's request.
  gen      -- build panda3dsr_gen_noaction_{train,eval}.jsonl (task_type
              "imgen", next-frame generation)
  dual     -- convert panda3dsr_ffs_noaction_{train,eval}.jsonl 1:1 into
              panda3dsr_dual_noaction_{train,eval}.jsonl via the SAME
              dataset-agnostic converter EPIC/NWM use
              (Model_Related/InternVLU/InternVL/internvl_chat/tools/
              make_nwm_dual_ssl.py, reused unmodified)

Usage (must run on the node that downloaded the clips, for manifest/anchors):
  python data/scripts/build_panda70m_ssl.py --stage manifest
  python data/scripts/build_panda70m_ssl.py --stage splits
  python data/scripts/build_panda70m_ssl.py --stage anchors
  python data/scripts/build_panda70m_ssl.py --stage ffs
  python data/scripts/build_panda70m_ssl.py --stage gen
  python data/scripts/build_panda70m_ssl.py --stage dual
"""
from __future__ import annotations

import argparse
import json
import random
import socket
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DATA = "/scratch/network/ssd2/junlin/ssl_mllm/data"
# v2 (2026-08-30): separate output tree from the v1 pipeline (data/datasets/panda70m_ssl/,
# 1500 anchors, 3dsrbench-filtered, video2dataset-downloaded) -- v1 left untouched
# as historical reference, not superseded in place. v2 sources clips via a plain
# random sample off training_10m.csv (no content filter) and a custom yt-dlp
# downloader; see build_panda70m_sample_random.py / build_panda70m_ytdlp_download.py.
OUT = f"{DATA}/panda70m_ssl_v2"
DUAL_TOOL_DIR = ("/scratch/network/ssd2/junlin/ssl_mllm/Model_Related/InternVLU/"
                  "InternVL/internvl_chat/tools")
LOCAL_CLIP_ROOT = "/scratch/local/ssd/junlin/data/datasets/panda70m/ytdlp_merged"
FRAMES_ROOT = "/scratch/network/ssd/junlin/panda70m_ssl_v2_frames"
CLIP_MANIFEST = f"{OUT}/clip_manifest.json"
DOWNLOAD_NODE = "torrnode11"

SAMPLE_FPS = 2.0          # working "grid" -- context frames are 0.5s apart
CONTEXT_SIZE = 3          # 3 context frames, spanning 1.0s -- matches VBVR's 3-frame
                            # convention (was 4; changed together with the v2 rescale below)
HORIZON = 1                # target = 1 grid-step (0.5s) after the last context frame
STRIDE = 4                 # grid-steps between anchor windows WITHIN a clip (2s); was 8,
                            # halved 2026-08-31 to cut the YouTube download volume needed --
                            # measured on the first 712 downloaded clips: STRIDE=8 gave 2587
                            # anchors (3.73/valid-clip), STRIDE=4 gives 4851 (7.0/valid-clip,
                            # 1.88x) from the SAME clips, no re-download needed. Tradeoff:
                            # windows from the same clip are more temporally correlated.
                            # horizon != stride still keeps windows from chaining (leak-free
                            # cond_image in dual, same reasoning as EPIC vs NWM)
NUM_CANDIDATES = 4
SPLIT_SEED = 42
ANCHOR_SEED = 0
EVAL_HOLDOUT_CLIPS = 500   # of the valid clip pool, held out whole for eval (S3/MCQ only --
                            # see TARGET_EVAL: S0/S1/S2 gen data takes NO eval holdout)
TARGET_TRAIN = 25000
TARGET_EVAL = 200
PER_CLIP_CAP_TRAIN = 3
PER_CLIP_CAP_EVAL = 3
# NOTE (2026-08-30, v2 rescale): earlier v1 numbers (TARGET_TRAIN=1500, CONTEXT_SIZE=4)
# were sized around a botched 5000-clip video2dataset download that got YouTube-bot-
# blocked at 21.5% success. v2 sources clips via a from-scratch random sample off
# panda70m_training_10m.csv (build_panda70m_sample_random.py, NO content filter -- the
# 3dsrbench/COCO-noun filtering pipeline is dropped entirely) + a from-scratch yt-dlp
# downloader (build_panda70m_ytdlp_download.py, replaces video2dataset) with real
# concurrency control this time. Target: 25k train anchors (== the reduced VBVR-side
# target: 5 tasks x 5000/task) for S0/S1/S2 (imgen, no eval split needed -- VBVR-Bench-
# style eval is out of scope for the "next frame" settings per project decision), 200
# eval anchors for S3 (MCQ+gen dual) only. EVAL_HOLDOUT_CLIPS=500 is a big buffer over
# the ~200*PER_CLIP_CAP_EVAL~=67 clips actually needed, cheap given the much larger
# clip pool this time.


def _assert_on_download_node():
    host = socket.gethostname()
    if host != DOWNLOAD_NODE:
        raise RuntimeError(
            f"Panda-70M downloaded clips only exist on {DOWNLOAD_NODE}'s local SSD "
            f"({LOCAL_CLIP_ROOT}); this node is {host!r}. The 'manifest'/'anchors' "
            f"stages (touch video files) must run on {DOWNLOAD_NODE}.")


@dataclass(frozen=True)
class TransitionRecord:
    sample_id: str
    clip_id: str
    context_grid: list[int]
    target_grid: int
    context_native: list[int]
    target_native: int
    caption: str  # provenance only, NOT used in noaction prompts

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


@dataclass(frozen=True)
class SelectionExample:
    example_id: str
    transition_id: str
    clip_id: str
    context_paths: list[str]
    caption: str
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


def _iter_downloaded_samples():
    """Yield (clip_id, mp4_path, json_meta) for every clip build_panda70m_ytdlp_download.py
    wrote (flat output dir, one .mp4 + .json sidecar per SUCCESSFUL clip only --
    unlike video2dataset, that downloader deletes partial/failed mp4s and skips the
    sidecar entirely on failure, so every yielded row here already has status=='success')."""
    root = Path(LOCAL_CLIP_ROOT)
    for json_path in sorted(root.glob("*.json")):
        meta = json.loads(json_path.read_text())
        clip_id = json_path.stem
        mp4_path = json_path.with_suffix(".mp4")
        yield clip_id, mp4_path, meta


def build_clip_manifest():
    """Probe fps/frame_count for every successfully-downloaded clip via cv2
    (mirrors EPIC's cv2-over-decord choice; decord hung unpredictably on this
    cluster's videos). Reads build_panda70m_ytdlp_download.py's per-clip .json
    sidecar for status/caption/provenance. Run once, cached to disk."""
    _assert_on_download_node()
    import cv2

    manifest = {}
    n_total = n_success = n_failed = n_probe_fail = 0
    for clip_id, mp4_path, meta in _iter_downloaded_samples():
        n_total += 1
        if meta.get("status") != "success" or not mp4_path.exists():
            n_failed += 1
            continue
        cap = cv2.VideoCapture(str(mp4_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if not fps or not n_frames:
            n_probe_fail += 1
            continue
        manifest[clip_id] = {
            "path": str(mp4_path), "fps": fps, "num_frames": n_frames,
            "duration_sec": n_frames / fps,
            "caption": meta.get("caption", ""),
            "videoID": meta.get("video_id", ""),
            "matching_score": meta.get("matching_score"),
        }
        n_success += 1
        if n_success % 500 == 0:
            print(f"[manifest] {n_success} valid clips probed so far...", flush=True)

    Path(OUT).mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(CLIP_MANIFEST, "w"), indent=2)
    print(f"[manifest] {n_total} downloaded samples: {n_success} valid, "
          f"{n_failed} failed_download, {n_probe_fail} failed_fps_probe "
          f"-> {CLIP_MANIFEST}")


def build_clip_splits():
    manifest = json.load(open(CLIP_MANIFEST))
    min_duration = (CONTEXT_SIZE - 1 + HORIZON) / SAMPLE_FPS
    valid_ids = sorted(cid for cid, info in manifest.items()
                        if info["duration_sec"] >= min_duration)
    print(f"[splits] {len(valid_ids)}/{len(manifest)} clips >= {min_duration:.1f}s "
          f"(minimum for 1 anchor)")

    rng = random.Random(SPLIT_SEED)
    rng.shuffle(valid_ids)
    eval_ids = sorted(valid_ids[:EVAL_HOLDOUT_CLIPS])
    train_ids = sorted(valid_ids[EVAL_HOLDOUT_CLIPS:])
    assert not (set(eval_ids) & set(train_ids))

    out_dir = Path(OUT) / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_clips.txt").write_text("\n".join(train_ids) + "\n")
    (out_dir / "eval_clips.txt").write_text("\n".join(eval_ids) + "\n")
    print(f"[splits] train={len(train_ids)} eval={len(eval_ids)} clips -> {out_dir}")


def compile_transitions(clip_ids, manifest):
    records = []
    for clip_id in clip_ids:
        info = manifest[clip_id]
        fps, n_native = info["fps"], info["num_frames"]
        native_stride = max(1, round(fps / SAMPLE_FPS))
        n_grid = (n_native - 1) // native_stride + 1
        for current in range(CONTEXT_SIZE - 1, n_grid - HORIZON, STRIDE):
            context_grid = list(range(current - CONTEXT_SIZE + 1, current + 1))
            context_native = [g * native_stride for g in context_grid]
            target_grid = current + HORIZON
            target_native = target_grid * native_stride
            if target_native >= n_native or context_native[-1] >= n_native:
                continue
            records.append(TransitionRecord(
                f"panda3dsr:{clip_id}:{current}:{target_grid}", clip_id,
                context_grid, target_grid, context_native, target_native,
                info.get("caption", "")))
    return records


def _same_clip_negatives(target, clip_pool, count):
    """Same-clip temporal-distance hard negatives, ported from
    build_panda70m_epic_ssl.py's _negatives() -- was cross-clip-only here
    (see _cross_clip_negatives below) because at the OLD grid
    (CONTEXT_SIZE=4, ~1 anchor/clip) same-clip negatives usually weren't
    available. At the current grid (CONTEXT_SIZE=3, STRIDE=4) clips average
    ~6.5-7 transitions each, so same-clip mining is now the norm, not the
    exception -- and it's a strictly harder, more meaningful negative
    (visually similar, same continuous shot, forces real temporal-order
    sensitivity instead of "which clip does this even belong to").
    near_wrong_state = closest by grid distance (hardest); mid_wrong_state =
    next closest; same_clip_distractor = temporally distant ones (easiest).
    `clip_pool` = all OTHER transitions from target's own clip_id."""
    seen, unused = {target.target_native}, []
    for t in clip_pool:
        if t.target_native in seen:
            continue
        seen.add(t.target_native)
        unused.append(t)
    if len(unused) < count:
        return []
    unused.sort(key=lambda t: abs(t.target_grid - target.target_grid))
    selected = [(unused[0], "near_wrong_state")]
    if len(unused) > 1:
        selected.append((unused[1], "mid_wrong_state"))
    remaining = unused[2:]
    remaining.sort(key=lambda t: -abs(t.target_grid - target.target_grid))
    selected += [(t, "same_clip_distractor") for t in remaining[: count - len(selected)]]
    return selected[:count]


def _cross_clip_negatives(target, pool, count, rng):
    """Cross-clip random hard negatives -- fallback only, for the minority of
    clips that don't yield enough same-clip transitions to fill the MCQ
    (e.g. very short clips still averaging ~1 transition). Draw `count`
    distinct negatives from OTHER clips' transitions in the same split. The
    dual converter (make_nwm_dual_ssl.py) is blind to negative provenance, so
    this substitution doesn't affect the dual-construction step at all."""
    candidates = [x for x in pool if x.clip_id != target.clip_id]
    if len(candidates) < count:
        return []
    picked = rng.sample(candidates, count)
    return [(x, "cross_clip_random") for x in picked]


def build_selection_examples(transitions, seed=0):
    rng = random.Random(seed)
    by_clip = defaultdict(list)
    for t in transitions:
        by_clip[t.clip_id].append(t)

    examples = []
    for record in transitions:
        same_clip_pool = [t for t in by_clip[record.clip_id] if t is not record]
        negatives = _same_clip_negatives(record, same_clip_pool, NUM_CANDIDATES - 1)
        if len(negatives) < NUM_CANDIDATES - 1:
            negatives = _cross_clip_negatives(record, transitions, NUM_CANDIDATES - 1, rng)
        if len(negatives) < NUM_CANDIDATES - 1:
            continue
        candidates = [(record.target_native, "ground_truth")] + \
                     [(x.target_native, kind) for x, kind in negatives]
        candidate_clip_ids = [record.clip_id] + [x.clip_id for x, _ in negatives]
        shuffle_rng = random.Random(f"{seed}:{record.sample_id}")
        order = list(range(len(candidates)))
        shuffle_rng.shuffle(order)
        candidates = [candidates[i] for i in order]
        candidate_clip_ids = [candidate_clip_ids[i] for i in order]
        natives, kinds = map(list, zip(*candidates))
        examples.append(SelectionExample(
            f"selection:{record.sample_id}", record.sample_id, record.clip_id,
            [_frame_path(record.clip_id, n) for n in record.context_native],
            record.caption,
            [_frame_path(cid, n) for cid, n in zip(candidate_clip_ids, natives)],
            kinds, kinds.index("ground_truth"),
            {"context_native": record.context_native, "target_native": record.target_native,
             "candidate_natives": natives, "candidate_clip_ids": candidate_clip_ids}))
    return examples


def _frame_path(clip_id, native_idx):
    return f"{FRAMES_ROOT}/{clip_id}/{native_idx}.jpg"


def _subsample_capped(examples, target_n, cap, seed):
    rng = random.Random(seed)
    order = list(examples)
    rng.shuffle(order)
    counts, selected, leftover = {}, [], []
    for ex in order:
        c = counts.get(ex.clip_id, 0)
        if c < cap:
            selected.append(ex)
            counts[ex.clip_id] = c + 1
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
    cv2 random-access seek+read (same reasoning as EPIC: decord hung
    unpredictably in this environment). Must run on the download node."""
    _assert_on_download_node()
    import cv2
    from PIL import Image

    needed = defaultdict(set)
    for ex in anchors:
        for p in ex.context_paths + ex.candidate_paths:
            clip_id = Path(p).parent.name
            native_idx = int(Path(p).stem)
            needed[clip_id].add(native_idx)

    manifest = json.load(open(CLIP_MANIFEST))
    total_written, total_skipped = 0, 0
    for i, (clip_id, indices) in enumerate(sorted(needed.items())):
        out_dir = Path(FRAMES_ROOT) / clip_id
        out_dir.mkdir(parents=True, exist_ok=True)
        todo = sorted(idx for idx in indices if not (out_dir / f"{idx}.jpg").exists())
        if not todo:
            total_skipped += len(indices)
            continue
        path = manifest[clip_id]["path"]
        cap = cv2.VideoCapture(path)
        for idx in todo:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                print(f"[frames] WARNING: failed to read {clip_id}/{idx}, skipping")
                continue
            Image.fromarray(frame[:, :, ::-1]).save(out_dir / f"{idx}.jpg", quality=95)
        cap.release()
        total_written += len(todo)
        total_skipped += len(indices) - len(todo)
        if (i + 1) % 500 == 0:
            print(f"[frames] {i + 1}/{len(needed)} clips processed "
                  f"(wrote {total_written}, skipped {total_skipped})", flush=True)
    print(f"[frames] total: wrote {total_written}, skipped (already existed) {total_skipped}")


def build_anchors():
    manifest = json.load(open(CLIP_MANIFEST))
    splits_dir = Path(OUT) / "splits"
    anchors_dir = Path(OUT) / "anchors"
    anchors_dir.mkdir(parents=True, exist_ok=True)

    for split, clip_file, target_n, cap in (
        ("train", splits_dir / "train_clips.txt", TARGET_TRAIN, PER_CLIP_CAP_TRAIN),
        ("eval", splits_dir / "eval_clips.txt", TARGET_EVAL, PER_CLIP_CAP_EVAL),
    ):
        clip_ids = [x.strip() for x in clip_file.read_text().splitlines() if x.strip()]
        transitions = compile_transitions(clip_ids, manifest)
        print(f"[anchors] {split}: {len(transitions)} raw transitions "
              f"across {len(clip_ids)} clips")
        examples = build_selection_examples(transitions, seed=ANCHOR_SEED)
        print(f"[anchors] {split}: {len(examples)} selection examples "
              f"(candidate mining succeeded)")
        anchors = _subsample_capped(examples, target_n, cap, ANCHOR_SEED)
        write_jsonl(anchors_dir / f"anchors_{split}.jsonl", anchors)
        clips_used = len({a.clip_id for a in anchors})
        print(f"[anchors] {split}: wrote {len(anchors)} anchors ({clips_used} distinct clips) "
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


def build_ffs_prompt_noaction(example: SelectionExample, has_caption: bool = True) -> str:
    n, k = len(example.context_paths), len(example.candidate_paths)
    labels = [chr(ord("A") + i) for i in range(k)]
    head = "".join(f"Frame {i + 1}: <image>\n" for i in range(n))
    body = f"These are {n} consecutive frames from a real-world video, in order.\n"
    if has_caption and example.caption:
        body = f"Video description: {example.caption.strip()}\n\n" + body
    opts = "".join(f"Option {labels[i]}: <image>\n" for i in range(k))
    return (head + body + opts +
            f"Which option is the true next frame? Answer with a single letter ({', '.join(labels)}).")


def _build_ffs_split(anchors, has_caption: bool = True):
    rows, manifest = [], []
    for idx, ex in enumerate(anchors):
        images = list(ex.context_paths) + list(ex.candidate_paths)
        prompt = build_ffs_prompt_noaction(ex, has_caption=has_caption)
        rows.append({
            "id": idx, "task_type": "understanding", "image": images,
            "conversations": [{"from": "human", "value": prompt},
                               {"from": "gpt", "value": ex.answer_label}],
        })
        manifest.append(ex.to_dict())
    return rows, manifest


def build_ffs():
    anchors_dir = Path(OUT) / "anchors"
    task_dir = Path(OUT) / "panda3dsr_ffs_noaction"
    task_dir.mkdir(parents=True, exist_ok=True)
    lengths = {}
    for split in ("train", "eval"):
        anchors = read_jsonl(anchors_dir / f"anchors_{split}.jsonl", SelectionExample.from_dict)
        rows, manifest = _build_ffs_split(anchors)
        with open(task_dir / f"panda3dsr_ffs_noaction_{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        write_jsonl(task_dir / f"manifest_{split}.jsonl", manifest)
        lengths[split] = len(rows)
        print(f"[ffs] {split}: {len(rows)} rows -> {task_dir}/panda3dsr_ffs_noaction_{split}.jsonl")

    meta_dir = Path(DATA) / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    json.dump({"panda3dsr_ffs_noaction": {
        "root": FRAMES_ROOT,
        "annotation": str(task_dir / "panda3dsr_ffs_noaction_train.jsonl"),
        "data_augment": False, "max_dynamic_patch": CONTEXT_SIZE + NUM_CANDIDATES,
        "repeat_time": 1, "length": lengths["train"], "task_type": "understanding"}},
        open(meta_dir / "panda3dsr_ffs_noaction_meta.json", "w"), indent=2)
    json.dump({"panda3dsr_ffs_noaction_eval": {
        "root": FRAMES_ROOT,
        "annotation": str(task_dir / "panda3dsr_ffs_noaction_eval.jsonl"),
        "data_augment": False, "max_dynamic_patch": CONTEXT_SIZE + NUM_CANDIDATES,
        "repeat_time": 1, "length": lengths["eval"], "task_type": "understanding"}},
        open(meta_dir / "panda3dsr_ffs_noaction_eval_meta.json", "w"), indent=2)

    with open(task_dir / "VERSION", "w") as f:
        f.write(
            "version: v1\n"
            "task: next-frame selection (understanding, 4-option MCQ)\n"
            "note: built ONLY as the internal source for panda3dsr_dual_noaction "
            "(mirrors EPIC/NWM's pattern) -- not intended as a standalone trained "
            "task per the original request.\n"
            "dataset: Panda-70M (10M-clip metadata tier), caption-filtered to match "
            "3DSRBench's visual distribution (real-world photos, >=2 spatially "
            "separate objects) via data/scripts/build_panda70m_3dsrbench_filter.py; "
            "5000 clips attempted via video2dataset, but YouTube bot-blocked this "
            "node's IP partway through (see build_panda70m_ssl.py module docstring) "
            "-- only 1,073 clips (21.5%) actually downloaded, targets rescaled down\n"
            f"sample_fps: {SAMPLE_FPS}, context_size: {CONTEXT_SIZE}, horizon: {HORIZON} "
            f"grid-step ({HORIZON / SAMPLE_FPS:.2f}s), stride: {STRIDE} grid-steps "
            f"({STRIDE / SAMPLE_FPS:.2f}s) between anchor windows within a clip\n"
            f"split: {lengths['train']} train + {lengths['eval']} eval "
            f"(clip-disjoint, held out {EVAL_HOLDOUT_CLIPS} whole clips for eval)\n"
            f"split_seed: {SPLIT_SEED}, anchor_seed: {ANCHOR_SEED}\n"
            "candidates: cross-clip random hard negatives (no per-clip temporal "
            "diversity available -- clips are short, median ~6.5s, mostly 1 "
            "transition/clip -- unlike EPIC's same-video temporal-distance tiers)\n"
            "no action variant: Panda-70M clips have no per-frame action labels; "
            "only noaction (pure visual next-frame prediction) is built\n"
            "created: 2026-08-03\n"
        )
    print("[ffs] done")


GEN_GPT_ANSWER_FORWARD = "The next frame should look like this: <img>"
GEN_GPT_ANSWER_BACKWARD = "The preceding frame should look like this: <img>"
DIRECTIONS = ("forward", "backward")


def _pick_direction(direction_arg: str, rng: random.Random) -> str:
    if direction_arg == "mixed":
        return "backward" if rng.random() < 0.5 else "forward"
    if direction_arg not in DIRECTIONS:
        raise ValueError(f"unknown direction {direction_arg!r}")
    return direction_arg


def _direction_ctx_target_cond(ctx_paths: list[str], target_path: str, direction: str):
    """Mirrors VBVR's _direction_paths: panda70m anchors are the SAME shape as
    VBVR's (CONTEXT_SIZE context frames + 1 target = CONTEXT_SIZE+1 total
    points), so "backward" is the same free reinterpretation of the same
    points -- no extra frame ever needs to be extracted. forward:
    ctx=[c0..c_{n-1}] -> target=t, cond=c_{n-1}. backward: ctx=[c1..c_{n-1},t]
    -> target=c0, cond=c1."""
    if direction == "backward":
        ctx = list(ctx_paths[1:]) + [target_path]
        return ctx, ctx_paths[0], ctx_paths[1]
    return list(ctx_paths), target_path, ctx_paths[-1]


def build_gen_prompt(n: int, direction: str, caption: str | None) -> str:
    head = "".join("<image>\n" for _ in range(n))
    if direction == "backward":
        closing = (
            f"These are {n} consecutive frames from a real-world video, occurring "
            f"immediately AFTER the frame you must predict. Based on the direction and "
            f"rate of motion established across them, predict the frame that occurred "
            f"immediately BEFORE the first frame shown."
        )
    else:
        closing = (
            f"These are {n} consecutive frames from a real-world video. "
            f"Observe these frames and produce the next frame of the video."
        )
    if caption:
        return head + f"Video description: {caption.strip()}\n\n" + closing
    return head + closing


def _build_gen_split(anchors, direction_arg: str, has_caption: bool, seed: int, setting_label: str | None):
    rows, manifest = [], []
    for idx, ex in enumerate(anchors):
        rng = random.Random(f"{seed}:{ex.example_id}:direction")
        direction = _pick_direction(direction_arg, rng)
        target = ex.candidate_paths[ex.answer_index]
        ctx, tgt, cond = _direction_ctx_target_cond(list(ex.context_paths), target, direction)
        caption = ex.caption if has_caption else None
        prompt = build_gen_prompt(len(ctx), direction, caption)
        gpt_value = GEN_GPT_ANSWER_BACKWARD if direction == "backward" else GEN_GPT_ANSWER_FORWARD
        row = {
            "id": idx, "task_type": "imgen",
            "direction": direction, "has_caption": bool(has_caption),
            "image": ctx, "target_image": tgt, "cond_image": cond,
            "conversations": [{"from": "human", "value": prompt},
                               {"from": "gpt", "value": gpt_value}],
        }
        if setting_label:
            row["setting"] = setting_label
        rows.append(row)
        manifest.append(ex.to_dict())
    return rows, manifest


def build_gen(direction: str = "forward", has_caption: bool = False, setting_label: str | None = None,
              out_stem: str = "panda3dsr_gen_noaction", seed: int = 0):
    """S0/S1/S2 (all "gen"/imgen settings) take NO eval holdout -- per project
    decision, only S3 (MCQ) reserves eval anchors -- so this only ever reads
    anchors_train.jsonl, never anchors_eval.jsonl."""
    anchors_dir = Path(OUT) / "anchors"
    task_dir = Path(OUT) / out_stem
    task_dir.mkdir(parents=True, exist_ok=True)

    anchors = read_jsonl(anchors_dir / "anchors_train.jsonl", SelectionExample.from_dict)
    rows, manifest = _build_gen_split(anchors, direction, has_caption, seed, setting_label)
    jsonl_path = task_dir / f"{out_stem}_train.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_jsonl(task_dir / "manifest_train.jsonl", manifest)
    print(f"[gen:{setting_label or out_stem}] train: {len(rows)} rows -> {jsonl_path}")

    meta_dir = Path(DATA) / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / f"{out_stem}_meta.json"
    json.dump({out_stem: {
        "root": FRAMES_ROOT,
        "annotation": str(jsonl_path),
        "data_augment": False, "max_dynamic_patch": CONTEXT_SIZE,
        "repeat_time": 1, "length": len(rows), "task_type": "imgen"}},
        open(meta_path, "w"), indent=2)

    with open(task_dir / "VERSION", "w") as f:
        f.write(
            "version: v2\n"
            f"task: next-frame generation (imgen), setting={setting_label}\n"
            "dataset: Panda-70M v2 (random-sampled off training_10m.csv, no content "
            "filter -- see build_panda70m_sample_random.py), downloaded via "
            "build_panda70m_ytdlp_download.py\n"
            f"sample_fps: {SAMPLE_FPS}, context_size: {CONTEXT_SIZE}, horizon: {HORIZON} "
            f"grid-step ({HORIZON / SAMPLE_FPS:.2f}s), stride: {STRIDE} grid-steps "
            f"({STRIDE / SAMPLE_FPS:.2f}s) between anchor windows within a clip\n"
            f"direction: {direction}, has_caption: {has_caption}\n"
            f"split: {len(rows)} train, 0 eval (S0/S1/S2 take no eval holdout)\n"
            f"split_seed: {SPLIT_SEED}, anchor_seed: {ANCHOR_SEED}, direction_seed: {seed}\n"
        )
    print(f"[gen:{setting_label or out_stem}] meta -> {meta_path}")


def build_dual():
    import sys
    if DUAL_TOOL_DIR not in sys.path:
        sys.path.insert(0, DUAL_TOOL_DIR)
    import make_nwm_dual_ssl as dual_tool  # fully dataset-agnostic converter

    src_dir = Path(OUT) / "panda3dsr_ffs_noaction"
    dst_dir = Path(OUT) / "panda3dsr_dual_noaction"
    dst_dir.mkdir(parents=True, exist_ok=True)
    lengths = {}
    for split in ("train", "eval"):
        src = src_dir / f"panda3dsr_ffs_noaction_{split}.jsonl"
        dst = dst_dir / f"panda3dsr_dual_noaction_{split}.jsonl"
        n = 0
        with open(src) as fin, open(dst, "w") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                row = dual_tool.convert_row(json.loads(line), n, n_ctx=CONTEXT_SIZE)
                row["setting"] = "S3"
                row["eval_holdout"] = (split == "eval")
                fout.write(json.dumps(row) + "\n")
                n += 1
        lengths[split] = n
        print(f"[dual] {split}: {n} rows -> {dst}")

    meta_dir = Path(DATA) / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    json.dump({"panda3dsr_dual_noaction": {
        "root": FRAMES_ROOT,
        "annotation": str(dst_dir / "panda3dsr_dual_noaction_train.jsonl"),
        "data_augment": False, "max_dynamic_patch": CONTEXT_SIZE + NUM_CANDIDATES,
        "repeat_time": 1, "length": lengths["train"], "task_type": "imgen"}},
        open(meta_dir / "panda3dsr_dual_noaction_meta.json", "w"), indent=2)
    json.dump({"panda3dsr_dual_noaction_eval": {
        "root": FRAMES_ROOT,
        "annotation": str(dst_dir / "panda3dsr_dual_noaction_eval.jsonl"),
        "data_augment": False, "max_dynamic_patch": CONTEXT_SIZE + NUM_CANDIDATES,
        "repeat_time": 1, "length": lengths["eval"], "task_type": "imgen"}},
        open(meta_dir / "panda3dsr_dual_noaction_eval_meta.json", "w"), indent=2)

    with open(dst_dir / "VERSION", "w") as f:
        f.write(
            "version: v1\n"
            "task: dual FFS-selection + generation (imgen, imgen_form=multimodal)\n"
            "converted_from: panda3dsr_ffs_noaction_{train,eval}.jsonl (1:1, via "
            "Model_Related/InternVLU/InternVL/internvl_chat/tools/make_nwm_dual_ssl.py, "
            "reused UNMODIFIED -- it's dataset-agnostic, only reads the FFS row's "
            "8-image list + answer letter, N_CTX=4 matches)\n"
            f"split: {lengths['train']} train + {lengths['eval']} eval\n"
            "cond_image: last context frame (ctx4). horizon=1 != stride=8 means anchor "
            "windows within a clip never chain -- pixel-disjoint from all 4 options, "
            "same leak-free property EPIC achieved (unlike NWM's horizon==stride==8).\n"
            "is_option: [0,0,0,0,1,1,1,1] -- read by the trainer to build "
            "option_context_mask; masking of option images out of the GENERATION "
            "conditioning only happens if --gen_no_leak is passed at train time "
            "(default False = leak ON -> this IS the 'dual-leak' variant requested). "
            "Do NOT pass --gen_no_leak to keep FFS option images unmasked for generation.\n"
            "created: 2026-08-03\n"
        )
    print("[dual] done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                     choices=["manifest", "splits", "anchors", "ffs", "gen", "dual"])
    ap.add_argument("--direction", choices=["forward", "backward", "mixed"], default="forward",
                     help="gen stage only. forward: S0/S2. mixed: S1 (~50/50 forward/backward "
                          "per anchor, seeded). backward alone is available but not one of the "
                          "named S0-S3 settings.")
    ap.add_argument("--with-caption", action="store_true",
                     help="gen stage only. Include the clip's Panda-70M caption in the prompt "
                          "(S2). Default off (S0/S1).")
    ap.add_argument("--setting-label", type=str, default=None,
                     help="gen stage only. Free-text label stamped into each row's \"setting\" "
                          "field (e.g. S0/S1/S2), pure bookkeeping.")
    ap.add_argument("--out-stem", type=str, default="panda3dsr_gen_noaction",
                     help="gen stage only. Output dir/file/meta name stem -- pass a distinct "
                          "stem per setting (e.g. panda3dsr_gen_S0) so S0/S1/S2 don't clobber "
                          "each other.")
    ap.add_argument("--seed", type=int, default=0, help="gen stage only. Seeds the per-anchor "
                     "direction pick for --direction mixed.")
    args = ap.parse_args()
    if args.stage == "manifest":
        build_clip_manifest()
    elif args.stage == "splits":
        build_clip_splits()
    elif args.stage == "anchors":
        build_anchors()
    elif args.stage == "ffs":
        build_ffs()
    elif args.stage == "gen":
        build_gen(direction=args.direction, has_caption=args.with_caption,
                   setting_label=args.setting_label, out_stem=args.out_stem, seed=args.seed)
    elif args.stage == "dual":
        build_dual()


if __name__ == "__main__":
    main()
