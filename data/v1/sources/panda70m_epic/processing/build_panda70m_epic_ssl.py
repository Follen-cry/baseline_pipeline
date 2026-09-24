"""Builder for a Panda-70M-sourced next-frame SSL task (generation + dual
FFS-selection/generation leak), mirroring data/scripts/build_epic_ssl.py's structure
(EPIC_SSL_DATA_SPEC.md) but sourced from the filtered kitchen-manipulation
pool in data/datasets/panda70m/epickitchen_filter/stage2_judged.jsonl (see
data/scripts/build_panda70m_epickitchen_filter.py) instead of real EPIC-KITCHENS
video.

Only "gen" and "dual" are built (no "ffs" task is shipped/registered, though
an FFS-shaped intermediate is written to disk since "dual" is converted 1:1
from it, same as EPIC). Both are noaction-only: Panda-70M captions describe
a whole clip, not per-frame timed actions like EPIC-100's narrations, so
there is no action_text to condition on -- this also matches the explicit
ask for no instruction-conditioning text. "dual" IS the "dual-leak" variant:
the jsonl schema is identical regardless of leak/no-leak, the difference is
purely a training-time flag (--gen_no_leak) -- see build_dual()'s VERSION
note. Do NOT pass --gen_no_leak at train time to get leak-ON ("dual-leak").

Important honesty note (see conversation): Panda-70M is NOT actually
egocentric footage -- a manual frame-level spot check found only ~25-30% of
sampled clips have hands/object-dominant framing; most are third-person
YouTube cooking content (full body/face visible, tripod/static shots, text
overlays). Prompts here therefore do NOT claim "first-person"/"egocentric"
camera framing (unlike EPIC's prompts) -- just "a kitchen video".

Negative-mining adaptation (the other real difference from EPIC): EPIC mines
FFS hard negatives from OTHER narrated transitions in the SAME long video
(116 videos, many transitions each). Panda-70M clips are short and each
contributes exactly ONE anchor, so there is no "other transition in the same
video" pool. Adaptation: mine same-clip negatives from spare frames later in
the SAME clip when the clip is long enough (visually closest, same shot --
if anything HARDER than EPIC's same-participant-different-moment negatives);
fall back to cross-clip negatives (a frame from a different downloaded clip)
to fill any remaining slots. See _negatives_for_clip().

Stages (run with --stage, in order):
  download -- randomly sample N_DOWNLOAD_SAMPLE clips from the filtered pool
              (kitchen AND manipulation, desirable_filtering=='desirable'),
              download each clip's exact timestamp range via yt-dlp
              (--download-sections, stream-copy, no re-encode -- re-encoding
              via --force-keyframes-at-cuts SEGFAULTS the static ffmpeg
              binary here, confirmed), to node-local SSD (CLIPS_ROOT).
              Probes fps/frame_count via cv2 (decord is known to hang
              unpredictably on this cluster, see epic_kitchens_ssl_next_frame
              memory) into a resumable clip_manifest.json.
  anchors  -- for each manifest clip with enough frames, build one
              context(4)->target(1) transition at sample_fps=2.0, mine 3
              negatives per _negatives_for_clip(), split into 4000 train +
              300 eval (video-disjoint by construction: 1 anchor/clip),
              extract only the used frame jpgs (cv2 seek+read) to
              FRAMES_ROOT (network storage, readable from any node).
  gen      -- build panda70m_gen_noaction_{train,eval}.jsonl (task_type
              "imgen").
  dual     -- build an internal FFS-shaped jsonl, then convert 1:1 into
              panda70m_dual_noaction_{train,eval}.jsonl via the SAME
              dataset-agnostic converter NWM/EPIC use
              (Model_Related/InternVLU/InternVL/internvl_chat/tools/
              make_nwm_dual_ssl.py), reused unmodified.

Usage:
  python data/scripts/build_panda70m_epic_ssl.py --stage download
  python data/scripts/build_panda70m_epic_ssl.py --stage anchors
  python data/scripts/build_panda70m_epic_ssl.py --stage gen
  python data/scripts/build_panda70m_epic_ssl.py --stage dual
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DATA = "/scratch/network/ssd2/junlin/ssl_mllm/data"
OUT = f"{DATA}/panda70m_epic_ssl"
DUAL_TOOL_DIR = ("/scratch/network/ssd2/junlin/ssl_mllm/Model_Related/InternVLU/"
                  "InternVL/internvl_chat/tools")
SRC_JUDGED = f"{DATA}/panda70m/epickitchen_filter/stage2_judged.jsonl"
CLIPS_ROOT = "/scratch/local/ssd/junlin/data/datasets/panda70m_epic_ssl/clips"
FRAMES_ROOT = "/scratch/network/ssd/junlin/panda70m_epic_ssl_frames"
MANIFEST_PATH = f"{OUT}/clip_manifest.json"
FAILED_PATH = f"{OUT}/download_failed.json"

SAMPLE_FPS = 2.0
CONTEXT_SIZE = 3   # was 4; shrunk 2026-08-31, shared real-video-mix convention
HORIZON = 1
STRIDE = 4          # was CONTEXT_SIZE+HORIZON=5 (chaining); now denser (2.0s),
                     # matching the panda70m-v2 tuning (STRIDE 8->4 nearly doubled
                     # yield/clip) -- these are short clips, density matters more
                     # than avoiding window overlap here.
NUM_CANDIDATES = 4
PER_CLIP_CAP = 20  # cap anchors drawn from any single clip (a 54s clip could
                    # otherwise yield ~21 of them -- avoid one video dominating)
N_DOWNLOAD_SAMPLE = 700  # NWM-style dense multi-anchor-per-clip extraction
                          # needs far fewer distinct clips than 1-anchor/clip
                          # did -- measured ~10.5 anchors/clip at this
                          # stride on the first 70 downloaded clips, so 700
                          # gives ~4300 needed anchors a comfortable margin.
TARGET_TRAIN = 5700      # was 4000; raised 2026-08-31, real-video-mix rebalance
                          # (measured ceiling at this grid: 5704 train -- near-full)
TARGET_EVAL = 300
SAMPLE_SEED = 0
SPLIT_SEED = 42
ANCHOR_SEED = 0


FFMPEG_ONLY_ENV = "/homes/55/junlin/miniconda3/envs/ffmpeg_only"


def _ffmpeg_exe():
    # The imageio_ffmpeg-bundled static binary (johnvansickle build) SEGFAULTS
    # intermittently (exit code -11) under concurrent invocation -- confirmed:
    # reliable standalone, ~majority-failure under 6 parallel workers. Use the
    # conda-forge build (dynamically linked, --enable-shared) instead.
    exe = f"{FFMPEG_ONLY_ENV}/bin/ffmpeg"
    if not Path(exe).exists():
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    return exe


@dataclass(frozen=True)
class SelectionExample:
    example_id: str
    clip_id: str
    context_paths: list[str]
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


# --------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------
def _load_filtered_pool():
    pool = []
    for line in open(SRC_JUDGED):
        rec = json.loads(line)
        if rec.get("kitchen") and rec.get("manipulation") and rec["desirable_filtering"] == "desirable":
            pool.append(rec)
    return pool


def _download_clip(rec, out_path, ffmpeg_exe):
    # player_client=android: avoids the "Sign in to confirm you're not a bot"
    # block that plain web-client requests hit under concurrency (confirmed
    # empirically -- a video that failed with the web client succeeded
    # immediately with android, no other change). --sleep-requests adds a
    # small per-request delay as extra headroom against re-triggering it.
    section = f"*{rec['timestamp'][0]}-{rec['timestamp'][1]}"
    cmd = ["yt-dlp", "-f", "best[height<=480][ext=mp4]/best[height<=480]",
           "--download-sections", section,
           "--extractor-args", "youtube:player_client=android",
           "--sleep-requests", "1",
           "--ffmpeg-location", ffmpeg_exe,
           "-o", str(out_path), "--quiet", "--no-warnings", rec["url"]]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    return out_path.exists(), r.stderr[-500:] if r.stderr else ""


def _download_one(rec, ffmpeg_exe):
    """Runs in a worker thread: download + cv2 probe, no shared state touched."""
    import cv2

    clip_id = f"{rec['videoID']}_{rec['clip_idx']}"
    out_path = Path(CLIPS_ROOT) / f"{clip_id}.mp4"
    try:
        ok, err = _download_clip(rec, out_path, ffmpeg_exe)
    except Exception as e:  # noqa: BLE001
        ok, err = False, str(e)
    if not ok:
        return clip_id, None, {"videoID": rec["videoID"], "clip_idx": rec["clip_idx"],
                                "url": rec["url"], "error": err}
    cap = cv2.VideoCapture(str(out_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not fps or n_frames < 2:
        out_path.unlink(missing_ok=True)
        return clip_id, None, {"videoID": rec["videoID"], "clip_idx": rec["clip_idx"],
                                "url": rec["url"], "error": "unreadable/empty after download"}
    entry = {"path": str(out_path), "fps": fps, "num_frames": n_frames,
              "duration_sec": n_frames / fps, "videoID": rec["videoID"],
              "clip_idx": rec["clip_idx"], "caption": rec["caption"],
              "pov_hint": rec["pov_hint"]}
    return clip_id, entry, None


def stage_download(workers: int = 16, batch_size: int = N_DOWNLOAD_SAMPLE, seed_offset: int = 0):
    """Draws `batch_size` FRESH, never-before-attempted candidates (success or
    failure -- permanently-unavailable videos are never retried) and attempts
    them. Successes accumulate in MANIFEST_PATH across repeated invocations,
    so re-run with a new seed_offset to top up if the manifest still doesn't
    have enough clips for the anchors stage's target."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    pool = _load_filtered_pool()
    print(f"[download] filtered pool: {len(pool):,} clips (kitchen & manipulation & "
          f"desirable_filtering=='desirable')", flush=True)
    Path(CLIPS_ROOT).mkdir(parents=True, exist_ok=True)
    Path(OUT).mkdir(parents=True, exist_ok=True)
    manifest = json.load(open(MANIFEST_PATH)) if Path(MANIFEST_PATH).exists() else {}
    failed = json.load(open(FAILED_PATH)) if Path(FAILED_PATH).exists() else {}
    ffmpeg_exe = _ffmpeg_exe()

    already_ids = set(manifest.keys()) | set(failed.keys())
    fresh_pool = [r for r in pool if f"{r['videoID']}_{r['clip_idx']}" not in already_ids]
    todo = random.Random(SAMPLE_SEED + seed_offset).sample(fresh_pool, min(batch_size, len(fresh_pool)))
    print(f"[download] {len(manifest):,} already succeeded, {len(failed):,} previously "
          f"failed (never retried), {len(todo):,} fresh candidates this batch, "
          f"{workers} workers", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_download_one, rec, ffmpeg_exe) for rec in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            clip_id, entry, err = fut.result()
            if entry is not None:
                manifest[clip_id] = entry
            else:
                failed[clip_id] = err
            if n % 100 == 0:
                json.dump(manifest, open(MANIFEST_PATH, "w"))
                json.dump(failed, open(FAILED_PATH, "w"))
                rate = n / (time.time() - t0)
                print(f"[download] {n:,}/{len(todo):,} ({len(manifest):,} ok total, "
                      f"{len(failed):,} failed total) {rate:.2f}/s, "
                      f"ETA {(len(todo) - n) / max(rate, 1e-6) / 60:.0f}min", flush=True)

    json.dump(manifest, open(MANIFEST_PATH, "w"))
    json.dump(failed, open(FAILED_PATH, "w"))
    print(f"[download] done: {len(manifest):,} ok, {len(failed):,} failed -> {MANIFEST_PATH}")


# --------------------------------------------------------------------------
# anchors -- NWM-style: dense sliding-window transitions per clip (stride ==
# context_size+horizon, chaining, matching NWM's horizon==stride convention),
# so a single clip yields multiple anchors and negatives can be mined from
# OTHER transitions in the SAME clip (mirrors EPIC's same-video negative
# mining almost exactly, now that clips supply multiple transitions each).
# --------------------------------------------------------------------------
def _frame_path(clip_id, native_idx):
    return f"{FRAMES_ROOT}/{clip_id}/{native_idx}.jpg"


def _clip_transitions(clip_id, info):
    """All context(4)->target(1) transitions on the clip's sampling grid,
    stepping STRIDE grid-steps between window starts."""
    fps, n_native = info["fps"], info["num_frames"]
    native_stride = max(1, round(fps / SAMPLE_FPS))
    n_grid = (n_native - 1) // native_stride + 1
    out = []
    for start in range(0, n_grid - (CONTEXT_SIZE + HORIZON) + 1, STRIDE):
        context_grid = list(range(start, start + CONTEXT_SIZE))
        context_native = [g * native_stride for g in context_grid]
        target_grid = start + CONTEXT_SIZE + HORIZON - 1
        target_native = target_grid * native_stride
        if target_native >= n_native or context_native[-1] >= n_native:
            continue
        out.append((f"{clip_id}:{start}", context_native, target_grid, target_native))
    return out


def _negatives(target_grid, target_native, pool, count):
    """Temporal-distance hard negatives mined from OTHER transitions in the
    SAME clip (near_wrong_state = closest by grid distance, hardest --
    visually similar, same continuous shot; mid_wrong_state = next closest;
    same_clip_distractor = a temporally distant one, easiest). Mirrors
    data/scripts/build_epic_ssl.py's _negatives(), same-video pool replaced by
    same-clip pool since each clip now supplies multiple transitions."""
    seen, unused = {target_native}, []
    for _, _, t_grid, t_native in pool:
        if t_native in seen:
            continue
        seen.add(t_native)
        unused.append((t_grid, t_native))
    if len(unused) < count:
        return []
    unused.sort(key=lambda x: abs(x[0] - target_grid))
    selected = [(unused[0], "near_wrong_state")]
    if len(unused) > 1:
        selected.append((unused[1], "mid_wrong_state"))
    remaining = unused[2:]
    remaining.sort(key=lambda x: -abs(x[0] - target_grid))
    selected += [(x, "same_clip_distractor") for x in remaining[: count - len(selected)]]
    return selected[:count]


def _build_selection_examples(clip_ids, manifest, seed):
    examples = []
    for clip_id in clip_ids:
        transitions = _clip_transitions(clip_id, manifest[clip_id])
        for sample_id, context_native, target_grid, target_native in transitions:
            negatives = _negatives(target_grid, target_native, transitions, NUM_CANDIDATES - 1)
            if len(negatives) < NUM_CANDIDATES - 1:
                continue
            candidates = [(target_native, "ground_truth")] + \
                         [(n[1], kind) for n, kind in negatives]
            rng = random.Random(f"{seed}:{sample_id}")
            rng.shuffle(candidates)
            natives, kinds = map(list, zip(*candidates))
            examples.append(SelectionExample(
                f"selection:panda70m:{sample_id}", clip_id,
                [_frame_path(clip_id, n) for n in context_native],
                [_frame_path(clip_id, n) for n in natives], kinds,
                kinds.index("ground_truth"),
                {"context_native": context_native, "target_native": target_native,
                 "caption": manifest[clip_id]["caption"], "videoID": manifest[clip_id]["videoID"]}))
    return examples


def _subsample_capped(examples, target_n, cap, seed):
    """Ported from data/scripts/build_epic_ssl.py: shuffle, take up to `cap` per clip
    first, relax the cap only if still short of target_n."""
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


def build_anchors():
    manifest = json.load(open(MANIFEST_PATH))
    print(f"[anchors] {len(manifest):,} downloaded clips in manifest", flush=True)

    eligible = [cid for cid, info in manifest.items() if _clip_transitions(cid, info)]
    print(f"[anchors] {len(eligible):,}/{len(manifest):,} clips long enough for >=1 transition "
          f"(need >= {(CONTEXT_SIZE + HORIZON) / SAMPLE_FPS:.1f}s of extractable content)",
          flush=True)

    # split at the CLIP level first (video-disjoint), THEN generate/cap
    # anchors within each pool -- avoids near-duplicate frames from the same
    # clip appearing in both train and eval.
    rng_split = random.Random(SPLIT_SEED)
    clip_order = list(eligible)
    rng_split.shuffle(clip_order)
    n_eval_clips = max(1, round(len(clip_order) * 0.15))
    eval_clip_ids = clip_order[:n_eval_clips]
    train_clip_ids = clip_order[n_eval_clips:]
    assert not (set(eval_clip_ids) & set(train_clip_ids))

    anchors_dir = Path(OUT) / "anchors"
    all_written = {}
    for split, clip_ids, target_n in (("train", train_clip_ids, TARGET_TRAIN),
                                        ("eval", eval_clip_ids, TARGET_EVAL)):
        examples = _build_selection_examples(clip_ids, manifest, ANCHOR_SEED)
        print(f"[anchors] {split}: {len(examples):,} candidate anchors from "
              f"{len(clip_ids):,} clips (negative mining succeeded)", flush=True)
        if len(examples) < target_n:
            raise SystemExit(f"[anchors] {split}: only {len(examples):,} anchors available, "
                              f"need {target_n:,} -- download more clips")
        anchors = _subsample_capped(examples, target_n, PER_CLIP_CAP, ANCHOR_SEED)
        write_jsonl(anchors_dir / f"anchors_{split}.jsonl", anchors)
        n_clips_used = len({a.clip_id for a in anchors})
        print(f"[anchors] {split}: wrote {len(anchors):,} anchors ({n_clips_used:,} distinct "
              f"clips) -> {anchors_dir}/anchors_{split}.jsonl", flush=True)
        all_written[split] = anchors

    assert not ({a.clip_id for a in all_written["train"]} & {a.clip_id for a in all_written["eval"]})
    for split, anchors in all_written.items():
        bad = [a.example_id for a in anchors if a.candidate_types[a.answer_index] != "ground_truth"]
        assert not bad, f"oracle sanity failed for {len(bad)} anchors in {split}: {bad[:5]}"
    print("[anchors] clip-disjointness + oracle sanity checks passed", flush=True)

    _extract_frames(all_written["train"] + all_written["eval"], manifest)


def _extract_frames(anchors, manifest):
    import cv2
    from PIL import Image

    needed = defaultdict(set)
    for ex in anchors:
        for p in ex.context_paths + ex.candidate_paths:
            clip_id = Path(p).parent.name
            native_idx = int(Path(p).stem)
            needed[clip_id].add(native_idx)

    total_written = total_skipped = total_failed = 0
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
                total_failed += 1
                continue
            Image.fromarray(frame[:, :, ::-1]).save(out_dir / f"{idx}.jpg", quality=95)
        cap.release()
        total_written += len(todo)
        total_skipped += len(indices) - len(todo)
        if (i + 1) % 500 == 0:
            print(f"[frames] {i + 1}/{len(needed)} clips processed", flush=True)
    print(f"[frames] total: wrote {total_written}, skipped (existing) {total_skipped}, "
          f"failed {total_failed}")


# --------------------------------------------------------------------------
# gen
# --------------------------------------------------------------------------
GEN_GPT_ANSWER = "The next frame should look like this: <img>"


def build_gen_prompt_noaction(n: int) -> str:
    head = "".join("<image>\n" for _ in range(n))
    return (head + f"These are {n} consecutive frames from a kitchen video. "
            f"Observe these frames and produce the next frame of the video.")


def _build_gen_split(anchors):
    rows, manifest = [], []
    for idx, ex in enumerate(anchors):
        n = len(ex.context_paths)
        target = ex.candidate_paths[ex.answer_index]
        prompt = build_gen_prompt_noaction(n)
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
    task_dir = Path(OUT) / "panda70m_gen_noaction"
    task_dir.mkdir(parents=True, exist_ok=True)
    lengths = {}
    for split in ("train", "eval"):
        anchors = read_jsonl(anchors_dir / f"anchors_{split}.jsonl", SelectionExample.from_dict)
        rows, manifest = _build_gen_split(anchors)
        with open(task_dir / f"panda70m_gen_noaction_{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        write_jsonl(task_dir / f"manifest_{split}.jsonl", manifest)
        lengths[split] = len(rows)
        print(f"[gen] {split}: {len(rows)} rows -> {task_dir}/panda70m_gen_noaction_{split}.jsonl")

    meta_dir = Path(DATA) / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    json.dump({"panda70m_gen_noaction": {
        "root": FRAMES_ROOT, "annotation": str(task_dir / "panda70m_gen_noaction_train.jsonl"),
        "data_augment": False, "max_dynamic_patch": CONTEXT_SIZE,
        "repeat_time": 1, "length": lengths["train"], "task_type": "imgen"}},
        open(meta_dir / "panda70m_gen_noaction_meta.json", "w"), indent=2)
    json.dump({"panda70m_gen_noaction_eval": {
        "root": FRAMES_ROOT, "annotation": str(task_dir / "panda70m_gen_noaction_eval.jsonl"),
        "data_augment": False, "max_dynamic_patch": CONTEXT_SIZE,
        "repeat_time": 1, "length": lengths["eval"], "task_type": "imgen"}},
        open(meta_dir / "panda70m_gen_noaction_eval_meta.json", "w"), indent=2)

    with open(task_dir / "VERSION", "w") as f:
        f.write(
            "version: v1\n"
            "task: next-frame generation (imgen), noaction (no instruction conditioning)\n"
            "dataset: Panda-70M training_10m, filtered to kitchen+manipulation via regex+"
            "Qwen3-VL-30B-AWQ judge (see data/datasets/panda70m/README.md), desirable_filtering=="
            "'desirable' only, 5000 clips randomly sampled and downloaded\n"
            f"sample_fps: {SAMPLE_FPS}, context_size: {CONTEXT_SIZE}, horizon: {HORIZON} "
            f"grid-step ({HORIZON / SAMPLE_FPS:.2f}s)\n"
            f"split: {lengths['train']} train + {lengths['eval']} eval (clip-disjoint, "
            "1 anchor per clip)\n"
            f"split_seed: {SPLIT_SEED}, anchor_seed: {ANCHOR_SEED}, sample_seed: {SAMPLE_SEED}\n"
            "prompt wording deliberately does NOT claim 'first-person'/'egocentric' framing "
            "-- a manual frame spot-check found Panda-70M kitchen clips are mostly "
            "third-person YouTube content, not true egocentric video (see conversation).\n"
            "created: 2026-08-03\n"
            "spec: see data/specs/EPIC_SSL_DATA_SPEC.md for the format this mirrors\n"
        )
    print("[gen] done")


# --------------------------------------------------------------------------
# dual
# --------------------------------------------------------------------------
def build_ffs_prompt_noaction(n: int, k: int, caption: str | None = None) -> str:
    labels = [chr(ord("A") + i) for i in range(k)]
    head = "".join(f"Frame {i + 1}: <image>\n" for i in range(n))
    body = f"These are {n} consecutive frames from a kitchen video, in order.\n"
    if caption:
        body = f"Video description: {caption.strip()}\n\n" + body
    opts = "".join(f"Option {labels[i]}: <image>\n" for i in range(k))
    return (head + body + opts +
            f"Which option is the true next frame? Answer with a single letter ({', '.join(labels)}).")


def _build_ffs_split(anchors):
    """S3 requires caption (VC2IA-F) -- inject each anchor's real caption
    (ex.metadata['caption'], same field panda70m_gen_S2 already uses)."""
    rows = []
    for idx, ex in enumerate(anchors):
        images = list(ex.context_paths) + list(ex.candidate_paths)
        caption = (ex.metadata or {}).get("caption")
        prompt = build_ffs_prompt_noaction(len(ex.context_paths), len(ex.candidate_paths), caption)
        rows.append({
            "id": idx, "task_type": "understanding", "image": images,
            "conversations": [{"from": "human", "value": prompt},
                               {"from": "gpt", "value": ex.answer_label}],
        })
    return rows


def build_dual():
    if DUAL_TOOL_DIR not in sys.path:
        sys.path.insert(0, DUAL_TOOL_DIR)
    import make_nwm_dual_ssl as dual_tool  # fully dataset-agnostic converter

    anchors_dir = Path(OUT) / "anchors"
    ffs_dir = Path(OUT) / "_ffs_internal"  # intermediate only, not a shipped task
    ffs_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "eval"):
        anchors = read_jsonl(anchors_dir / f"anchors_{split}.jsonl", SelectionExample.from_dict)
        rows = _build_ffs_split(anchors)
        with open(ffs_dir / f"_ffs_internal_{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dst_dir = Path(OUT) / "panda70m_dual_noaction"
    dst_dir.mkdir(parents=True, exist_ok=True)
    lengths = {}
    for split in ("train", "eval"):
        src = ffs_dir / f"_ffs_internal_{split}.jsonl"
        dst = dst_dir / f"panda70m_dual_noaction_{split}.jsonl"
        n = dual_tool.convert_split(src, dst, n_ctx=CONTEXT_SIZE)
        lengths[split] = n
        print(f"[dual] {split}: {n} rows -> {dst}")

    meta_dir = Path(DATA) / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    json.dump({"panda70m_dual_noaction": {
        "root": FRAMES_ROOT, "annotation": str(dst_dir / "panda70m_dual_noaction_train.jsonl"),
        "data_augment": False, "max_dynamic_patch": CONTEXT_SIZE + NUM_CANDIDATES,
        "repeat_time": 1, "length": lengths["train"], "task_type": "imgen"}},
        open(meta_dir / "panda70m_dual_noaction_meta.json", "w"), indent=2)
    json.dump({"panda70m_dual_noaction_eval": {
        "root": FRAMES_ROOT, "annotation": str(dst_dir / "panda70m_dual_noaction_eval.jsonl"),
        "data_augment": False, "max_dynamic_patch": CONTEXT_SIZE + NUM_CANDIDATES,
        "repeat_time": 1, "length": lengths["eval"], "task_type": "imgen"}},
        open(meta_dir / "panda70m_dual_noaction_eval_meta.json", "w"), indent=2)

    with open(dst_dir / "VERSION", "w") as f:
        f.write(
            "version: v1\n"
            "task: dual FFS-selection + generation, 'dual-leak' (imgen, imgen_form=multimodal)\n"
            "converted_from: _ffs_internal_{train,eval}.jsonl (1:1, via "
            "Model_Related/InternVLU/InternVL/internvl_chat/tools/make_nwm_dual_ssl.py, "
            "reused UNMODIFIED -- dataset-agnostic, only reads the FFS row's 8-image list "
            "+ answer letter, N_CTX=4 matches)\n"
            f"split: {lengths['train']} train + {lengths['eval']} eval\n"
            "cond_image: last context frame (ctx4).\n"
            "is_option: [0,0,0,0,1,1,1,1] -- read by the trainer to build "
            "option_context_mask; masking of option images out of the GENERATION "
            "conditioning only happens if --gen_no_leak is passed at train time "
            "(default False = leak ON). This IS the 'dual-leak' variant: do NOT pass "
            "--gen_no_leak at train time.\n"
            "negatives: same-clip spare frames (near/mid, hardest -- same continuous shot) "
            "when the clip is long enough, else cross-clip fallback from a different "
            "downloaded clip (easiest). See build_panda70m_epic_ssl.py's module docstring "
            "for why this differs from EPIC's same-video-different-transition negatives "
            "(Panda-70M clips are short, one anchor per clip, no multi-transition pool).\n"
            "created: 2026-08-03\n"
        )
    print("[dual] done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["download", "anchors", "gen", "dual"])
    ap.add_argument("--workers", type=int, default=16, help="download stage concurrency")
    ap.add_argument("--batch_size", type=int, default=N_DOWNLOAD_SAMPLE,
                     help="download stage: #fresh candidates to attempt this run")
    ap.add_argument("--seed_offset", type=int, default=0,
                     help="download stage: bump to draw a different fresh batch on retry")
    args = ap.parse_args()
    if args.stage == "download":
        stage_download(args.workers, args.batch_size, args.seed_offset)
    elif args.stage == "anchors":
        build_anchors()
    elif args.stage == "gen":
        build_gen()
    elif args.stage == "dual":
        build_dual()


if __name__ == "__main__":
    main()
