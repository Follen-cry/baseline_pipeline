#!/usr/bin/env python3
"""Apply the global window rule (window_sampler.py) to a clip list and write the frames.

Input: a jsonl of clips, one row each, with
  source, clip_id : frames go to <frames-root>/<source>/<clip_id>/<frame_idx:05d>.jpg
  video           : a video file (mp4/webm/...) or a directory of frame images (sorted by name)
  fps             : required for a frame directory; ignored for a video (container fps is used)
  windows         : optional [[dt, k], ...] -> keep only these windows of the global rule
                    (a planned subset, e.g. from recipes/temporal_ssl/merge_pools.py)
  num_frames_expected : optional; the decoded frame count the plan was made from. A clip that
                    now decodes to a different count gets status count_mismatch.
All other fields (caption, split, ...) are copied into every window row of that clip.

Video decoding is two sequential passes with OpenCV: pass 1 counts the frames that actually
decode (container frame counts are unreliable, e.g. webm), pass 2 keeps only the needed ones.
A frame shared by several windows is written once. Frames keep their native resolution and aspect
ratio, and are only downscaled when a side exceeds --max-side (fit in a max-side square, never
upscaled); saved as JPEG (--quality).

Output:
  <out>             one row per window: clip fields + window_id, dt, k, start_s, frame_idx,
                    timestamps, frames (4 absolute paths), fps, num_frames, duration
  <out>.clips.jsonl one status row per clip (ok / too_long / too_short / decode_fail /
                  count_mismatch);
                    resumable: clips listed there are skipped.

Usage:
    python extract_frames.py --clips raw/<src>/clips.jsonl --out raw/<src>/windows.jsonl [--procs 16]
"""
import argparse, json, os
from multiprocessing import Pool

import cv2

from paths import FRAMES_ROOT
from window_sampler import MAX_DURATION, DURATION_TOL, sample_windows

IMG_EXT = (".jpg", ".jpeg", ".png")

cv2.setNumThreads(1)


def resize(img, max_side):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


def count_video_frames(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n, fps


def clip_length(video, fps=None):
    """(num_frames, fps, frame files or None) exactly as process() sees the clip; (0, 0.0, None)
    if missing. Shared with the planner so planned windows match the extracted ones."""
    if os.path.isdir(video):
        files = sorted(f for f in os.listdir(video) if f.lower().endswith(IMG_EXT))
        return len(files), float(fps or 0), files
    if os.path.isfile(video):
        n, vfps = count_video_frames(video)
        return n, vfps, None
    return 0, 0.0, None


def read_video_frames(path, wanted):
    """{idx: image} for the wanted frame indices, in one sequential pass."""
    cap = cv2.VideoCapture(path)
    out, i, last = {}, 0, max(wanted)
    while i <= last and cap.grab():
        if i in wanted:
            ok, f = cap.retrieve()
            if ok:
                out[i] = f
        i += 1
    cap.release()
    return out


def process(args):
    row, frames_root, max_side, quality = args
    src, cid, video = row["source"], str(row["clip_id"]), row["video"]
    status = {"source": src, "clip_id": cid, "video": video}
    is_dir = os.path.isdir(video)
    n, fps, files = clip_length(video, row.get("fps"))
    status.update(num_frames=n, fps=fps, duration=round(n / fps, 4) if fps > 0 else None)
    if n == 0 or fps <= 0:
        return {**status, "status": "decode_fail"}, []
    expected = row.get("num_frames_expected")
    if expected is not None and expected != n:
        return {**status, "status": "count_mismatch", "num_frames_expected": expected}, []
    windows = sample_windows(n, fps)
    if not windows:
        why = "too_long" if n / fps > MAX_DURATION + DURATION_TOL else "too_short"
        return {**status, "status": why}, []
    if row.get("windows") is not None:
        keep = {(float(dt), int(k)) for dt, k in row["windows"]}
        windows = [w for w in windows if (w["dt"], w["k"]) in keep]
        if len(windows) != len(keep):
            return {**status, "status": "window_mismatch"}, []

    clip_dir = os.path.join(frames_root, src, cid.replace("/", "__"))
    os.makedirs(clip_dir, exist_ok=True)
    wanted = sorted({i for w in windows for i in w["frame_idx"]})
    path_of = {i: os.path.join(clip_dir, f"{i:05d}.jpg") for i in wanted}
    todo = {i for i in wanted if not os.path.exists(path_of[i])}
    if todo:
        imgs = ({i: cv2.imread(os.path.join(video, files[i])) for i in todo} if is_dir
                else read_video_frames(video, todo))
        missing = [i for i in todo if imgs.get(i) is None]
        if missing:
            return {**status, "status": "decode_fail", "missing_idx": missing[:10]}, []
        for i in sorted(todo):
            cv2.imwrite(path_of[i], resize(imgs[i], max_side), [cv2.IMWRITE_JPEG_QUALITY, quality])

    base = {k: v for k, v in row.items() if k not in ("fps", "windows", "num_frames_expected")}
    rows = []
    for w in windows:
        rows.append({**base, "clip_id": cid,
                     "window_id": f"{cid}__dt{w['dt']}__k{w['k']}",
                     "dt": w["dt"], "k": w["k"], "start_s": w["start_s"],
                     "frame_idx": w["frame_idx"],
                     "timestamps": [round(i / fps, 4) for i in w["frame_idx"]],
                     "frames": [path_of[i] for i in w["frame_idx"]],
                     "fps": fps, "num_frames": n, "duration": status["duration"]})
    return {**status, "status": "ok", "n_windows": len(rows), "n_frames_written": len(todo)}, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames-root", default=FRAMES_ROOT)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--limit", type=int, help="first N clips only (smoke test)")
    a = ap.parse_args()

    clips = [json.loads(l) for l in open(a.clips)]
    if a.limit:
        clips = clips[:a.limit]
    log = a.out + ".clips.jsonl"
    done = set()
    if os.path.exists(log):
        done = {json.loads(l)["clip_id"] for l in open(log)}
    if os.path.exists(a.out):  # drop windows of clips that crashed before their status row
        kept = [l for l in open(a.out) if json.loads(l)["clip_id"] in done]
        with open(a.out, "w") as f:
            f.writelines(kept)
    todo = [c for c in clips if str(c["clip_id"]) not in done]
    print(f"{len(clips)} clips, {len(done)} done, {len(todo)} to process")

    counts = {}
    with open(a.out, "a") as fw, open(log, "a") as fl, Pool(a.procs) as pool:
        jobs = ((c, a.frames_root, a.max_side, a.quality) for c in todo)
        for n, (st, rows) in enumerate(pool.imap_unordered(process, jobs, chunksize=4), 1):
            for r in rows:
                fw.write(json.dumps(r) + "\n")
            fw.flush()
            fl.write(json.dumps(st) + "\n")
            fl.flush()
            counts[st["status"]] = counts.get(st["status"], 0) + 1
            if n % 500 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} {counts}", flush=True)


if __name__ == "__main__":
    main()
