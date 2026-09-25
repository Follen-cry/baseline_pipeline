#!/usr/bin/env python3
"""Per-clip quality metrics for v2 video sources (CPU only, deterministic).

For every row of a source's selection manifest (needs `id`, `video`), sample frames at
SAMPLE_FPS, downscale to width W, and measure:

  decode_ok, fps, n_frames, duration, width, height
  n_cuts        : consecutive samples whose HSV-histogram correlation < CUT_CORR (scene cuts)
  cam_motion    : median over sample pairs of the mean corner displacement of the ORB+RANSAC
                  homography, in fractions of the frame width per second (0 = locked-off camera)
  cam_motion_p90: same, 90th percentile
  obj_motion    : median over pairs of the fraction of pixels that still change (>25/255) after
                  warping the previous sample by the homography (0 = nothing moves)
  obj_motion_max: same, maximum over pairs (catches short events in otherwise static clips)

Output: one json line per clip in <out>. Resumable (clips already in <out> are skipped).

Usage:
    python clip_qc.py --manifest raw/<src>/selection_10k.jsonl --out raw/<src>/qc.jsonl [--procs 32]
"""
import argparse, json, os
from multiprocessing import Pool

import cv2
import numpy as np

SAMPLE_FPS = 4.0
W = 320
CUT_CORR = 0.5
DIFF_THR = 25

cv2.setNumThreads(1)


def sample_frames(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, int(round(fps / SAMPLE_FPS))) if fps > 0 else 1
    frames, i = [], 0
    while True:
        if i % step == 0:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.resize(f, (W, max(2, int(round(h * W / max(w, 1)))))))
        elif not cap.grab():
            break
        i += 1
    cap.release()
    return frames, fps, i, w, h, step


def pair_metrics(a, b, orb):
    ga, gb = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    ha = cv2.calcHist([cv2.cvtColor(a, cv2.COLOR_BGR2HSV)], [0, 1], None, [32, 32], [0, 180, 0, 256])
    hb = cv2.calcHist([cv2.cvtColor(b, cv2.COLOR_BGR2HSV)], [0, 1], None, [32, 32], [0, 180, 0, 256])
    corr = cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)
    H = None
    ka, da = orb.detectAndCompute(ga, None)
    kb, db = orb.detectAndCompute(gb, None)
    if da is not None and db is not None and len(ka) >= 8 and len(kb) >= 8:
        m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
        if len(m) >= 8:
            pa = np.float32([ka[x.queryIdx].pt for x in m])
            pb = np.float32([kb[x.trainIdx].pt for x in m])
            H, _ = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
    h, w = ga.shape
    if H is None:
        H = np.eye(3)
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    disp = float(np.linalg.norm(cv2.perspectiveTransform(corners, H) - corners, axis=2).mean()) / w
    warped = cv2.warpPerspective(ga, H, (w, h))
    valid = cv2.warpPerspective(np.full_like(ga, 255), H, (w, h)) > 0
    diff = (cv2.absdiff(warped, gb) > DIFF_THR) & valid
    obj = float(diff.sum()) / max(1, int(valid.sum()))
    return corr, disp, obj


def qc(row):
    out = {"id": row["id"], "decode_ok": False}
    try:
        frames, fps, n, w, h, step = sample_frames(row["video"])
    except Exception as e:                         # noqa: BLE001
        out["error"] = repr(e)
        return out
    out.update(fps=round(fps, 3), n_frames=n, width=w, height=h,
               duration=round(n / fps, 3) if fps else 0.0)
    if len(frames) < 3 or fps <= 0:
        return out
    orb = cv2.ORB_create(500)
    dt = step / fps
    corrs, disps, objs = [], [], []
    for a, b in zip(frames, frames[1:]):
        c, d, o = pair_metrics(a, b, orb)
        corrs.append(c); disps.append(d / dt); objs.append(o)
    cut = [c < CUT_CORR for c in corrs]
    keep = [i for i, x in enumerate(cut) if not x] or list(range(len(cut)))
    d, o = np.array(disps)[keep], np.array(objs)[keep]
    out.update(decode_ok=True, n_cuts=int(sum(cut)),
               cam_motion=round(float(np.median(d)), 4), cam_motion_p90=round(float(np.percentile(d, 90)), 4),
               obj_motion=round(float(np.median(o)), 4), obj_motion_max=round(float(o.max()), 4))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=32)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.manifest)]
    done = {json.loads(l)["id"] for l in open(a.out)} if os.path.exists(a.out) else set()
    todo = [r for r in rows if r["id"] not in done and os.path.exists(r["video"])]
    print(f"[qc] {len(rows)} rows, {len(done)} done, {len(todo)} to do", flush=True)
    with Pool(a.procs) as p, open(a.out, "a") as f:
        for k, res in enumerate(p.imap_unordered(qc, todo, chunksize=4), 1):
            f.write(json.dumps(res) + "\n")
            if k % 500 == 0:
                f.flush()
                print(f"[qc] {k}/{len(todo)}", flush=True)
    print("[qc] complete", flush=True)


if __name__ == "__main__":
    main()
