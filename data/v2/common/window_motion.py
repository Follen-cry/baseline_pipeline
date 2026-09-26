#!/usr/bin/env python3
"""How much a 4-frame window changes: the static-window filter of v2.

Frames are downscaled (short side SHORT_SIDE), converted to gray, blurred (BLUR), and two frames
differ at a pixel if |a - b| > DIFF_THR (same pixel rule as clip_qc.py's obj_motion). The change
between two frames = fraction of differing pixels. Nothing is written to disk.

Per window (frames F0..F3):
  change_vs_f0   [F0-F1, F0-F2, F0-F3]
  change_adjacent[F0-F1, F1-F2, F2-F3]

Filter (used by recipes/temporal_ssl/merge_pools.py): a window is static, and dropped, if
max(change_vs_f0) < STATIC_THR, i.e. none of F1..F3 differs from F0. STATIC_THR is a fraction of
pixels, set very low on purpose: small synthetic objects (a moving VBVR ball, a growing arrow) change
only 0.4-1% of the frame and must be kept. Windows that stop part-way (an adjacent pair below the
threshold) are kept; their scores are recorded so later steps can keep them out of ORDER / MISSING
labels, where two identical frames are ambiguous.
"""
import os

import cv2
import numpy as np

from extract_frames import IMG_EXT, read_video_frames

SHORT_SIDE = 256
BLUR = 5
DIFF_THR = 25
STATIC_THR = 0.001

cv2.setNumThreads(1)


def prep(img):
    h, w = img.shape[:2]
    s = SHORT_SIDE / min(h, w)
    if s < 1:
        img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(g, (BLUR, BLUR), 0).astype(np.int16)


def change(a, b):
    return round(float((np.abs(a - b) > DIFF_THR).mean()), 6)


def window_scores(frames):
    """frames: 4 prepped frames F0..F3 -> {change_vs_f0, change_adjacent}."""
    return {"change_vs_f0": [change(frames[0], frames[j]) for j in (1, 2, 3)],
            "change_adjacent": [change(frames[j], frames[j + 1]) for j in range(3)]}


def is_static(scores, thr=STATIC_THR):
    return max(scores["change_vs_f0"]) < thr


def score_clip(video, windows):
    """Scores of every window of one clip ({dt, k, frame_idx}); None if a frame fails to decode.
    `video` is a video file or a frame directory (sorted by name, as in extract_frames.py)."""
    wanted = sorted({i for w in windows for i in w["frame_idx"]})
    if os.path.isdir(video):
        files = sorted(f for f in os.listdir(video) if f.lower().endswith(IMG_EXT))
        imgs = {i: cv2.imread(os.path.join(video, files[i])) for i in wanted if i < len(files)}
    else:
        imgs = read_video_frames(video, set(wanted))
    if any(imgs.get(i) is None for i in wanted):
        return None
    p = {i: prep(imgs[i]) for i in wanted}
    return [{"dt": w["dt"], "k": w["k"], **window_scores([p[i] for i in w["frame_idx"]])} for w in windows]
