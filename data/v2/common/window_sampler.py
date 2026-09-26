#!/usr/bin/env python3
"""Global v2 window-sampling rule: which 4-frame windows to take from one clip.

Uses only the clip length (num_frames, fps); no content signal, so every label is derived from
timestamps alone. Pure function, no IO; `extract_frames.py` applies it to every source.

Rule (see data/v2/common/README.md):
  - Clip duration T = num_frames / fps. Keep clips with T <= MAX_DURATION (+ DURATION_TOL:
    nominal 20 s re-encoded clips measure 20.05-20.10 s).
  - Scales Δt ∈ SCALES; a window spans 3Δt (frames at t, t+Δt, t+2Δt, t+3Δt).
  - Per scale: N = floor(T / 3Δt) windows starting at t_k = k·3Δt, k = 0..N-1.
  - Frame index = round(ts · fps); an index equal to num_frames (the last window ends exactly
    at T) uses the last frame, num_frames - 1.
  - At most N_MAX windows per clip. Over the cap, water-fill: each round gives one slot to
    every scale that still has windows, larger Δt first, until N_MAX slots are used.
    Within a scale keep windows at round(linspace(0, N-1, quota)); quota 1 = middle window.

Usage (prints the windows for a clip length):
    python window_sampler.py --num-frames 600 --fps 30
"""
import argparse, math

SCALES = (0.5, 1.0, 2.0)  # Δt in seconds
N_MAX = 10
MAX_DURATION = 20.0
DURATION_TOL = 0.25
EPS = 1e-6


def allocate(counts, n_max=N_MAX):
    """Water-fill n_max slots over scales; counts = {dt: N_dt}. Larger Δt served first."""
    if sum(counts.values()) <= n_max:
        return dict(counts)
    quota = {dt: 0 for dt in counts}
    order = sorted(counts, reverse=True)
    while sum(quota.values()) < n_max:
        for dt in order:
            if quota[dt] < counts[dt] and sum(quota.values()) < n_max:
                quota[dt] += 1
    return quota


def spread(n, q):
    """q of n window indices, evenly spaced over time; q = 1 -> the middle window."""
    if q >= n:
        return list(range(n))
    if q == 1:
        return [int((n - 1) / 2 + 0.5)]
    return [int(i * (n - 1) / (q - 1) + 0.5) for i in range(q)]


def sample_windows(num_frames, fps, scales=SCALES, n_max=N_MAX, max_duration=MAX_DURATION):
    """Windows of one clip, as dicts {dt, k, start_s, frame_idx: [4 ints]}.
    Returns [] if the clip is longer than max_duration or too short for the smallest scale."""
    if num_frames <= 0 or fps <= 0:
        return []
    T = num_frames / fps
    if T > max_duration + DURATION_TOL:
        return []
    counts = {dt: int(math.floor(T / (3 * dt) + EPS)) for dt in scales}
    counts = {dt: n for dt, n in counts.items() if n > 0}
    quota = allocate(counts, n_max)
    out = []
    for dt in sorted(counts):
        span = 3 * dt
        for k in spread(counts[dt], quota[dt]):
            start = k * span
            idx = [int(round((start + j * dt) * fps)) for j in range(4)]
            idx = [num_frames - 1 if i == num_frames else i for i in idx]
            assert all(0 <= i < num_frames for i in idx) and idx == sorted(set(idx)), (idx, num_frames)
            out.append({"dt": dt, "k": k, "start_s": round(start, 4), "frame_idx": idx})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-frames", type=int, required=True)
    ap.add_argument("--fps", type=float, required=True)
    a = ap.parse_args()
    ws = sample_windows(a.num_frames, a.fps)
    print(f"T = {a.num_frames / a.fps:.3f} s, {len(ws)} windows")
    for w in ws:
        print(w)


if __name__ == "__main__":
    main()
