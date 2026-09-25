#!/usr/bin/env python3
"""Download the physics-filtered Panda-70M clips (v2 source) until --target clips succeed.

Input: <OUT>/pool_passed.jsonl from filter_panda70m.py (one row per clip; `segment` is the
longest single shot of the Panda-70M clip, capped at 20 s).

Order (seed 42): all tier-A rows first, then tier B (the fallback pool). Within a tier, rows
are grouped by the judge's `category`, shuffled within each group, and interleaved round-robin
across groups, so stopping at --target keeps the category mix as even as the pool allows.

Per clip:
  1. yt-dlp downloads the FULL video at <=360p (progressive itag 18, `android` then
     `tv_simply` client). yt-dlp --download-sections pipes the remote stream through ffmpeg and
     segfaults (exit -11) on this cluster for most videos (re-confirmed 2026-09-24), so the cut is
     done locally instead.
  2. ffmpeg cuts `segment` and RE-ENCODES it (libx264, no audio). Stream-copy cuts start on a
     non-keyframe and leave frames that cannot be decoded; v1's panda70m_epic_ssl clips lost
     about half their frames that way.
  3. cv2 decodes the whole output sequentially. The clip counts only if
     decoded >= DECODE_MIN_FRAC * fps * segment_dur and decoded / fps >= MIN_DUR_S.

Every attempt is appended to <OUT>/download_log.jsonl (resumable: done and permanently failed
clip_keys are skipped; rate-limit failures are retried on the next run). Successful clips go to
<OUT>/videos/<clip_key>.mp4, and the final manifest <OUT>/selection_5k.jsonl is written from the
successes, in download order, capped at --target.

Rate limiting: if >= --cb-threshold of the last --cb-window attempts look like a YouTube block
(429 / "confirm you're not a bot"), all workers pause --cb-pause seconds; after --cb-max-pauses
consecutive pauses without a success in between, the run exits (rerun later to resume).

Usage:
    python download_panda70m.py --limit 30                # pilot
    python download_panda70m.py --target 5000 --workers 4
    python download_panda70m.py --target 5000 --workers 2 --cookies ~/yt_cookies.txt  # after a bot-check block
"""
import argparse, json, os, random, shutil, subprocess, threading, time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
from pathlib import Path

OUT = Path("/scratch/network/ssd/junlin/raw/panda70m")
POOL = OUT / "pool_passed.jsonl"
VIDEOS = OUT / "videos"
LOG = OUT / "download_log.jsonl"
SELECTION = OUT / "selection_5k.jsonl"
TMP = Path("/scratch/local/ssd/junlin/tmp/panda70m_v2_dl")  # full videos are big; keep off NFS
SEED = 42
DECODE_MIN_FRAC = 0.95
MIN_DUR_S = 3.0  # tier B floor (filter_panda70m.MIN_SHOT_S_B)
CLIENTS = ("android", "tv_simply")
COOKIES: str | None = None  # --cookies (Netscape cookies.txt); needed once YouTube bot-checks the IP
# With cookies, the android/tv_simply clients are unusable (no cookie support / no itag 18);
# yt-dlp's default client and mweb work (tested 2026-09-25). None = default client.
COOKIE_CLIENTS = (None, "mweb")
BLOCK_MARKERS = ("429", "confirm you", "sign in to confirm")
# failures that won't change on retry
PERMANENT_MARKERS = ("video unavailable", "private video", "has been removed", "not available",
                     "copyright", "account associated", "members-only", "age", "terminated")


def _ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


FFMPEG = _ffmpeg()


def _classify(stderr: str) -> str:
    s = stderr.lower()
    if any(m in s for m in BLOCK_MARKERS):
        return "blocked"
    if any(m in s for m in PERMANENT_MARKERS):
        return "unavailable"
    return "error"


def _fetch_full(url: str, tmp: Path, timeout: int) -> tuple[bool, str, str]:
    kind, msg = "error", ""
    ck = None
    if COOKIES:  # yt-dlp rewrites the cookie file; give each attempt its own copy
        ck = tmp.with_suffix(".cookies.txt")
        shutil.copyfile(COOKIES, ck)
        os.chmod(ck, 0o600)
    for client in (COOKIE_CLIENTS if COOKIES else CLIENTS):
        tmp.unlink(missing_ok=True)
        cmd = ["yt-dlp", "-f", "18/best[height<=360]/best", "--no-playlist", "--no-progress",
               "--quiet", "--no-warnings", "--socket-timeout", "20", "-o", str(tmp), url]
        if client:
            cmd[1:1] = ["--extractor-args", f"youtube:player_client={client}"]
        if ck:
            cmd[1:1] = ["--cookies", str(ck)]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            kind, msg = "error", f"timeout({client})"
            continue
        if p.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            break
        kind, msg = _classify(p.stderr), p.stderr.strip()[-200:]
        if kind == "blocked":
            break
    if ck:
        ck.unlink(missing_ok=True)
    if tmp.exists() and tmp.stat().st_size > 0:
        return True, "", ""
    return False, kind, msg


def _decoded(path: Path) -> tuple[float, int, int, int, int]:
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(3)), int(cap.get(4))
    reported, n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0
    while cap.grab():
        n += 1
    cap.release()
    return fps, reported, n, w, h


def download_one(row: dict, timeout: int) -> dict:
    t0 = time.time()
    r = _download_one(row, timeout)
    r["time_s"] = round(time.time() - t0, 1)
    return r


def _download_one(row: dict, timeout: int) -> dict:
    key = row["clip_key"]
    tmp = TMP / f"{key}.full.mp4"
    out = VIDEOS / f"{key}.mp4"
    rec = {"clip_key": key}
    try:
        ok, kind, msg = _fetch_full(row["url"], tmp, timeout)
        if not ok:
            return dict(rec, status=kind, error=msg)
        s, e = row["segment"]
        p = subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", s, "-to", e, "-i", str(tmp),
                            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                            "-pix_fmt", "yuv420p", str(out)],
                           capture_output=True, text=True, timeout=300)
        if p.returncode != 0 or not out.exists():
            out.unlink(missing_ok=True)
            return dict(rec, status="trim_failed", error=p.stderr.strip()[-200:])
        fps, reported, n, w, h = _decoded(out)
        need = DECODE_MIN_FRAC * fps * row["segment_dur_s"] if fps else float("inf")
        if n < need or not fps or n / fps < MIN_DUR_S:
            out.unlink(missing_ok=True)
            return dict(rec, status="decode_short", error=f"decoded {n}, need {need:.0f}, fps {fps}")
        return dict(rec, status="success", fps=fps, num_frames=n, duration_s=round(n / fps, 3),
                    width=w, height=h, bytes=out.stat().st_size)
    except Exception as ex:  # noqa: BLE001 -- one bad clip must not kill the run
        out.unlink(missing_ok=True)
        return dict(rec, status="error", error=f"{type(ex).__name__}: {ex}")
    finally:
        tmp.unlink(missing_ok=True)


def ordered_pool() -> list[dict]:
    rng = random.Random(SEED)
    rows = [json.loads(l) for l in open(POOL)]
    order = []
    for tier in ("A", "B"):
        groups = defaultdict(list)
        for r in rows:
            if r.get("tier", "A") == tier:
                groups[r["judge"]["category"]].append(r)
        for c in sorted(groups):
            rng.shuffle(groups[c])
        cats = sorted(groups)
        while any(groups.values()):
            for c in cats:
                if groups[c]:
                    order.append(groups[c].pop())
    return order


def write_selection(target: int):
    pool = {r["clip_key"]: r for r in (json.loads(l) for l in open(POOL))}
    rows = [json.loads(l) for l in open(LOG)]
    succ = [r for r in rows if r["status"] == "success"][:target]
    with open(SELECTION, "w") as f:
        for r in succ:
            p = pool[r["clip_key"]]
            f.write(json.dumps({
                "id": r["clip_key"], "tier": p.get("tier", "A"),
                "desirable_filtering": p.get("desirable_filtering", "desirable"),
                "video_id": p["videoID"], "url": p["url"],
                "panda_timestamp": p["timestamp"], "segment": p["segment"],
                "caption": p["caption"], "matching_score": p["matching_score"],
                "category": p["judge"]["category"], "judge": p["judge"],
                "fps": r["fps"], "num_frames": r["num_frames"], "duration_s": r["duration_s"],
                "width": r["width"], "height": r["height"],
                "video": str(VIDEOS / f"{r['clip_key']}.mp4"),
            }) + "\n")
    cats = Counter(json.loads(l)["category"] for l in open(SELECTION))
    print(f"[selection] {len(succ)} clips -> {SELECTION}  {dict(cats.most_common())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=300, help="per-video yt-dlp timeout (s)")
    ap.add_argument("--jitter", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0, help="attempt at most N new clips (pilot)")
    ap.add_argument("--cb-window", type=int, default=30)
    ap.add_argument("--cb-threshold", type=float, default=0.4)
    ap.add_argument("--cb-pause", type=int, default=900)
    ap.add_argument("--cb-max-pauses", type=int, default=4)
    ap.add_argument("--cookies", default=None, help="YouTube cookies.txt (Netscape format)")
    a = ap.parse_args()
    global COOKIES
    COOKIES = a.cookies
    VIDEOS.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)

    done, n_ok = set(), 0
    if LOG.exists():
        for l in open(LOG):
            r = json.loads(l)
            if r["status"] != "blocked":
                done.add(r["clip_key"])
            n_ok += r["status"] == "success"
    todo = [r for r in ordered_pool() if r["clip_key"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"[dl] ffmpeg={FFMPEG}; {n_ok} already succeeded, {len(todo)} candidates left, "
          f"target {a.target}", flush=True)

    stats, recent = Counter(success=n_ok), deque(maxlen=a.cb_window)
    lock, pauses, t0 = threading.Lock(), 0, time.time()

    def job(row):
        time.sleep(random.uniform(0, a.jitter))
        return download_one(row, a.timeout)

    it = iter(todo)
    with open(LOG, "a") as log, ThreadPoolExecutor(a.workers) as ex:
        pending = set()
        while True:
            while len(pending) < a.workers and stats["success"] < a.target:
                row = next(it, None)
                if row is None:
                    break
                pending.add(ex.submit(job, row))
            if not pending:
                break
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                r = fut.result()
                with lock:
                    log.write(json.dumps(r) + "\n")
                    log.flush()
                    stats[r["status"]] += 1
                    recent.append(r["status"] == "blocked")
                    if r["status"] == "success":
                        pauses = 0
                    n = sum(stats.values()) - n_ok
                    if n % 25 == 0:
                        print(f"[dl] {dict(stats)} {n / (time.time() - t0) * 3600:.0f} attempts/h",
                              flush=True)
            if len(recent) == a.cb_window and sum(recent) / a.cb_window >= a.cb_threshold:
                pauses += 1
                if pauses > a.cb_max_pauses:
                    print(f"[dl] still blocked after {a.cb_max_pauses} pauses; exiting. "
                          f"Rerun later to resume. {dict(stats)}", flush=True)
                    break
                print(f"[dl] rate-limited ({sum(recent)}/{a.cb_window} blocked); pause "
                      f"{a.cb_pause}s ({pauses}/{a.cb_max_pauses})", flush=True)
                for f in pending:
                    wait([f])  # drain in-flight work before sleeping
                for f in pending:
                    r = f.result()
                    log.write(json.dumps(r) + "\n")
                    stats[r["status"]] += 1
                pending = set()
                recent.clear()
                time.sleep(a.cb_pause)
    print(f"[dl] done: {dict(stats)} in {time.time() - t0:.0f}s", flush=True)
    write_selection(a.target)


if __name__ == "__main__":
    main()
