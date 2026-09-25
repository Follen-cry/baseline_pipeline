#!/usr/bin/env python3
"""Select 5K PhysInOne scenes and download ONE static camera's RGB frames per scene (v2 source).

PhysInOne (github.com/vLAR-group/PhysInOne, CC BY-NC-SA 4.0) ships each scene as one zip
(~0.8 GB: 13 cameras x {rgb, depth, seg} x 150 frames @ 30 fps + json), spread over 16 HF
repos PhysInOneP01..P16 (~111 TB total). Whole-zip download (the official
download_selected.py) would be ~4.5 TB for 5K scenes. Instead this script reads each zip over
HTTP Range: the central directory (~0.7 MB), then the chosen camera's 150 rgb jpgs, which
are stored contiguously (~25 MB, one request), plus caption.txt and the small per-scene json.
Measured ~24.5 MB/scene (varies with frame count: 90/150/300 frames), so ~125 GB for 5K.

Case index: repo_assignment.txt + repo_map.json copied from the PhysInOne GitHub repo
(commit in _meta/physinone_github_commit.txt) into <RAW_ROOT>/_meta/.

Selection (seed 42, --n scenes, default 5K), Train split only (the index is all Train):
  water-fill over Single+DoublePhysics scenes -- each step serves the currently least-covered
  phenomenon, preferring (1) scenes already downloaded by an earlier run, (2) SinglePhysics,
  (3) random. TriplePhysics is skipped (three superimposed phenomena). Each scene gets one
  static camera drawn at random (never CineCamera_Moving); reused scenes keep theirs.

Output per scene: <RAW_ROOT>/<case_id>/{rgb/0000..0149.jpg, caption.txt, recorder_stats.json,
<case>_trajectory.json, blender_<cam>.json, static_camera_list.txt, DONE}.
Manifest: <RAW_ROOT>/selection_<n/1000>k.jsonl (an earlier 10K selection, partly downloaded
before the target dropped to 5K, is kept as selection_10k_superseded.jsonl).

Usage:
    python download_physinone.py --dry-run          # selection only
    python download_physinone.py --limit 3          # smoke test on the first 3 cases
    python download_physinone.py                    # all 5K (resumable)
"""
import argparse, glob, io, json, os, random, sys, threading, time, zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

RAW_ROOT = "/scratch/network/ssd/junlin/raw/physinone"
META = os.path.join(RAW_ROOT, "_meta")
TOKEN_FILE = "/scratch/network/ssd2/junlin/huggingface/token"
N_DEFAULT = 5_000
SEED = 42
FPS = 30
STATIC_CAMS = [f"CineCamera_{i}" for i in range(13) if i != 11]  # repo lists 0-10,12 (+Moving)
KEEP_FILES = ("caption.txt", "recorder_stats.json", "static_camera_list.txt")


# ---------------------------------------------------------------- selection
def load_cases():
    repo_map = json.load(open(os.path.join(META, "repo_map.json")))
    cases = []
    for line in open(os.path.join(META, "repo_assignment.txt")):
        if not line.strip():
            continue
        ue, part = line.rstrip("\n").split("\t")
        p = ue.split("/")                       # /Game/PhysInOne/Scenes/<Split>/<Activity>/<name>.<name>
        split, act, name = p[4], p[5], p[6].split(".")[0]
        cases.append(dict(case_id=name, split=split, activity_type=act,
                          phenomena=name.split("__")[0].split("_"), part_id=part,
                          repo=repo_map[part],
                          zip_path=f"{split}/{act}/{name}_trajectory.zip"))
    return cases


def select(cases, n_total, prior):
    """Water-fill n_total Single+Double Train scenes so every phenomenon is covered as evenly as
    possible. Each step serves the least-covered phenomenon, preferring (1) scenes already
    downloaded (`prior`: case_id -> earlier row, whose camera is kept), (2) SinglePhysics,
    (3) random order."""
    rng = random.Random(SEED)
    pool = [c for c in cases if c["split"] == "Train" and c["activity_type"] in ("SinglePhysics", "DoublePhysics")]
    rng.shuffle(pool)
    rank = lambda c: (c["case_id"] not in prior, c["activity_type"] != "SinglePhysics")
    by_ph = defaultdict(list)
    for c in sorted(pool, key=rank, reverse=True):  # pop() from the end gives the best rank first
        for ph in c["phenomena"]:
            by_ph[ph].append(c)
    chosen, used, cover = [], set(), Counter()
    while len(chosen) < n_total:
        ph = min((p for p in by_ph if by_ph[p]), key=lambda p: (cover[p], p), default=None)
        if ph is None:
            break
        c = by_ph[ph].pop()
        if c["case_id"] in used:
            continue
        used.add(c["case_id"])
        chosen.append(c)
        cover.update(c["phenomena"])
    for c in chosen:
        c["camera"] = prior[c["case_id"]]["camera"] if c["case_id"] in prior else rng.choice(STATIC_CAMS)
        c["out_dir"] = os.path.join(RAW_ROOT, c["case_id"])
        c["fps"] = FPS
    return chosen, cover


# ---------------------------------------------------------------- range-read download
class RangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file via HTTP Range; serves a prefetched span from memory."""
    def __init__(self, sess, url, size):
        self.s, self.u, self.n, self.p = sess, url, size, 0
        self.cache_lo, self.cache = None, b""
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.p
    def seek(self, off, whence=0):
        self.p = {0: off, 1: self.p + off, 2: self.n + off}[whence]
        return self.p
    def fetch(self, lo, hi):
        for attempt in range(6):
            try:
                r = self.s.get(self.u, headers={"Range": f"bytes={lo}-{hi}"}, timeout=120)
                if r.status_code == 206:
                    return r.content
                err = RuntimeError(f"HTTP {r.status_code}")
            except requests.exceptions.RequestException as e:
                err = e
            time.sleep(2 ** attempt)
        raise err
    def prefetch(self, lo, hi):
        self.cache_lo, self.cache = lo, self.fetch(lo, hi)
    def readinto(self, b):
        if self.p >= self.n:
            return 0
        want = min(len(b), self.n - self.p)
        if self.cache_lo is not None and self.cache_lo <= self.p and self.p + want <= self.cache_lo + len(self.cache):
            d = self.cache[self.p - self.cache_lo:self.p - self.cache_lo + want]
        else:
            d = self.fetch(self.p, self.p + want - 1)
        b[:len(d)] = d
        self.p += len(d)
        return len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n", type=int, default=N_DEFAULT, help="number of scenes to select")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rate", type=float, default=2.0, help="max scenes started per second (HF resolve quota)")
    args = ap.parse_args()

    man = os.path.join(RAW_ROOT, f"selection_{args.n // 1000}k.jsonl")
    if os.path.exists(man):
        rows = [json.loads(l) for l in open(man)]
        print(f"[select] reusing {man}: {len(rows)} scenes", flush=True)
    else:
        prior = {}  # scenes already on disk (e.g. from an earlier, larger selection): reuse them
        for old in glob.glob(os.path.join(RAW_ROOT, "selection_*.jsonl")):
            for l in open(old):
                r = json.loads(l)
                if os.path.exists(os.path.join(r["out_dir"], "DONE")):
                    prior[r["case_id"]] = r
        rows, cover = select(load_cases(), args.n, prior)
        print(f"[select] reused {sum(r['case_id'] in prior for r in rows)} already-downloaded scenes", flush=True)
        with open(man, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        act = Counter(r["activity_type"] for r in rows)
        cv = sorted(cover.values())
        print(f"[select] {len(rows)} scenes {dict(act)}; per-phenomenon coverage min/median/max "
              f"{cv[0]}/{cv[len(cv)//2]}/{cv[-1]} over {len(cv)} phenomena -> {man}", flush=True)
    if args.dry_run:
        return 0
    if args.limit:
        rows = rows[:args.limit]

    token = os.environ.get("HF_TOKEN") or open(TOKEN_FILE).read().strip()
    lock = threading.Lock()
    state = {"next": 0.0, "paused_until": 0.0}

    def throttle():
        while True:
            with lock:
                now = time.time()
                wait = max(state["paused_until"], state["next"]) - now
                if wait <= 0:
                    state["next"] = now + 1.0 / args.rate
                    return
            time.sleep(min(wait, 5))

    def resolve(sess, r):
        url = f"https://huggingface.co/datasets/{r['repo']}/resolve/main/{r['zip_path']}"
        for attempt in range(8):
            throttle()
            try:
                h = sess.head(url, headers={"Authorization": f"Bearer {token}"}, allow_redirects=False, timeout=60)
            except requests.exceptions.RequestException:
                time.sleep(2 ** attempt)
                continue
            if h.status_code == 302:
                return h.headers["location"], int(h.headers["x-linked-size"])
            if h.status_code == 429:
                with lock:
                    state["paused_until"] = max(state["paused_until"], time.time() + 305)
                print(f"[rate-limit] 429 on {r['case_id']}, pausing 305s", flush=True)
                continue
            raise RuntimeError(f"resolve HTTP {h.status_code}")
        raise RuntimeError("resolve failed after retries")

    def fetch(r):
        out = r["out_dir"]
        if os.path.exists(os.path.join(out, "DONE")):
            return "skip", 0
        sess = requests.Session()
        url, size = resolve(sess, r)
        rf = RangeFile(sess, url, size)
        z = zipfile.ZipFile(io.BufferedReader(rf, buffer_size=1 << 16))
        root = f"{r['case_id']}_trajectory/"
        infos = z.infolist()
        cam = r["camera"]
        # compare the camera folder name only: many case_ids themselves contain "Moving"
        present = sorted({c for c in (i.filename[len(root):].split("/")[0] for i in infos if "/rgb/" in i.filename)
                          if c != "CineCamera_Moving"})
        if cam not in present:  # camera set varies per scene: fall back deterministically
            if not present:
                raise RuntimeError("no static-camera rgb in zip")
            cam = random.Random(r["case_id"]).choice(present)
        rgb = sorted((i for i in infos if i.filename.startswith(f"{root}{cam}/rgb/")),
                     key=lambda i: i.header_offset)
        rf.prefetch(rgb[0].header_offset, rgb[-1].header_offset + rgb[-1].compress_size + 1024)
        tmp = out + ".partial"
        os.makedirs(os.path.join(tmp, "rgb"), exist_ok=True)
        nbytes = 0
        for i in rgb:
            data = z.read(i)
            nbytes += len(data)
            with open(os.path.join(tmp, "rgb", i.filename.rsplit("/", 1)[-1]), "wb") as f:
                f.write(data)
        want = [root + n for n in KEEP_FILES] + [f"{root}{r['case_id']}_trajectory.json", f"{root}blender_{cam}.json"]
        names = set(z.namelist())
        for n in want:
            if n in names:
                with open(os.path.join(tmp, n[len(root):]), "wb") as f:
                    f.write(z.read(n))
        if os.path.isdir(out):
            import shutil
            shutil.rmtree(out)
        os.rename(tmp, out)
        with open(os.path.join(out, "DONE"), "w") as f:
            f.write(json.dumps(dict(frames=len(rgb), camera=cam, camera_planned=r["camera"],
                                    zip_bytes=size, rgb_bytes=nbytes)))
        return "ok", nbytes

    ok = fail = 0
    total_bytes = 0
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(fetch, r): r for r in rows}
        for i, fu in enumerate(as_completed(futs), 1):
            r = futs[fu]
            try:
                status, nb = fu.result()
                ok += 1
                total_bytes += nb
            except Exception as e:
                fail += 1
                print(f"FAIL {r['case_id']} ({r['repo']}): {type(e).__name__} {str(e)[:150]}", flush=True)
            if i % 250 == 0 or i == len(rows):
                el = time.time() - t0
                print(f"[download] {i}/{len(rows)} ok={ok} fail={fail} {total_bytes/1e9:.1f} GB {el:.0f}s", flush=True)
    print(f"[done] ok={ok} fail={fail}", flush=True)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
