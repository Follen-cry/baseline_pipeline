#!/usr/bin/env python3
"""Select and download a stratified 10K-clip oversample of BAAI Physics-aware videos (v2 source).

Repo: https://huggingface.co/datasets/BAAI-DataCube/Physics-aware-videos (CC-BY-NC-4.0, ungated;
note the org is BAAI-DataCube, not BAAI). 83,223 real fixed-camera clips (5-20 s) in 961 opaque
query folders `aNNNNN/`, captions in metadata/dataset.jsonl (`{video, caption}`, Qwen-VL-72B).
The videos are ONE uncompressed `videos.tar` split into 11 byte-parts (446 GB). We never
download it: the parts' signed CDN URLs accept HTTP Range, so

  meta     : fetch metadata/dataset.jsonl (16 MB)
  index    : walk the 512-byte tar headers by Range reads -> tar_index.jsonl (path, offset, size).
             The tar is cut into SEGMENTS pieces; each piece's first header is found by scanning
             for the ustar magic (with checksum check), and the pieces are walked in parallel.
  select   : join index + captions; physics caption-keyword groups (clips with no keyword are
             dropped); water-fill N_TOTAL over the groups (each clip counted under its rarest
             group), random within a group, at most FOLDER_CAP clips per query folder -> selection_10k.jsonl
  download : Range-fetch each selected member -> videos/<folder>/<file>.mp4 (size-checked, resumable)

Usage:
    python download_baai_physics.py                     # all stages, each resumable / skipped if done
    python download_baai_physics.py --stages index      # one stage
    python download_baai_physics.py --stages select --reselect
"""
import argparse, json, os, random, re, threading, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

REPO = "BAAI-DataCube/Physics-aware-videos"
RAW_ROOT = "/scratch/network/ssd/junlin/raw/baai_physics"
N_TOTAL = 10_000
FOLDER_CAP = 20
MAX_BYTES = 150 * 2**20        # skip the rare huge 4K files
SEED = 42
SEGMENTS = 64
PART_SIZE = 42949672960
PARTS = [f"videos.tar.part_a{c}" for c in "abcdefghijk"]
PART_SIZES = [PART_SIZE] * 10 + [16946257920]
TOTAL = sum(PART_SIZES)

# rarest first: a clip is counted under the first group it matches
KW_GROUPS = [
    ("topple", r"\b(toppl\w*|tips? over|tipping|tipped|knock\w* (over|down))"),
    ("bounce", r"\bbounc\w*"),
    ("slide", r"\bslid\w*"),
    ("swing_spin", r"\b(swing\w*|swung|pendulum\w*|spin\w*|spun|rotat\w*)"),
    ("break", r"\b(shatter\w*|break\w*|broke\w*|crack\w*|burst\w*|explod\w*|explosion\w*|smash\w*)"),
    ("collide", r"\b(collid\w*|collision\w*|crash\w*|hits?|hitting|strik\w*|struck|impact\w*)"),
    ("fall", r"\b(fall\w*|fell|drop\w*)"),
    ("smoke_particle", r"\b(smoke\w*|dust\w*|powder\w*|sand|flour|particle\w*)"),
    ("roll", r"\broll\w*"),
    ("fluid", r"\b(pour\w*|splash\w*|spill\w*|flow\w*|drip\w*|water|liquid\w*|waves?)"),
]
KW_RE = [(g, re.compile(p)) for g, p in KW_GROUPS]

_tls = threading.local()
_urls, _url_lock = {}, threading.Lock()


def session():
    if not hasattr(_tls, "s"):
        _tls.s = requests.Session()
    return _tls.s


def part_url(k, refresh=False):
    with _url_lock:
        if refresh or k not in _urls:
            r = requests.head(f"https://huggingface.co/datasets/{REPO}/resolve/main/{PARTS[k]}",
                              allow_redirects=False, timeout=60)
            r.raise_for_status()
            _urls[k] = r.headers["location"]
        return _urls[k]


def read_range(off, n):
    """Bytes [off, off+n) of the virtual concatenated tar (may span parts)."""
    out = bytearray()
    while n > 0:
        k = min(off // PART_SIZE, len(PARTS) - 1)
        lo = off - k * PART_SIZE
        m = min(n, PART_SIZES[k] - lo)
        for attempt in range(8):
            try:
                r = session().get(part_url(k, refresh=attempt > 0),
                                  headers={"Range": f"bytes={lo}-{lo + m - 1}"}, timeout=120)
                if r.status_code == 206 and len(r.content) == m:
                    break
                if r.status_code == 429:
                    time.sleep(300)
            except requests.RequestException:
                pass
            time.sleep(min(60, 2 ** attempt))
        else:
            raise RuntimeError(f"range read failed at {off}+{m}")
        out += r.content
        off += m
        n -= m
    return bytes(out)


# ---------------------------------------------------------------- tar headers
def parse_header(h):
    """-> (name, size, type) or None if h is not a valid ustar header."""
    if len(h) < 512 or h[257:262] != b"ustar":
        return None
    try:
        chk = int(h[148:156].rstrip(b"\0 ").decode() or "0", 8)
    except ValueError:
        return None
    if chk != sum(h[:148]) + 256 + sum(h[156:512]):
        return None
    name = h[:100].split(b"\0")[0].decode("utf-8", "replace")
    prefix = h[345:500].split(b"\0")[0].decode("utf-8", "replace")
    raw = h[124:136]
    size = int.from_bytes(raw[1:], "big") if raw[0] & 0x80 else int(raw.rstrip(b"\0 ").decode() or "0", 8)
    return ((prefix + "/" + name) if prefix else name), size, h[156:157]


def first_header_at_or_after(start, chunk=16 * 2**20):
    off = start
    while off < TOTAL:
        n = min(chunk, TOTAL - off)
        buf = read_range(off, n)
        for i in range(0, n - 511, 512):
            if buf[i + 257:i + 262] == b"ustar" and parse_header(buf[i:i + 512]):
                return off + i
        off += n
    return TOTAL


def walk_segment(i, start, end, out_dir):
    """Walk headers from `start` (a header) until the next header offset >= end."""
    path = f"{out_dir}/seg{i:03d}.jsonl"
    if os.path.exists(path + ".done"):
        return i, 0
    off, n = start, 0
    if os.path.exists(path):                       # resume after the last entry
        last = None
        for line in open(path):
            last = json.loads(line)
            n += 1
        if last:
            off = last["offset"] + 512 + (last["size"] + 511) // 512 * 512
    long_name = None
    with open(path, "a") as f:
        while off < end:
            h = read_range(off, 512)
            if h == b"\0" * 512:                    # end-of-archive
                break
            p = parse_header(h)
            if p is None:
                raise RuntimeError(f"seg {i}: bad header at {off}")
            name, size, typ = p
            data_blocks = (size + 511) // 512 * 512
            if typ == b"L":                         # GNU long name: next header uses it
                long_name = read_range(off + 512, size).split(b"\0")[0].decode()
            elif typ in (b"0", b"\0"):
                f.write(json.dumps({"path": long_name or name, "offset": off, "size": size}) + "\n")
                f.flush()
                n += 1
                long_name = None
            off += 512 + data_blocks
    open(path + ".done", "w").write(str(off))
    return i, n


def stage_meta():
    p = f"{RAW_ROOT}/metadata/dataset.jsonl"
    if os.path.exists(p):
        return
    os.makedirs(os.path.dirname(p), exist_ok=True)
    r = requests.get(f"https://huggingface.co/datasets/{REPO}/resolve/main/metadata/dataset.jsonl", timeout=300)
    r.raise_for_status()
    open(p + ".tmp", "wb").write(r.content)
    os.replace(p + ".tmp", p)
    print(f"[meta] {sum(1 for _ in open(p))} rows")


def stage_index(workers):
    out = f"{RAW_ROOT}/tar_index.jsonl"
    if os.path.exists(out):
        return
    seg_dir = f"{RAW_ROOT}/_index_segments"
    os.makedirs(seg_dir, exist_ok=True)
    starts_p = f"{seg_dir}/starts.json"
    if os.path.exists(starts_p):
        starts = json.load(open(starts_p))
    else:
        cuts = [TOTAL * s // SEGMENTS // 512 * 512 for s in range(SEGMENTS)]
        with ThreadPoolExecutor(workers) as ex:
            starts = [0] + list(ex.map(first_header_at_or_after, cuts[1:]))
        starts = sorted(set(starts))
        json.dump(starts, open(starts_p, "w"))
    bounds = list(zip(starts, starts[1:] + [TOTAL]))
    print(f"[index] {len(bounds)} segments")
    t0, total = time.time(), 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(walk_segment, i, s, e, seg_dir) for i, (s, e) in enumerate(bounds)]
        for k, fu in enumerate(as_completed(futs), 1):
            i, n = fu.result()
            total += n
            print(f"[index] seg {i} done ({k}/{len(bounds)}), {time.time() - t0:.0f}s", flush=True)
    rows = {}
    for i in range(len(bounds)):
        for line in open(f"{seg_dir}/seg{i:03d}.jsonl"):
            r = json.loads(line)
            rows[r["path"]] = r
    with open(out + ".tmp", "w") as f:
        for r in sorted(rows.values(), key=lambda r: r["offset"]):
            f.write(json.dumps(r) + "\n")
    os.replace(out + ".tmp", out)
    print(f"[index] {len(rows)} files -> {out}")


def stage_select(reselect):
    out = f"{RAW_ROOT}/selection_10k.jsonl"
    if os.path.exists(out) and not reselect:
        return
    idx = {json.loads(l)["path"]: json.loads(l) for l in open(f"{RAW_ROOT}/tar_index.jsonl")}
    meta = [json.loads(l) for l in open(f"{RAW_ROOT}/metadata/dataset.jsonl")]
    pools = defaultdict(list)
    miss = 0
    for m in meta:
        e = idx.get("videos/" + m["video"])
        if e is None:
            miss += 1
            continue
        if e["size"] > MAX_BYTES:
            continue
        cap = m["caption"].lower()
        groups = [g for g, rx in KW_RE if rx.search(cap)]
        if groups:
            pools[groups[0]].append((m, e, groups))
    print(f"[select] {len(meta)} captions, {miss} not in tar, eligible per group:",
          {g: len(pools[g]) for g, _ in KW_GROUPS})
    rng = random.Random(SEED)
    for g in pools:
        pools[g].sort(key=lambda x: x[0]["video"])
        rng.shuffle(pools[g])
    per_folder, picked, ptr = Counter(), defaultdict(list), {g: 0 for g in pools}
    active = [g for g, _ in KW_GROUPS if pools.get(g)]
    while sum(map(len, picked.values())) < N_TOTAL and active:
        for g in list(active):                     # round-robin = water-filling
            while ptr[g] < len(pools[g]):
                m, e, groups = pools[g][ptr[g]]
                ptr[g] += 1
                folder = m["video"].split("/")[0]
                if per_folder[folder] < FOLDER_CAP:
                    per_folder[folder] += 1
                    picked[g].append((m, e, groups))
                    break
            else:
                active.remove(g)
            if sum(map(len, picked.values())) >= N_TOTAL:
                break
    with open(out + ".tmp", "w") as f:
        for g, _ in KW_GROUPS:
            for m, e, groups in picked[g]:
                folder, fn = m["video"].split("/")
                f.write(json.dumps({
                    "id": f"baai_physics__{folder}__{fn[:-4]}", "source": "baai_physics",
                    "query_folder": folder, "kw_group": g, "kw_groups": groups,
                    "caption": m["caption"], "tar_path": "videos/" + m["video"],
                    "tar_offset": e["offset"], "bytes": e["size"],
                    "video": f"{RAW_ROOT}/videos/{m['video']}"}) + "\n")
    os.replace(out + ".tmp", out)
    n = sum(map(len, picked.values()))
    gb = sum(e["size"] for p in picked.values() for _, e, _ in p) / 1e9
    print(f"[select] {n} clips, {len(per_folder)} folders, {gb:.1f} GB:",
          {g: len(picked[g]) for g, _ in KW_GROUPS})


def fetch(row):
    dst = row["video"]
    if os.path.exists(dst) and os.path.getsize(dst) == row["bytes"]:
        return 0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    data = read_range(row["tar_offset"] + 512, row["bytes"])
    tmp = dst + ".part"
    open(tmp, "wb").write(data)
    os.replace(tmp, dst)
    return len(data)


def stage_download(workers):
    rows = [json.loads(l) for l in open(f"{RAW_ROOT}/selection_10k.jsonl")]
    t0, done, got = time.time(), 0, 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(fetch, r) for r in rows]
        for fu in as_completed(futs):
            got += fu.result()
            done += 1
            if done % 200 == 0 or done == len(rows):
                el = time.time() - t0
                print(f"[download] {done}/{len(rows)}  {got / 1e9:.1f} GB new  {got / 1e6 / max(el, 1):.1f} MB/s",
                      flush=True)
    bad = [r["id"] for r in rows if not (os.path.exists(r["video"]) and os.path.getsize(r["video"]) == r["bytes"])]
    print(f"[download] complete, {len(bad)} missing/size-mismatch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="meta,index,select,download")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--reselect", action="store_true")
    a = ap.parse_args()
    os.makedirs(RAW_ROOT, exist_ok=True)
    st = a.stages.split(",")
    if "meta" in st: stage_meta()
    if "index" in st: stage_index(a.workers)
    if "select" in st: stage_select(a.reselect)
    if "download" in st: stage_download(min(a.workers, 16))


if __name__ == "__main__":
    main()
