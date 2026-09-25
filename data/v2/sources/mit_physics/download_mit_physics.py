#!/usr/bin/env python3
"""Select and download a class-stratified oversample of Moments in Time physics clips (v2 source).

Source: Moments_in_Time_Raw_v2.zip via the ungated HF mirror
https://huggingface.co/datasets/Pai3dot14/Moments_in_Time_Raw_v2_hf (7 byte-parts, 294.7 GB, zip64).
Official route is the request form on http://moments.csail.mit.edu/ (license: non-commercial
research only, no redistribution -- see license.txt, fetched into RAW_ROOT).
305 training classes, 727,305 train + 30,500 val clips, all 3.0 s, ~0.39 MB each, NO captions.

We never download the zip: the parts' signed CDN URLs accept HTTP Range and every member is
compressed on its own, so

  index    : read the zip64 central directory (125 MB) -> zip_index.jsonl (name, method, sizes,
             local-header offset, crc) + the small metadata files (csv, README, license)
  select   : CLASSES (24 object-physics classes) x PER_CLASS clips from the training split, random
             (seed 42), each source video (filename stem) used at most once -> selection_15k.jsonl
  download : Range-fetch each member's compressed bytes, inflate, CRC-check
             -> videos/<class>/<file>.mp4 (resumable)

The validation split is left untouched (candidate v2 eval pool).

Usage:
    python download_mit_physics.py                        # all stages, each resumable / skipped if done
    python download_mit_physics.py --stages select --reselect
    python download_mit_physics.py --per-class 0 --out selection_pool.jsonl   # every clip of CLASSES
"""
import argparse, json, os, random, re, struct, threading, time, zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

REPO = "Pai3dot14/Moments_in_Time_Raw_v2_hf"
PART_PREFIX = "Moments_in_Time_Raw_v2_seg/Moments_in_Time_Raw_v2.zip.part"
RAW_ROOT = "/scratch/network/ssd/junlin/raw/mit_physics"
PART_SIZE = 42949672960
PART_SIZES = [PART_SIZE] * 6 + [36966115017]
TOTAL = sum(PART_SIZES)
PER_CLASS = 600          # x24 = 14,400 candidates, ~3x the final 5K (camera motion / label noise)
SEED = 42
META_FILES = ["trainingSet.csv", "validationSet.csv", "moments_categories.txt", "README.txt", "license.txt"]

# classes whose clips are mostly objects/materials obeying physics (human-action classes such as
# kicking / throwing / landing / descending / flipping are left out)
CLASSES = [
    "falling", "dropping", "bouncing", "rolling", "sliding",                 # rigid-body motion
    "breaking", "cracking", "crushing", "erupting",                         # fracture / burst
    "spinning", "swinging", "rocking", "floating",                          # rotation / oscillation / buoyancy
    "splashing", "spilling", "pouring", "flowing", "dripping",              # fluids
    "overflowing", "draining", "leaking", "bubbling", "boiling", "burning",  # fluids / thermal
]

_tls = threading.local()
_urls, _url_lock = {}, threading.Lock()


def session():
    if not hasattr(_tls, "s"):
        _tls.s = requests.Session()
    return _tls.s


def part_url(k, refresh=False):
    with _url_lock:
        if refresh or k not in _urls:
            r = requests.head(f"https://huggingface.co/datasets/{REPO}/resolve/main/{PART_PREFIX}{k}",
                              allow_redirects=False, timeout=60)
            r.raise_for_status()
            _urls[k] = r.headers["location"]
        return _urls[k]


def read_range(off, n):
    """Bytes [off, off+n) of the virtual concatenated zip (may span parts)."""
    out = bytearray()
    while n > 0:
        k = min(off // PART_SIZE, len(PART_SIZES) - 1)
        lo = off - k * PART_SIZE
        m = min(n, PART_SIZES[k] - lo)
        for attempt in range(8):
            try:
                r = session().get(part_url(k, refresh=attempt > 0),
                                  headers={"Range": f"bytes={lo}-{lo + m - 1}"}, timeout=300)
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


def parse_central_directory(cd):
    rows, p = [], 0
    while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        (_, _, _, meth, _, _, crc, csz, usz, nl, el, cl, _, _, _, lho) = struct.unpack("<HHHHHHIIIHHHHHII", cd[p + 4:p + 46])
        name = cd[p + 46:p + 46 + nl].decode("utf-8", "replace")
        ex, q = cd[p + 46 + nl:p + 46 + nl + el], 0
        while q + 4 <= len(ex):                         # zip64 extra field
            hid, hs = struct.unpack("<HH", ex[q:q + 4])
            if hid == 1:
                d, k = ex[q + 4:q + 4 + hs], 0
                if usz == 0xFFFFFFFF: usz = struct.unpack("<Q", d[k:k + 8])[0]; k += 8
                if csz == 0xFFFFFFFF: csz = struct.unpack("<Q", d[k:k + 8])[0]; k += 8
                if lho == 0xFFFFFFFF: lho = struct.unpack("<Q", d[k:k + 8])[0]; k += 8
            q += 4 + hs
        rows.append({"name": name, "method": meth, "csize": csz, "usize": usz, "offset": lho, "crc": crc})
        p += 46 + nl + el + cl
    return rows


def read_member(e):
    h = read_range(e["offset"], 30)
    assert h[:4] == b"PK\x03\x04", e["name"]
    nl, el = struct.unpack("<HH", h[26:30])
    data = read_range(e["offset"] + 30 + nl + el, e["csize"])
    if e["method"] == 8:
        data = zlib.decompress(data, -15)
    if len(data) != e["usize"] or zlib.crc32(data) != e["crc"]:
        raise RuntimeError(f"size/crc mismatch: {e['name']}")
    return data


def stage_index():
    out = f"{RAW_ROOT}/zip_index.jsonl"
    if not os.path.exists(out):
        tail = read_range(TOTAL - 70000, 70000)
        j = tail.rfind(b"PK\x06\x06")
        assert j >= 0, "zip64 EOCD not found"
        z = struct.unpack("<QHHIIQQQQ", tail[j + 4:j + 56])
        n_ent, cd_size, cd_off = z[6], z[7], z[8]
        rows = parse_central_directory(read_range(cd_off, cd_size))
        assert len(rows) == n_ent, (len(rows), n_ent)
        with open(out + ".tmp", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        os.replace(out + ".tmp", out)
        print(f"[index] {len(rows)} entries -> {out}")
    idx = {json.loads(l)["name"]: json.loads(l) for l in open(out)}
    for fn in META_FILES:
        p = f"{RAW_ROOT}/metadata/{fn}"
        if not os.path.exists(p):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "wb").write(read_member(idx[f"Moments_in_Time_Raw/{fn}"]))
    return idx


def stage_select(idx, reselect, per_class, out):
    if os.path.exists(out) and not reselect:
        return
    rng = random.Random(SEED)
    pools = {c: sorted(n for n in idx if n.startswith(f"Moments_in_Time_Raw/training/{c}/") and n.endswith(".mp4"))
             for c in CLASSES}
    for c in CLASSES:
        rng.shuffle(pools[c])
    # a YouTube/getty source video can appear in several classes: use each stem once,
    # assigning classes in order of pool size (smallest first) so small classes keep their clips
    stem = lambda n: re.sub(r"_\d+\.mp4$", "", n.rsplit("/", 1)[1])
    used, rows = set(), []
    for c in sorted(CLASSES, key=lambda c: len(pools[c])):
        k = 0
        for n in pools[c]:
            if per_class and k == per_class:
                break
            if stem(n) in used:
                continue
            used.add(stem(n))
            k += 1
            e, fn = idx[n], n.rsplit("/", 1)[1]
            rows.append({"id": f"mit_physics__{c}__{fn[:-4]}", "source": "mit_physics", "class": c,
                         "split": "train", "zip_path": n, "bytes": e["usize"],
                         "video": f"{RAW_ROOT}/videos/{c}/{fn}"})
        print(f"[select] {c}: {k}/{len(pools[c])}")
    with open(out + ".tmp", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(out + ".tmp", out)
    print(f"[select] {len(rows)} clips, {sum(r['bytes'] for r in rows) / 1e9:.1f} GB -> {out}")


def fetch(row, idx):
    dst = row["video"]
    if os.path.exists(dst) and os.path.getsize(dst) == row["bytes"]:
        return 0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    data = read_member(idx[row["zip_path"]])
    open(dst + ".part", "wb").write(data)
    os.replace(dst + ".part", dst)
    return len(data)


def stage_download(idx, workers, manifest):
    rows = [json.loads(l) for l in open(manifest)]
    t0, done, got = time.time(), 0, 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(fetch, r, idx) for r in rows]
        for fu in as_completed(futs):
            got += fu.result()
            done += 1
            if done % 1000 == 0 or done == len(rows):
                print(f"[download] {done}/{len(rows)}  {got / 1e9:.2f} GB new  {time.time() - t0:.0f}s", flush=True)
    bad = [r["id"] for r in rows if not (os.path.exists(r["video"]) and os.path.getsize(r["video"]) == r["bytes"])]
    print(f"[download] complete, {len(bad)} missing/size-mismatch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="index,select,download")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--reselect", action="store_true")
    ap.add_argument("--per-class", type=int, default=PER_CLASS, help="0 = every clip of CLASSES")
    ap.add_argument("--out", default="selection_15k.jsonl", help="manifest name under RAW_ROOT")
    a = ap.parse_args()
    manifest = f"{RAW_ROOT}/{a.out}"
    os.makedirs(RAW_ROOT, exist_ok=True)
    st = a.stages.split(",")
    idx = stage_index()
    if "select" in st: stage_select(idx, a.reselect, a.per_class, manifest)
    if "download" in st: stage_download(idx, a.workers, manifest)


if __name__ == "__main__":
    main()
