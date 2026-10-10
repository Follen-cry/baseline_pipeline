#!/usr/bin/env python3
"""Select and download a stratified 10K-video oversample of PhysicTran38K (v2 source).

Repo: https://huggingface.co/datasets/metazlb/PhysicTran38K (Apache-2.0, model-generated videos).
Layout: <Domain>/<SubDomain>/<transition_type>/{N.mp4, metadata.json, final_filter_videos.txt,
filtered_videos.txt, moving_videos.txt, ...} -- 46 transition types, ~1000 videos each.

Selection (seeded):
  tier A = every video in the authors' final_filter_videos.txt (7,371 in total -- fewer than 10K)
  tier B = top-up to N_TOTAL, filling the types with the fewest tier-A videos first so the 46
           types end up as balanced as possible; within a type prefer filtered_videos.txt
           ("B1") over the rest ("B2").
Each row records tier, whether the video is in moving_videos.txt (meaning undocumented --
decide after measuring camera motion), and the generation prompt (which states the final
state, so it is NOT usable as a v2 caption as-is).

Full repo (--all): every mp4 of every type -> manifest_all.jsonl (same row schema; `tier` A / B1 /
"unlisted" from the authors' lists, `in_selection_10k` marks the frozen 10K). selection_10k.jsonl
is untouched, so the v2 pools (which read it) do not change.

Usage:
    python download_physictran38k.py            # reuse selection_10k.jsonl (or build it) + download, resumable
    python download_physictran38k.py --all      # list the whole repo -> manifest_all.jsonl + download, resumable
    python download_physictran38k.py --reselect # rebuild the selection from the repo listing
    python download_physictran38k.py --dry-run  # selection + manifest only
"""
import argparse, json, os, random, sys, threading, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Free HF accounts get 1000 API requests / 5 min. The Xet path spends one API call per file
# (xet-read-token), which exhausted the quota after ~1000 files; plain HTTP `resolve`
# downloads count against the separate, larger resolver quota instead.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile, RepoFolder

REPO = "metazlb/PhysicTran38K"
RAW_ROOT = "/scratch/network/ssd/junlin/raw/physictran38k"
TOKEN_FILE = "/scratch/network/ssd2/junlin/huggingface/token"
N_TOTAL = 10_000
SEED = 42
LIST_FILES = ["final_filter_videos.txt", "filtered_videos.txt", "moving_videos.txt", "metadata.json"]


def token():
    return os.environ.get("HF_TOKEN") or open(TOKEN_FILE).read().strip()


def list_types(api):
    ls = lambda p=None: list(api.list_repo_tree(REPO, path_in_repo=p, repo_type="dataset", token=token()))
    types = []
    for d in [x.path for x in ls() if isinstance(x, RepoFolder)]:
        for s in [x.path for x in ls(d) if isinstance(x, RepoFolder)]:
            types += [x.path for x in ls(s) if isinstance(x, RepoFolder)]
    return sorted(types)


def load_type(api, t):
    """mp4 inventory from the repo + the per-type lists/metadata (downloaded into RAW_ROOT)."""
    entries = list(api.list_repo_tree(REPO, path_in_repo=t, repo_type="dataset", token=token()))
    mp4 = {x.path.split("/")[-1]: x.size for x in entries if isinstance(x, RepoFile) and x.path.endswith(".mp4")}
    present = {x.path.split("/")[-1] for x in entries if isinstance(x, RepoFile)}
    lists = {}
    for n in LIST_FILES:
        if n not in present:
            lists[n] = None
            continue
        p = hf_hub_download(REPO, f"{t}/{n}", repo_type="dataset", local_dir=RAW_ROOT, token=token())
        lists[n] = json.load(open(p)) if n.endswith(".json") else set(open(p).read().split())
    return mp4, lists


def select(types_info):
    rng = random.Random(SEED)
    chosen = {t: {} for t in types_info}          # t -> {video: tier}
    pools = {}
    for t, (mp4, L) in types_info.items():
        final = sorted(v for v in (L["final_filter_videos.txt"] or set()) if v in mp4)
        for v in final:
            chosen[t][v] = "A"
        b1 = [v for v in sorted(L["filtered_videos.txt"] or set()) if v in mp4 and v not in chosen[t]]
        b2 = [v for v in sorted(mp4) if v not in chosen[t] and v not in set(b1)]
        rng.shuffle(b1); rng.shuffle(b2)
        pools[t] = [(v, "B1") for v in b1] + [(v, "B2") for v in b2]
    total = sum(len(c) for c in chosen.values())
    # water-filling: repeatedly give one video to the type with the fewest selected so far
    while total < N_TOTAL:
        open_types = [t for t in types_info if pools[t]]
        if not open_types:
            break
        t = min(open_types, key=lambda x: (len(chosen[x]), x))
        v, tier = pools[t].pop(0)
        chosen[t][v] = tier
        total += 1
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reselect", action="store_true",
                    help="rebuild selection_10k.jsonl from the repo instead of reusing it")
    ap.add_argument("--all", action="store_true",
                    help="every video in the repo -> manifest_all.jsonl (reused if present) + download")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--rate", type=float, default=2.5, help="max file downloads started per second")
    args = ap.parse_args()
    os.makedirs(RAW_ROOT, exist_ok=True)
    man = os.path.join(RAW_ROOT, "selection_10k.jsonl")
    if args.all:
        man_all = os.path.join(RAW_ROOT, "manifest_all.jsonl")
        if os.path.exists(man_all) and not args.reselect:
            rows = [json.loads(l) for l in open(man_all)]
            print(f"[all] reusing {man_all}: {len(rows)} videos", flush=True)
        else:
            rows = build_all(man_all, man)
    elif os.path.exists(man) and not args.reselect:
        # reuse the frozen selection: no repo listing, so no API quota spent before downloading
        rows = [json.loads(l) for l in open(man)]
        print(f"[select] reusing {man}: {len(rows)} videos", flush=True)
    else:
        rows = build_selection(man)
    if args.dry_run:
        return
    download(rows, args)


def list_repo():
    api = HfApi()
    types = list_types(api)
    print(f"[select] {len(types)} transition types", flush=True)
    info = {}
    with ThreadPoolExecutor(8) as ex:
        for t, res in zip(types, ex.map(lambda t: load_type(api, t), types)):
            info[t] = res
    return types, info


def manifest_rows(types, info, chosen):
    """chosen: t -> {video: tier}; one manifest row per chosen video, in type / index order."""
    rows = []
    for t in types:
        mp4, L = info[t]
        meta = {f"{m['idx']}.mp4": m for m in (L["metadata.json"] or [])}
        moving = L["moving_videos.txt"] or set()
        domain, subdomain, trans = t.split("/")
        for v, tier in sorted(chosen[t].items(), key=lambda kv: int(kv[0].split(".")[0])):
            m = meta.get(v, {})
            rows.append(dict(
                id=f"physictran38k__{trans}__{v[:-4]}", source="physictran38k",
                domain=domain, subdomain=subdomain, transition=trans,
                repo_path=f"{t}/{v}", video=os.path.join(RAW_ROOT, t, v), bytes=mp4[v],
                tier=tier, in_moving_list=v in moving,
                prompt=m.get("prompt"), state=m.get("State"), transition_name=m.get("Transition")))
    return rows


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def build_all(man_all, man_10k):
    types, info = list_repo()
    chosen = {}
    for t, (mp4, L) in info.items():
        final, filt = L["final_filter_videos.txt"] or set(), L["filtered_videos.txt"] or set()
        chosen[t] = {v: "A" if v in final else "B1" if v in filt else "unlisted" for v in mp4}
    rows = manifest_rows(types, info, chosen)
    sel = {r["id"] for r in map(json.loads, open(man_10k))} if os.path.exists(man_10k) else set()
    for r in rows:
        r["in_selection_10k"] = r["id"] in sel
    write_jsonl(man_all, rows)
    by_tier = defaultdict(int)
    for r in rows:
        by_tier[r["tier"]] += 1
    print(f"[all] {len(rows)} videos in {len(types)} types, tiers {dict(by_tier)}, "
          f"{sum(r['in_selection_10k'] for r in rows)} in selection_10k (of {len(sel)}), "
          f"{sum(r['bytes'] for r in rows)/1e9:.2f} GB -> {man_all}", flush=True)
    return rows


def build_selection(man):
    types, info = list_repo()
    chosen = select(info)
    rows = manifest_rows(types, info, chosen)
    write_jsonl(man, rows)
    by_tier = defaultdict(int)
    for r in rows:
        by_tier[r["tier"]] += 1
    per_type = sorted(len(c) for c in chosen.values())
    print(f"[select] {len(rows)} videos, tiers {dict(by_tier)}, per-type min/median/max "
          f"{per_type[0]}/{per_type[len(per_type)//2]}/{per_type[-1]}, "
          f"{sum(r['bytes'] for r in rows)/1e9:.2f} GB -> {man}", flush=True)
    return rows


def download(rows, args):
    lock = threading.Lock()
    state = {"next": 0.0, "paused_until": 0.0}

    def throttle():
        """global rate limit (args.rate files/s) + shared pause after any 429"""
        while True:
            with lock:
                now = time.time()
                wait = max(state["paused_until"], state["next"]) - now
                if wait <= 0:
                    state["next"] = now + 1.0 / args.rate
                    return
            time.sleep(min(wait, 5))

    def fetch(r):
        dst = r["video"]
        if os.path.exists(dst) and os.path.getsize(dst) == r["bytes"]:
            return "skip"
        err = None
        for attempt in range(8):
            throttle()
            try:
                hf_hub_download(REPO, r["repo_path"], repo_type="dataset", local_dir=RAW_ROOT, token=token())
                return "ok"
            except Exception as e:
                err = e
                if "429" in str(e):  # quota window is 5 min: pause every worker until it resets
                    with lock:
                        state["paused_until"] = max(state["paused_until"], time.time() + 305)
                    print(f"[rate-limit] 429 on {r['repo_path']}, pausing 305s", flush=True)
                else:
                    time.sleep(2 ** attempt)
        return f"FAIL {r['repo_path']}: {type(err).__name__} {str(err)[:120]}"

    done = fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(fetch, r) for r in rows]
        for i, fu in enumerate(as_completed(futs), 1):
            res = fu.result()
            if res.startswith("FAIL"):
                fail += 1
                print(res, flush=True)
            else:
                done += 1
            if i % 500 == 0 or i == len(rows):
                print(f"[download] {i}/{len(rows)} ok={done} fail={fail} {time.time()-t0:.0f}s", flush=True)
    print(f"[done] ok={done} fail={fail}", flush=True)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
