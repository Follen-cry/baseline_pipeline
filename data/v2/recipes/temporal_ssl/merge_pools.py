#!/usr/bin/env python3
"""Build v2 window pools (the input of T0-T4) from the 8 v2 sources.

A *run* is one pool, named by --name. The default run `main` is the 60K mix: TOTAL = 60,000 train
windows over 6 sources split by SHARES (PhysInOne, PhysicTran38K, VBVR, BAAI, MiT, SSv2), plus EVAL_CLIPS clips per source held out (whole clips, one window each).
Other runs reuse everything per source, e.g. extra windows from one source as a separate training
set (see Usage). Each window = 4 frames F0..F3 at t, t+Δt, t+2Δt, t+3Δt from the global rule in
common/window_sampler.py (Δt ∈ {0.5, 1, 2} s, full clip, no trimming).

Stages (each per source, resumable / deterministic; `--stage all` runs them in order):

  probe     decoded frame count + fps of every candidate clip, with the same function that
            extract_frames.py uses -> <work>/probe_<src>.jsonl (shared by all runs). VBVR (134K
            train clips) only probes a task-stratified candidate subset of
            VBVR_CANDIDATE_FACTOR x the run's need; a bigger run's subset contains a smaller one's.
  score     change scores of every window of every candidate clip (common/window_motion.py:
            low-res gray in memory, nothing written) -> <work>/score_<src>.jsonl (shared). Static
            windows (none of F1..F3 differs from F0 in >= STATIC_THR of pixels) are dropped before
            plan, so the quota is filled from the remaining windows and no oversampling is needed.
  plan      per source, seeded, over the non-static windows only:
            1. eval: EVAL_CLIPS clips, round-robin over categories, 1 window each (the Δt least
               used so far), or with --eval-from RUN exactly that run's eval clips + windows.
               Eval clips are never used for train.
            2. train: Δt targets = the quota split as evenly as the source's windows allow
               (water-fill: a Δt with too few windows gives its share to the others). Clips,
               interleaved by category, get windows in rounds (every clip one more window per
               round) until the quota is met. Within a clip the next window is the Δt furthest
               below its target (ties: the scarcest Δt), then the one overlapping least in time with the clip's chosen
               windows. --exclude-from RUN drops that run's train windows from the candidates.
            -> <work>/runs/<name>/plan_<src>.jsonl
  extract   common/extract_frames.py on the plan -> frames under FRAMES_ROOT/<src>/<clip>/
            (shared: a frame already on disk is not written again),
            window rows -> <work>/runs/<name>/windows_<src>.jsonl
  finalize  final rows + checks -> datasets/temporal_ssl/pools/<name>/<src>_{train,eval}.jsonl
            and pools/<name>/summary.json

Pool row (one per window; frames, frame_idx, timestamps are chronological, aligned with `order`):
    id, source, clip_id, split, caption, caption_source,
    gap_s, gap_frames, order ["F0".."F3"], frames, frame_idx, timestamps,
    window {k, start_s, end_s}, clip {video, fps, num_frames, duration, category},
    motion {change_vs_f0 [F0-F1, F0-F2, F0-F3], change_adjacent [F0-F1, F1-F2, F2-F3],
            stalled (an adjacent pair below STATIC_THR)}, source_meta

Usage:
    # the 60K mix
    python merge_pools.py --stage all --procs 48
    # 20K more PhysInOne windows as a separate set: same eval clips as main, no main train window
    python merge_pools.py --stage all --name physinone_extra --sources physinone \\
        --quota physinone=20000 --eval-from main --exclude-from main
"""
import argparse, collections, json, os, random, subprocess, sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.normpath(os.path.join(HERE, "../.."))
sys.path.insert(0, os.path.join(V2, "common"))
from paths import FRAMES_ROOT  # noqa: E402
from window_sampler import sample_windows  # noqa: E402
from window_motion import STATIC_THR, is_static, score_clip  # noqa: E402
import clip_sources  # noqa: E402

TOTAL = 60_000
# Relative weights of the sources in `main`, normalised to shares. PhyCo-Sim and Panda-70M were
# dropped on 2026-09-26 (loaders stay in clip_sources.py); the other six keep their original
# relative weights (15 / 20 / 15 / 15 / 10 / 10).
WEIGHTS = {"physinone": 15, "physictran38k": 20, "vbvr": 15, "baai_physics": 15, "mit_physics": 10, "ssv2": 10}
SHARES = {s: w / sum(WEIGHTS.values()) for s, w in WEIGHTS.items()}
EVAL_CLIPS = 125
VBVR_CANDIDATE_FACTOR = 1.2
SEED = 42
ORDER = ["F0", "F1", "F2", "F3"]

WORK = os.path.join(FRAMES_ROOT, "_work", "temporal_ssl")
POOLS = os.path.join(V2, "datasets", "temporal_ssl", "pools")
EXTRACT = os.path.join(V2, "common", "extract_frames.py")


class Run:
    """One pool: name, per-source train quota, eval clips, runs to take eval from / exclude."""

    def __init__(self, name="main", quotas=None, total=TOTAL, eval_clips=EVAL_CLIPS,
                 eval_from=None, exclude_from=()):
        self.name, self.total, self.eval_clips = name, total, eval_clips
        self.quotas = {s: round(total * sh) for s, sh in SHARES.items()}
        self.quotas.update(quotas or {})
        self.eval_from, self.exclude_from = eval_from, list(exclude_from)
        self.work = os.path.join(WORK, "runs", name)
        self.pools = os.path.join(POOLS, name)

    def quota(self, src):
        return self.quotas[src]

    def config(self):
        return {"name": self.name, "quotas": self.quotas, "eval_clips": self.eval_clips,
                "eval_from": self.eval_from, "exclude_from": self.exclude_from, "seed": SEED,
                "static_thr": STATIC_THR}


def read_jsonl(p):
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(l) for l in f]


def write_jsonl(p, rows):
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, p)


def interleave(clips, rng):
    """Shuffle within category, then round-robin over categories (category order shuffled)."""
    by = collections.defaultdict(list)
    for c in clips:
        by[c["category"]].append(c)
    cats = sorted(by)
    rng.shuffle(cats)
    for c in cats:
        rng.shuffle(by[c])
    out = []
    while any(by[c] for c in cats):
        for c in cats:
            if by[c]:
                out.append(by[c].pop())
    return out


def candidates(src, run):
    """Clips to probe/plan from. All clips, except VBVR: per task a seeded prefix of a fixed
    shuffle, sized to the run's need (so a larger run's candidates contain a smaller run's)."""
    clips = clip_sources.load(src)
    if src != "vbvr":
        return clips
    rng = random.Random(SEED)
    by = collections.defaultdict(list)
    for c in clips:
        by[c["category"]].append(c)
    per_task = -(-int((run.quota(src) + run.eval_clips) * VBVR_CANDIDATE_FACTOR) // len(by))
    out = []
    for t in sorted(by):
        rs = sorted(by[t], key=lambda c: c["clip_id"])
        rng.shuffle(rs)
        out += rs[:per_task]
    return out


# ---------------------------------------------------------------- probe
def _probe_one(c):
    from extract_frames import clip_length
    n, fps, _ = clip_length(c["video"], c.get("fps"))
    return {"clip_id": c["clip_id"], "num_frames": n, "fps": fps}


def probe(src, run, procs):
    path = os.path.join(WORK, f"probe_{src}.jsonl")
    done = {r["clip_id"] for r in read_jsonl(path)}
    todo = [c for c in candidates(src, run) if c["clip_id"] not in done]
    print(f"[probe] {src}: {len(done)} cached, {len(todo)} to probe", flush=True)
    with open(path, "a") as f, Pool(procs) as pool:
        for i, r in enumerate(pool.imap_unordered(_probe_one, todo, chunksize=8), 1):
            f.write(json.dumps(r) + "\n")
            if i % 2000 == 0:
                f.flush()
                print(f"[probe] {src}: {i}/{len(todo)}", flush=True)


# ---------------------------------------------------------------- score
def _score_one(a):
    c, L = a
    ws = sample_windows(L["num_frames"], L["fps"]) if L["fps"] > 0 else []
    out = {"clip_id": c["clip_id"], "num_frames": L["num_frames"], "fps": L["fps"]}
    if not ws:
        return {**out, "status": "no_window", "windows": []}
    sc = score_clip(c["video"], ws)
    if sc is None:
        return {**out, "status": "decode_fail", "windows": []}
    return {**out, "status": "ok", "windows": sc}


def score(src, run, procs):
    path = os.path.join(WORK, f"score_{src}.jsonl")
    length = {r["clip_id"]: r for r in read_jsonl(os.path.join(WORK, f"probe_{src}.jsonl"))}
    done = {r["clip_id"] for r in read_jsonl(path)}
    todo = [(c, length[c["clip_id"]]) for c in candidates(src, run) if c["clip_id"] not in done]
    print(f"[score] {src}: {len(done)} cached, {len(todo)} to score", flush=True)
    with open(path, "a") as f, Pool(procs) as pool:
        for i, r in enumerate(pool.imap_unordered(_score_one, todo, chunksize=4), 1):
            f.write(json.dumps(r) + "\n")
            if i % 1000 == 0:
                f.flush()
                print(f"[score] {src}: {i}/{len(todo)}", flush=True)


# ---------------------------------------------------------------- plan
def _overlap(a, b):
    lo = max(a["start_s"], b["start_s"])
    hi = min(a["start_s"] + 3 * a["dt"], b["start_s"] + 3 * b["dt"])
    return max(0.0, hi - lo)


def dt_targets(q, avail):
    """Split q over Δt as evenly as the available window counts allow (water-fill)."""
    tgt = {d: 0.0 for d in avail}
    left, open_ = float(q), [d for d in avail if avail[d] > 0]
    while left > 1e-9 and open_:
        share = left / len(open_)
        for d in list(open_):
            g = min(share, avail[d] - tgt[d])
            tgt[d] += g
            left -= g
            if avail[d] - tgt[d] <= 1e-9:
                open_.remove(d)
    return tgt


def _pick(avail, chosen, dt_count, target, rng, scarcity=None):
    """Δt furthest below its target first; on a tie the scarcest Δt (largest share of its available
    windows needed), so clips that can give a rare Δt give it; then least time overlap in the clip."""
    scarcity = scarcity or {}

    def key(w):
        t = target.get(w["dt"], 0.0)
        return (dt_count[w["dt"]] / t if t > 0 else float("inf"), -scarcity.get(w["dt"], 0.0),
                sum(_overlap(w, p) for p in chosen), rng.random())
    return min(avail, key=key)


def _run_windows(run_name, src, split):
    """{clip_id: [(dt, k), ...]} of one split of another run's plan."""
    out = collections.defaultdict(list)
    for r in read_jsonl(os.path.join(WORK, "runs", run_name, f"plan_{src}.jsonl")):
        if r["split"] == split:
            out[r["clip_id"]] += [(float(dt), int(k)) for dt, k in r["windows"]]
    return out


def plan(src, run):
    rng = random.Random(f"{SEED}:{src}:{run.name}")
    scored = {r["clip_id"]: r for r in read_jsonl(os.path.join(WORK, f"score_{src}.jsonl"))}
    excluded = set()
    for other in run.exclude_from:
        for cid, ws in _run_windows(other, src, "train").items():
            excluded |= {(cid, *w) for w in ws}
    clips, n_all, n_static = [], 0, 0
    for c in candidates(src, run):
        S = scored.get(c["clip_id"])
        if S is None:
            raise SystemExit(f"{src}: {c['clip_id']} not scored; run --stage probe, score first")
        sc = {(w["dt"], w["k"]): w for w in S["windows"]}
        ws = [w for w in sample_windows(S["num_frames"], S["fps"]) if (w["dt"], w["k"]) in sc] if sc else []
        n_all += len(ws)
        keep = [{**w, "_score": sc[(w["dt"], w["k"])]} for w in ws if not is_static(sc[(w["dt"], w["k"])])]
        n_static += len(ws) - len(keep)
        if keep:
            clips.append({**c, "_len": S, "_ws": keep})
    order = interleave(clips, rng)

    picked = {c["clip_id"]: [] for c in order}
    if run.eval_from:
        ev = _run_windows(run.eval_from, src, "eval")
        if not ev:
            raise SystemExit(f"{src}: run '{run.eval_from}' has no eval plan")
        eval_clips = [c for c in order if c["clip_id"] in ev]
        for c in eval_clips:
            picked[c["clip_id"]] = [w for w in c["_ws"] if (w["dt"], w["k"]) in set(ev[c["clip_id"]])]
        if len(eval_clips) != len(ev):
            print(f"[plan] WARNING {src}: {len(ev) - len(eval_clips)} eval clips of '{run.eval_from}' "
                  f"are not candidates here", flush=True)
    else:
        eval_clips = order[:run.eval_clips]
        dt_eval = collections.Counter()
        for c in eval_clips:
            w = _pick(c["_ws"], [], dt_eval, {d: 1.0 for d in (0.5, 1.0, 2.0)}, rng)
            picked[c["clip_id"]].append(w)
            dt_eval[w["dt"]] += 1
    eval_ids = {c["clip_id"] for c in eval_clips}
    train_clips = []
    for c in order:
        if c["clip_id"] in eval_ids:
            continue
        ws = [w for w in c["_ws"] if (c["clip_id"], w["dt"], w["k"]) not in excluded]
        if ws:
            train_clips.append({**c, "_ws": ws})

    Q = run.quota(src)
    avail_dt = collections.Counter(w["dt"] for c in train_clips for w in c["_ws"])
    target = dt_targets(Q, avail_dt)
    scarcity = {d: target[d] / avail_dt[d] for d in target if avail_dt[d]}
    got, dt_train = 0, collections.Counter()
    while got < Q:
        progress = False
        for c in train_clips:
            chosen = picked[c["clip_id"]]
            avail = [w for w in c["_ws"] if w not in chosen]
            if not avail:
                continue
            w = _pick(avail, chosen, dt_train, target, rng, scarcity)
            chosen.append(w)
            dt_train[w["dt"]] += 1
            got += 1
            progress = True
            if got == Q:
                break
        if not progress:
            break
    if got < Q:
        print(f"[plan] WARNING {src}: only {got} of {Q} train windows available", flush=True)

    rows = []
    for split, group in (("eval", eval_clips), ("train", train_clips)):
        for c in group:
            ws = picked[c["clip_id"]]
            if not ws:
                continue
            row = {k: v for k, v in c.items() if not k.startswith("_")}
            ws = sorted(ws, key=lambda w: (w["dt"], w["k"]))
            row.update(split=split, num_frames_expected=c["_len"]["num_frames"],
                       windows=[[w["dt"], w["k"]] for w in ws],
                       window_scores={f"{w['dt']}_{w['k']}": {k: w["_score"][k] for k in
                                      ("change_vs_f0", "change_adjacent")} for w in ws})
            rows.append(row)
    os.makedirs(run.work, exist_ok=True)
    write_jsonl(os.path.join(run.work, f"plan_{src}.jsonl"), rows)
    per_clip = collections.Counter(len(r["windows"]) for r in rows if r["split"] == "train")
    dt_eval = collections.Counter(dt for r in rows if r["split"] == "eval" for dt, _ in r["windows"])
    print(f"[plan] {run.name}/{src}: static windows dropped {n_static}/{n_all} ({n_static / max(n_all, 1):.1%}); "
          f"excluded {len(excluded)}; eligible clips {len(clips)}, train {got}/{Q} windows on "
          f"{sum(per_clip.values())} clips {dict(sorted(per_clip.items()))}, "
          f"dt {dict(sorted(dt_train.items()))} (target {({d: round(t) for d, t in sorted(target.items())})}); "
          f"eval {sum(dt_eval.values())} {dict(sorted(dt_eval.items()))}", flush=True)


# ---------------------------------------------------------------- extract
def extract(src, run, procs):
    cmd = [sys.executable, EXTRACT, "--clips", os.path.join(run.work, f"plan_{src}.jsonl"),
           "--out", os.path.join(run.work, f"windows_{src}.jsonl"), "--procs", str(procs)]
    print("[extract]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=os.path.join(V2, "common"))


# ---------------------------------------------------------------- finalize
def final_row(w):
    idx, ts, fps = w["frame_idx"], w["timestamps"], w["fps"]
    return {"id": f"{w['source']}__{w['window_id']}", "source": w["source"], "clip_id": w["clip_id"],
            "split": w["split"], "caption": w["caption"], "caption_source": w["caption_source"],
            "gap_s": w["dt"], "gap_frames": [b - a for a, b in zip(idx, idx[1:])],
            "order": ORDER, "frames": w["frames"], "frame_idx": idx, "timestamps": ts,
            "window": {"k": w["k"], "start_s": w["start_s"], "end_s": round(w["start_s"] + 3 * w["dt"], 4)},
            "clip": {"video": w["video"], "fps": fps, "num_frames": w["num_frames"],
                     "duration": w["duration"], "category": w["category"]},
            "motion": {**(m := w["window_scores"][f"{w['dt']}_{w['k']}"]),
                       "stalled": min(m["change_adjacent"]) < STATIC_THR},
            "source_meta": w.get("source_meta", {})}


def check(rows, src):
    errs = []
    for r in rows:
        idx, ts, fps = r["frame_idx"], r["timestamps"], r["clip"]["fps"]
        if len(idx) != 4 or idx != sorted(set(idx)):
            errs.append(f"{r['id']}: frame_idx {idx}")
        # each index is rounded (±0.5 frame) and a window ending exactly at T uses the last
        # frame (-1 frame), so real spacing is within 2 frames of Δt; `timestamps` are exact
        if any(abs((b - a) - r["gap_s"]) > 2.0 / fps + 1e-6 for a, b in zip(ts, ts[1:])):
            errs.append(f"{r['id']}: timestamps {ts} vs gap {r['gap_s']}")
        if not all(os.path.exists(p) for p in r["frames"]):
            errs.append(f"{r['id']}: missing frame file")
        if max(r["motion"]["change_vs_f0"]) < STATIC_THR:
            errs.append(f"{r['id']}: static window")
        if not r["caption"]:
            errs.append(f"{r['id']}: empty caption")
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        errs.append("duplicate window ids")
    split_of = collections.defaultdict(set)
    for r in rows:
        split_of[r["clip_id"]].add(r["split"])
    leak = [c for c, s in split_of.items() if len(s) > 1]
    if leak:
        errs.append(f"{len(leak)} clips in both train and eval")
    if errs:
        raise SystemExit(f"[finalize] {src}: {len(errs)} problems, e.g. {errs[:5]}")


def finalize(sources, run):
    os.makedirs(run.pools, exist_ok=True)
    sum_path = os.path.join(run.pools, "summary.json")
    summary = json.load(open(sum_path)) if os.path.exists(sum_path) else {"sources": {}}
    summary.update(run.config())
    for src in sources:
        wins = read_jsonl(os.path.join(run.work, f"windows_{src}.jsonl"))
        status = collections.Counter(r["status"] for r in
                                     read_jsonl(os.path.join(run.work, f"windows_{src}.jsonl.clips.jsonl")))
        rows = sorted((final_row(w) for w in wins), key=lambda r: r["id"])
        check(rows, src)
        s = {"extract_status": dict(status), "train_quota": run.quota(src)}
        for split in ("train", "eval"):
            part = [r for r in rows if r["split"] == split]
            write_jsonl(os.path.join(run.pools, f"{src}_{split}.jsonl"), part)
            s[split] = {"windows": len(part), "clips": len({r["clip_id"] for r in part}),
                        "gap_s": {str(k): v for k, v in
                                  sorted(collections.Counter(r["gap_s"] for r in part).items())},
                        "stalled": sum(r["motion"]["stalled"] for r in part),
                        "distinct_captions": len({r["caption"] for r in part})}
        if s["train"]["windows"] < run.quota(src):
            print(f"[finalize] WARNING {src}: {s['train']['windows']} < quota {run.quota(src)}", flush=True)
        summary["sources"][src] = s
        print(f"[finalize] {run.name}/{src}: train {s['train']['windows']} ({s['train']['clips']} clips) "
              f"eval {s['eval']['windows']}  gap {s['train']['gap_s']}  status {dict(status)}", flush=True)
    tot = collections.Counter()
    for s in summary["sources"].values():
        tot.update(s["train"]["gap_s"])
    summary["train_windows"] = sum(s["train"]["windows"] for s in summary["sources"].values())
    summary["eval_windows"] = sum(s["eval"]["windows"] for s in summary["sources"].values())
    summary["train_gap_s"] = dict(tot)
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[finalize] {run.name}: total train {summary['train_windows']}, eval {summary['eval_windows']}, "
          f"gap {dict(tot)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["probe", "score", "plan", "extract", "finalize", "all"], required=True)
    ap.add_argument("--name", default="main", help="run name: pools/<name>/, <work>/runs/<name>/")
    ap.add_argument("--sources", default=",".join(SHARES))
    ap.add_argument("--total", type=int, default=TOTAL, help="train windows, split by SHARES")
    ap.add_argument("--quota", action="append", default=[], metavar="SRC=N",
                    help="override one source's train quota (repeatable)")
    ap.add_argument("--eval-clips", type=int, default=EVAL_CLIPS)
    ap.add_argument("--eval-from", help="reuse this run's eval clips + windows (keeps eval identical)")
    ap.add_argument("--exclude-from", action="append", default=[],
                    help="never reuse this run's train windows (repeatable)")
    ap.add_argument("--procs", type=int, default=32)
    a = ap.parse_args()
    sources = a.sources.split(",")
    assert set(sources) <= set(SHARES), sources
    quotas = {}
    for q in a.quota:
        s, n = q.split("=")
        assert s in SHARES, s
        quotas[s] = int(n)
    run = Run(a.name, quotas, a.total, a.eval_clips, a.eval_from, a.exclude_from)
    os.makedirs(run.work, exist_ok=True)
    stages = ["probe", "score", "plan", "extract", "finalize"] if a.stage == "all" else [a.stage]
    for st in stages:
        if st == "finalize":
            finalize(sources, run)
            continue
        for src in sources:
            if st == "probe":
                probe(src, run, a.procs)
            elif st == "score":
                score(src, run, a.procs)
            elif st == "plan":
                plan(src, run)
            else:
                extract(src, run, a.procs)


if __name__ == "__main__":
    main()
