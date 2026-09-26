#!/usr/bin/env python3
"""Derive the T0-T4 training / eval jsonl (+ trainer meta) from a window pool.

Input : datasets/temporal_ssl/pools/<run>/<src>_{train,eval}.jsonl (merge_pools.py)
Output: datasets/temporal_ssl/settings/<run>/T{0..4}_{train,eval}.jsonl + summary.json
        meta/<run>/T{0..4}_{train,eval}_meta.json (InternVL-U trainer meta, task_type imgen)

Every setting uses every pool window once (same windows, captions and split across T0-T4).
Prompts and answer grammar: common/prompts.py. Per window:

  T0   F0 F1 F2 + GAP -> F3
  T1   F0 F1 F2 -> {"gap"} + F3
  T2   drop Fk (k ∈ 1..3), show the rest in order + GAP + MISSING -> Fk
  T3   F0 F1 F2 shuffled as A B C + GAP -> {"order"} + F3
  T4   50% T4-A (= T0) / 50% T4-B: drop Fk, shuffle the rest as A B C, no GAP / MISSING
       -> {"order", "gap", "missing"} + Fk.  T4_eval holds T4-B rows only (T4-A eval = T0 eval).

Assignments (seeded, SEED = 42): within each (split, source), windows are ordered by a seeded
shuffle and the i-th window gets choice i mod n, so every choice is equally frequent per source:
permutation of A B C (6), missing k (3), T4 variant (2). Each factor uses its own shuffle, so the
factors are independent. T4-B draws its own permutation and k.

cond_image (VAE pixel condition) = the shown frame nearest in time to the target; on a tie
(missing F1 or F2) the earlier frame.

Stalled windows are kept (same windows in every setting). For eval, `answer.order_equiv` lists
label pairs whose frames are adjacent in time and nearly identical (change < STATIC_THR), so a
scorer can accept either order for them.

Usage:
    python derive_settings.py [--run main] [--check-files]
"""
import argparse, collections, glob, hashlib, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.normpath(os.path.join(HERE, "../.."))
sys.path.insert(0, os.path.join(V2, "common"))
from prompts import FRAMES, LABELS, gpt_response, human_prompt  # noqa: E402
from window_motion import STATIC_THR  # noqa: E402

SEED = 42
SETTINGS = ("T0", "T1", "T2", "T3", "T4")
PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
MAX_DYNAMIC_PATCH = 3
POOLS = os.path.join(V2, "datasets", "temporal_ssl", "pools")
SETS = os.path.join(V2, "datasets", "temporal_ssl", "settings")
META = os.path.join(V2, "meta")


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def cycle_assign(windows, n, tag):
    """{window id: choice in range(n)}, each choice equally often, order from a seeded shuffle."""
    ids = sorted(w["id"] for w in windows)
    random.Random(f"{SEED}:{tag}").shuffle(ids)
    return {wid: i % n for i, wid in enumerate(ids)}


def nearest_shown(target, shown):
    """Shown frame index nearest to target; tie -> the earlier one."""
    return min(shown, key=lambda i: (abs(i - target), i))


def order_equiv(w, label_of):
    """Label pairs of shown frames that are adjacent in time and nearly identical."""
    adj = w["motion"]["change_adjacent"]
    return [sorted([label_of[i], label_of[i + 1]]) for i in range(3)
            if adj[i] < STATIC_THR and i in label_of and i + 1 in label_of]


def base_row(w, setting, variant=None):
    tag = setting if variant is None else variant
    return {"id": f"{tag}__{w['id']}", "setting": setting, "variant": variant or setting,
            "task_type": "imgen", "source": w["source"], "window_id": w["id"], "clip_id": w["clip_id"],
            "split": w["split"], "caption": w["caption"], "gap_s": w["gap_s"],
            "stalled": w["motion"]["stalled"]}


def ordered_row(w, setting, kind, variant=None):
    """T0 / T1 / T4-A: F0 F1 F2 in order -> F3."""
    f = w["frames"]
    r = base_row(w, setting, variant)
    ans = {"gap": w["gap_s"]} if kind == "T1" else None
    r.update(image=f[:3], target_image=f[3], cond_image=f[2],
             layout={"shown": ["F0", "F1", "F2"], "target": "F3", "cond": "F2"},
             answer=ans or {},
             conversations=[{"from": "human", "value": human_prompt(kind, w["caption"], ["F0", "F1", "F2"], gap=w["gap_s"])},
                            {"from": "gpt", "value": gpt_response(ans)}])
    return r


def missing_row(w, k):
    """T2: drop Fk, others in order, GAP + MISSING given."""
    f = w["frames"]
    shown = [i for i in range(4) if i != k]
    c = nearest_shown(k, shown)
    r = base_row(w, "T2")
    r.update(image=[f[i] for i in shown], target_image=f[k], cond_image=f[c],
             layout={"shown": [FRAMES[i] for i in shown], "target": FRAMES[k], "cond": FRAMES[c]},
             answer={},
             conversations=[{"from": "human", "value": human_prompt("T2", w["caption"], [FRAMES[i] for i in shown],
                                                                   gap=w["gap_s"], missing=FRAMES[k])},
                            {"from": "gpt", "value": gpt_response(None)}])
    return r


def shuffled_row(w, setting, kind, shown, perm, variant=None, k=None):
    """T3 (shown = F0 F1 F2, target F3) / T4-B (shown = F0..F3 minus Fk, target Fk).
    perm[j] = position in `shown` of the frame labeled LABELS[j]."""
    f = w["frames"]
    frame_of = {LABELS[j]: shown[perm[j]] for j in range(3)}      # label -> frame index
    label_of = {i: l for l, i in frame_of.items()}
    target = 3 if k is None else k
    c = nearest_shown(target, shown)
    order = [label_of[i] for i in sorted(shown)]
    ans = {"order": order} if kind == "T3" else {"order": order, "gap": w["gap_s"], "missing": FRAMES[k]}
    r = base_row(w, setting, variant)
    r.update(image=[f[frame_of[l]] for l in LABELS], target_image=f[target], cond_image=f[c],
             layout={"labels": {l: FRAMES[i] for l, i in frame_of.items()}, "target": FRAMES[target],
                     "cond": FRAMES[c]},
             answer={**ans, "order_equiv": order_equiv(w, label_of)},
             conversations=[{"from": "human", "value": human_prompt(kind, w["caption"], list(LABELS),
                                                                   gap=w["gap_s"] if kind == "T3" else None)},
                            {"from": "gpt", "value": gpt_response(ans)}])
    return r


def derive(windows, split, source):
    t = f"{split}:{source}"
    perm3 = cycle_assign(windows, 6, f"T3perm:{t}")
    k2 = cycle_assign(windows, 3, f"T2k:{t}")
    var4 = cycle_assign(windows, 2, f"T4var:{t}")
    b = [w for w in windows if var4[w["id"]] == 1]
    perm4, k4 = cycle_assign(b, 6, f"T4perm:{t}"), cycle_assign(b, 3, f"T4k:{t}")
    out = collections.defaultdict(list)
    for w in windows:
        out["T0"].append(ordered_row(w, "T0", "T0"))
        out["T1"].append(ordered_row(w, "T1", "T1"))
        out["T2"].append(missing_row(w, 1 + k2[w["id"]]))
        out["T3"].append(shuffled_row(w, "T3", "T3", [0, 1, 2], PERMS[perm3[w["id"]]]))
        if var4[w["id"]] == 0:
            if split == "train":
                out["T4"].append(ordered_row(w, "T4", "T0", variant="T4A"))
        else:
            k = 1 + k4[w["id"]]
            out["T4"].append(shuffled_row(w, "T4", "T4B", [i for i in range(4) if i != k],
                                          PERMS[perm4[w["id"]]], variant="T4B", k=k))
    if split == "eval":  # T4 eval = T4-B for every eval window (T4-A eval is the T0 eval)
        out["T4"] = []
        pe, ke = cycle_assign(windows, 6, f"T4perm:{t}:all"), cycle_assign(windows, 3, f"T4k:{t}:all")
        for w in windows:
            k = 1 + ke[w["id"]]
            out["T4"].append(shuffled_row(w, "T4", "T4B", [i for i in range(4) if i != k],
                                          PERMS[pe[w["id"]]], variant="T4B", k=k))
    return out


def validate(rows, check_files):
    errs = []
    for r in rows:
        convs = r["conversations"]
        if convs[0]["value"].count("<image>") != len(r["image"]) or len(r["image"]) != 3:
            errs.append(f"{r['id']}: image count")
        if not convs[1]["value"].endswith("<img>") or convs[1]["value"].count("<img>") != 1:
            errs.append(f"{r['id']}: gpt must end with one <img>")
        if r["target_image"] in r["image"]:
            errs.append(f"{r['id']}: target among inputs")
        if r["cond_image"] not in r["image"]:
            errs.append(f"{r['id']}: cond_image not an input")
        if check_files and not all(os.path.exists(p) for p in r["image"] + [r["target_image"]]):
            errs.append(f"{r['id']}: missing file")
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        errs.append("duplicate ids")
    return errs


def stats(rows):
    s = {"rows": len(rows), "by_source": dict(collections.Counter(r["source"] for r in rows)),
         "gap_s": {str(k): v for k, v in sorted(collections.Counter(r["gap_s"] for r in rows).items())},
         "variant": dict(collections.Counter(r["variant"] for r in rows)),
         "target": dict(sorted(collections.Counter(r["layout"]["target"] for r in rows).items())),
         "stalled": sum(r["stalled"] for r in rows)}
    orders = [" ".join(r["answer"]["order"]) for r in rows if "order" in r["answer"]]
    if orders:
        s["order"] = dict(sorted(collections.Counter(orders).items()))
        s["order_equiv_rows"] = sum(bool(r["answer"]["order_equiv"]) for r in rows if "order" in r["answer"])
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--check-files", action="store_true", help="stat every frame path (slow)")
    a = ap.parse_args()
    pool_dir, out_dir, meta_dir = (os.path.join(POOLS, a.run), os.path.join(SETS, a.run),
                                   os.path.join(META, a.run))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    summary = {"run": a.run, "seed": SEED, "pool": os.path.relpath(pool_dir, V2), "settings": {}}
    for split in ("train", "eval"):
        acc = collections.defaultdict(list)
        for p in sorted(glob.glob(os.path.join(pool_dir, f"*_{split}.jsonl"))):
            src = os.path.basename(p)[: -len(f"_{split}.jsonl")]
            for st, rows in derive(read_jsonl(p), split, src).items():
                acc[st] += rows
        for st in SETTINGS:
            rows = sorted(acc[st], key=lambda r: r["id"])
            errs = validate(rows, a.check_files)
            if errs:
                raise SystemExit(f"{st}_{split}: {len(errs)} problems, e.g. {errs[:5]}")
            path = os.path.join(out_dir, f"{st}_{split}.jsonl")
            with open(path + ".tmp", "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            os.replace(path + ".tmp", path)
            name = f"v2_{a.run}_{st}_{split}"
            with open(os.path.join(meta_dir, f"{st}_{split}_meta.json"), "w") as f:
                json.dump({name: {"root": "/", "annotation": path, "data_augment": False,
                                  "max_dynamic_patch": MAX_DYNAMIC_PATCH, "repeat_time": 1,
                                  "length": len(rows), "task_type": "imgen"}}, f, indent=2)
            with open(path, "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()[:16]
            summary["settings"][f"{st}_{split}"] = {**stats(rows), "sha256_16": sha}
            print(f"[derive] {a.run} {st}_{split}: {len(rows)} rows", flush=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
