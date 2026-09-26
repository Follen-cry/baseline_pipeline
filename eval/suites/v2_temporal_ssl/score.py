#!/usr/bin/env python3
"""Rule-based scorer for the v2 T0-T4 eval (full eval split or the evalmini validation subset).

Inputs
  --eval   settings/<run>/T{x}_eval[mini].jsonl (derive_settings.py / make_eval_mini.py)
  --pred   predictions jsonl, one line per eval row:
             {"id": <eval row id>, "response": <model text up to and incl. <img>>, "image": <png path>}
           A row with no prediction, or an unreadable image, counts as a failure (image metrics
           missing, text parse error "no_pred").
  --oracle gt | copy   build the predictions instead (sanity checks): gt = target image + the
           training answer; copy = cond_image + a bare "<img>" (the copy-last-frame baseline).

Text (settings with a JSON answer line; grammar = data/v2/common/prompts.py parse_answer)
  parse_ok      the line parses with exactly the expected keys and values in the grammar
  gap_acc       gap == truth                                                   (T1, T4-B)
  order_exact   order == truth, allowing swaps of answer.order_equiv pairs     (T3, T4-B)
  order_pair    fraction of the 3 label pairs in the right relative order     (T3, T4-B)
  missing_acc   missing == truth                                               (T4-B)
  all_correct   order_exact and gap and missing                                (T4-B)
  An unparsable answer scores 0 on every text metric (parse failures are not dropped).

Image (prediction, target and cond_image all resized to S x S with bicubic, S = --size, like training's
_make_vae_target_transform, which squashes every target to gen_image_size^2)
  psnr, ssim, mae           prediction vs target
  *_copy                    cond_image vs target: the copy-nearest-observed-frame baseline
  dpsnr, dssim              prediction minus copy baseline (> 0 = better than copying)
  beat_copy                 1 if MSE(pred, target) < MSE(cond, target)
  motion_psnr, dmotion_psnr PSNR on the pixels where target differs from cond_image by more than
                            --motion-thr (max over RGB, [0, 1] scale), and its gain over copy;
                            rows whose moving area is < --min-motion-frac of the image get none.

Output (--out DIR): per_row.jsonl, summary.json (flat {"<group>/<metric>": value} dict ready for
wandb.log, plus "<group>/n"), and a printed table. Groups: all, src=<source>, gap=<g>,
target=<Fk>, moving (non-stalled rows).

Usage:
    python score.py --eval .../T3_evalmini.jsonl --pred preds.jsonl --out results/T3_step500
    python score.py --eval .../T3_evalmini.jsonl --oracle copy --out results/T3_copy
"""
import argparse, collections, itertools, json, math, os, sys
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "../../../data/v2/common")))
from prompts import GAPS, IMG, parse_answer  # noqa: E402

TEXT_KEYS = {"T1": ("gap",), "T3": ("order",), "T4B": ("order", "gap", "missing")}
PSNR_CAP = 60.0
IMG_METRICS = ("psnr", "ssim", "mae", "psnr_copy", "ssim_copy", "dpsnr", "dssim", "beat_copy",
               "motion_psnr", "motion_psnr_copy", "dmotion_psnr", "motion_frac")
TEXT_METRICS = ("parse_ok", "gap_acc", "order_exact", "order_pair", "missing_acc", "all_correct")


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------- text
def accepted_orders(order, equiv):
    """Truth order plus every order reachable by swapping labels of near-identical adjacent frames."""
    seen, todo = {tuple(order)}, [tuple(order)]
    while todo:
        cur = todo.pop()
        for a, b in equiv:
            if a in cur and b in cur:
                l = list(cur)
                i, j = l.index(a), l.index(b)
                l[i], l[j] = l[j], l[i]
                if tuple(l) not in seen:
                    seen.add(tuple(l))
                    todo.append(tuple(l))
    return seen


def pair_acc(pred, order, equiv):
    eq = {frozenset(p) for p in equiv}
    rank_t, rank_p = {l: i for i, l in enumerate(order)}, {l: i for i, l in enumerate(pred)}
    ok = [frozenset((a, b)) in eq or (rank_t[a] < rank_t[b]) == (rank_p[a] < rank_p[b])
          for a, b in itertools.combinations(order, 2)]
    return sum(ok) / len(ok)


def score_text(row, response):
    keys = TEXT_KEYS.get(row["variant"])
    if not keys:
        return {}
    ans = row["answer"]
    d, err = (None, "no_pred") if response is None else parse_answer(response, keys)
    out = {"parse_ok": float(d is not None), "parse_err": err, "pred_answer": d}
    if "gap" in keys:
        out["gap_acc"] = float(d is not None and d["gap"] == ans["gap"])
    if "order" in keys:
        ok = d is not None
        out["order_exact"] = float(ok and tuple(d["order"]) in accepted_orders(ans["order"], ans["order_equiv"]))
        out["order_pair"] = pair_acc(d["order"], ans["order"], ans["order_equiv"]) if ok else 0.0
    if "missing" in keys:
        out["missing_acc"] = float(d is not None and d["missing"] == ans["missing"])
    if row["variant"] == "T4B":
        out["all_correct"] = float(out["order_exact"] and out["gap_acc"] and out["missing_acc"])
    return out


# ---------------------------------------------------------------- image
def load(path, size):
    im = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)  # = training target transform
    return np.asarray(im, dtype=np.float64) / 255.0


def psnr_from_mse(mse):
    return PSNR_CAP if mse <= 10 ** (-PSNR_CAP / 10) else min(PSNR_CAP, -10 * math.log10(mse))


def ssim(a, b):
    """Grayscale SSIM (Wang et al. 2004: 11x11 Gaussian, sigma 1.5, K1 .01, K2 .03, L = 1)."""
    w = np.array([0.299, 0.587, 0.114])
    x, y = a @ w, b @ w
    blur = lambda z: cv2.GaussianBlur(z, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT)
    mx, my = blur(x), blur(y)
    sxx, syy, sxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return float(m[5:-5, 5:-5].mean())


def score_image(args):
    row, pred_path, size, motion_thr, min_motion_frac = args
    if not pred_path or not os.path.exists(pred_path):
        return {"img_missing": 1.0}
    try:
        p = load(pred_path, size)
    except Exception:  # noqa: BLE001
        return {"img_missing": 1.0}
    t, c = load(row["target_image"], size), load(row["cond_image"], size)
    mse, mse_c = float(((p - t) ** 2).mean()), float(((c - t) ** 2).mean())
    o = {"img_missing": 0.0, "psnr": psnr_from_mse(mse), "psnr_copy": psnr_from_mse(mse_c),
         "ssim": ssim(p, t), "ssim_copy": ssim(c, t), "mae": float(np.abs(p - t).mean()),
         "beat_copy": float(mse < mse_c)}
    o["dpsnr"], o["dssim"] = o["psnr"] - o["psnr_copy"], o["ssim"] - o["ssim_copy"]
    mask = np.abs(t - c).max(axis=2) > motion_thr
    o["motion_frac"] = float(mask.mean())
    if o["motion_frac"] >= min_motion_frac:
        o["motion_psnr"] = psnr_from_mse(float(((p - t) ** 2)[mask].mean()))
        o["motion_psnr_copy"] = psnr_from_mse(float(((c - t) ** 2)[mask].mean()))
        o["dmotion_psnr"] = o["motion_psnr"] - o["motion_psnr_copy"]
    return o


# ---------------------------------------------------------------- aggregate
def groups(row):
    g = ["all", f"src={row['source']}", f"gap={row['gap_s']}", f"target={row['layout']['target']}"]
    return g + ([] if row["stalled"] else ["moving"])


def summarize(scored):
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in scored:
        for g in groups(r):
            acc[g]["n"].append(1)
            for k in IMG_METRICS + TEXT_METRICS + ("img_missing",):
                if r.get(k) is not None:
                    acc[g][k].append(r[k])
    out = {}
    for g in sorted(acc, key=lambda g: (g != "all", g)):
        out[f"{g}/n"] = len(acc[g]["n"])
        for k, v in acc[g].items():
            if k != "n" and v:
                out[f"{g}/{k}"] = float(np.mean(v))
    return out


def print_table(summary):
    cols = ["n", "psnr", "psnr_copy", "dpsnr", "ssim", "dssim", "beat_copy", "dmotion_psnr",
            "parse_ok", "gap_acc", "order_exact", "missing_acc", "all_correct"]
    cols = [c for c in cols if any(k.endswith("/" + c) for k in summary)]
    names = sorted({k.rsplit("/", 1)[0] for k in summary}, key=lambda g: (g != "all", g))
    print(f"{'group':<22}" + "".join(f"{c:>13}" for c in cols))
    for g in names:
        cells = []
        for c in cols:
            v = summary.get(f"{g}/{c}")
            cells.append(f"{'-':>13}" if v is None else f"{v:>13d}" if c == "n" else f"{v:>13.3f}")
        print(f"{g:<22}" + "".join(cells))


def oracle_preds(rows, kind):
    if kind == "gt":
        return {r["id"]: {"response": r["conversations"][1]["value"], "image": r["target_image"]} for r in rows}
    return {r["id"]: {"response": IMG, "image": r["cond_image"]} for r in rows}


def score(rows, preds, size=512, motion_thr=0.1, min_motion_frac=0.002, workers=8):
    jobs = [(r, (preds.get(r["id"]) or {}).get("image"), size, motion_thr, min_motion_frac) for r in rows]
    if workers <= 1:  # in-process (e.g. inside a training job, where forking is unwelcome)
        img = [score_image(j) for j in jobs]
    else:
        with Pool(min(workers, max(1, len(jobs)))) as pool:
            img = pool.map(score_image, jobs, chunksize=4)
    scored = []
    for r, im in zip(rows, img):
        p = preds.get(r["id"])
        s = {"id": r["id"], "variant": r["variant"], "source": r["source"], "gap_s": r["gap_s"],
             "layout": r["layout"], "stalled": r["stalled"], "has_pred": p is not None,
             "response": None if p is None else p.get("response")}
        s.update(score_text(r, s["response"]))
        s.update(im)
        scored.append(s)
    return scored, summarize(scored)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pred")
    src.add_argument("--oracle", choices=["gt", "copy"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=512, help="compare at SIZE x SIZE (= gen_image_size)")
    ap.add_argument("--motion-thr", type=float, default=0.1)
    ap.add_argument("--min-motion-frac", type=float, default=0.002)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    rows = read_jsonl(a.eval)
    if a.oracle:
        preds = oracle_preds(rows, a.oracle)
    else:
        preds = {p["id"]: p for p in read_jsonl(a.pred)}
        unknown = set(preds) - {r["id"] for r in rows}
        if unknown:
            print(f"[score] warning: {len(unknown)} predictions for ids not in --eval", file=sys.stderr)
    scored, summary = score(rows, preds, a.size, a.motion_thr, a.min_motion_frac, a.workers)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "per_row.jsonl"), "w") as f:
        for s in scored:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    meta = {"eval": os.path.abspath(a.eval), "pred": a.pred and os.path.abspath(a.pred), "oracle": a.oracle,
            "size": a.size, "motion_thr": a.motion_thr, "min_motion_frac": a.min_motion_frac,
            "chance": {"gap_acc_majority": max(collections.Counter(r["gap_s"] for r in rows).values()) / len(rows),
                       "order_exact": 1 / 6, "missing_acc": 1 / 3, "gaps": list(GAPS)}}
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump({"meta": meta, "metrics": summary}, f, indent=1)
    print_table(summary)


if __name__ == "__main__":
    main()
