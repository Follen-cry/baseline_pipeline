#!/usr/bin/env python3
"""Index the VBVR raw videos for v2 and mark which ones are reserved for the existing VBVR evals.

Raw videos: /scratch/network/ssd/junlin/vbvr_next_frame/raw/<task>/<chunk>/<prefix>_task/<task_id>/
{ground_truth.mp4, first_frame.png, final_frame.png, metadata.json, prompt.txt} -- the pool
v1's S0-S3 was built from (14 tasks). Sample names restart per chunk, so a sample is identified
by its path. Nothing is copied; the manifest points at the files in place.

(The target_pred pool /scratch/network/ssd/junlin/vbvr_target_pred/ has first/final frames only,
no ground_truth.mp4, so it cannot feed video windows; its eval rows are only used for leak checks.)

split = eval_reserved (never used for v2 training) if s3_eval or tp_eval_md5; duplicate if only
dup; train otherwise. Flags:
  s3_eval      : it is the source_video of a v1 S3 eval-holdout window
                 (data/v1/datasets/final_s0s3/S3_eval.jsonl -> samples/<task>/<n>/window_meta.json)
  tp_eval_md5  : its first_frame.png is byte-identical to the input image of a target_pred eval row
                 (vbvr_target_pred_id/{target_pred_id_eval,target_pred_id_eval_sotm}.jsonl,
                  vbvr_target_pred_ood/target_pred_ood_eval.jsonl) -- the eval v2 will reuse
  dup          : its ground_truth.mp4 is byte-identical to an earlier sample's (keep the first only).
                 Only samples sharing a first_frame.png are hashed; a shared first frame alone is
                 NOT a duplicate (e.g. most mirror_reflection samples share a start state but
                 differ in the video).

Output (under /scratch/network/ssd/junlin/raw/vbvr/):
  manifest_all.jsonl : every sample with split = train | eval_reserved | duplicate, flags, bytes, prompt
  summary.json       : per-task counts and sizes

Usage:
    python build_vbvr_manifest.py [--workers 32]
"""
import argparse, hashlib, json, os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

RAW = "/scratch/network/ssd/junlin/vbvr_next_frame/raw"
OUT = "/scratch/network/ssd/junlin/raw/vbvr"
S3_EVAL = "/scratch/network/ssd2/junlin/ssl_mllm/baseline_pipeline/data/v1/datasets/final_s0s3/S3_eval.jsonl"
TP = "/scratch/network/ssd2/junlin/ssl_mllm/data/datasets"
TP_EVAL = [f"{TP}/vbvr_target_pred_id/target_pred_id_eval.jsonl",
           f"{TP}/vbvr_target_pred_id/target_pred_id_eval_sotm.jsonl",
           f"{TP}/vbvr_target_pred_ood/target_pred_ood_eval.jsonl"]
# tasks in the locked 10-task VBVR eval set (eval/suites/vbvr/README.md)
LOCKED_EVAL_TASKS = {"shape_color_then_move", "ball_bounces_given_time", "glass_refraction", "stable_sort",
                     "multi_object_placement", "animal_size_sorting", "maze", "grid_shift",
                     "rotation_puzzle", "2d_geometric_transformation"}


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def list_samples():
    out = []
    for task in sorted(os.listdir(RAW)):
        for chunk in sorted(os.listdir(f"{RAW}/{task}")):
            cdir = f"{RAW}/{task}/{chunk}"
            for sub in sorted(os.listdir(cdir)):
                if not sub.endswith("_task"):
                    continue
                for tid in sorted(os.listdir(f"{cdir}/{sub}")):
                    d = f"{cdir}/{sub}/{tid}"
                    if os.path.exists(f"{d}/ground_truth.mp4"):
                        out.append((task, chunk, tid, d))
    return out


def describe(s):
    task, chunk, tid, d = s
    files = os.listdir(d)
    dir_bytes = sum(os.path.getsize(f"{d}/{f}") for f in files)
    prompt = open(f"{d}/prompt.txt").read().strip() if "prompt.txt" in files else None
    return {"id": f"vbvr__{task}__{chunk}__{tid}", "source": "vbvr", "task": task, "chunk": chunk,
            "task_id": tid, "sample_dir": d, "video": f"{d}/ground_truth.mp4",
            "video_bytes": os.path.getsize(f"{d}/ground_truth.mp4"), "dir_bytes": dir_bytes,
            "first_frame_md5": md5(f"{d}/first_frame.png"), "prompt": prompt,
            "in_locked_eval_tasks": task in LOCKED_EVAL_TASKS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    samples = list_samples()
    print(f"[index] {len(samples)} raw samples with ground_truth.mp4", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(describe, samples, chunksize=64))

    s3_src = set()
    for l in open(S3_EVAL):
        r = json.loads(l)
        img0 = r["image"][0]
        if "/vbvr_next_frame/samples/" in img0:
            s3_src.add(json.load(open(os.path.join(os.path.dirname(img0), "window_meta.json")))["source_video"])
    tp_imgs = [json.loads(l)["image"][0] for f in TP_EVAL for l in open(f)]
    with ThreadPoolExecutor(a.workers) as ex:
        tp_md5 = set(ex.map(md5, tp_imgs))
    print(f"[eval] S3 eval source videos: {len(s3_src)}; target_pred eval images: {len(tp_imgs)} "
          f"({len(tp_md5)} unique)", flush=True)

    ff_count = defaultdict(int)
    for r in rows:
        ff_count[r["first_frame_md5"]] += 1
    cand = [r for r in rows if ff_count[r["first_frame_md5"]] > 1]
    with ThreadPoolExecutor(a.workers) as ex:
        for r, h in zip(cand, ex.map(lambda r: md5(r["video"]), cand)):
            r["video_md5"] = h
    print(f"[dup] {len(cand)} samples share a first frame; hashed their mp4s", flush=True)

    seen = set()
    for r in rows:
        flags = []
        if r["video"] in s3_src:
            flags.append("s3_eval")
        if r["first_frame_md5"] in tp_md5:
            flags.append("tp_eval_md5")
        key = r.get("video_md5")
        if key is not None and key in seen:
            flags.append("dup")
        if key is not None:
            seen.add(key)
        r["eval_flags"] = flags
        r["split"] = ("eval_reserved" if {"s3_eval", "tp_eval_md5"} & set(flags)
                      else "duplicate" if flags else "train")
    missing_s3 = s3_src - {r["video"] for r in rows}

    with open(f"{OUT}/manifest_all.jsonl.tmp", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(f"{OUT}/manifest_all.jsonl.tmp", f"{OUT}/manifest_all.jsonl")

    per = defaultdict(lambda: defaultdict(float))
    for r in rows:
        p = per[r["task"]]
        p["samples"] += 1
        p["video_GB"] += r["video_bytes"] / 1e9
        p["dir_GB"] += r["dir_bytes"] / 1e9
        p[r["split"]] += 1
        if r["split"] == "train":
            p["train_video_GB"] += r["video_bytes"] / 1e9
            p["train_dir_GB"] += r["dir_bytes"] / 1e9
        for fl in r["eval_flags"]:
            p["flag_" + fl] += 1
    summary = {"tasks": {t: {k: (round(v, 3) if isinstance(v, float) and not v.is_integer() else int(v))
                             for k, v in p.items()} for t, p in sorted(per.items())},
               "s3_eval_sources_not_found": len(missing_s3)}
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
    tot = defaultdict(float)
    for p in per.values():
        for k, v in p.items():
            tot[k] += v
    print("[summary] total:", {k: round(v, 2) for k, v in tot.items()}, "| S3 sources not found:", len(missing_s3))


if __name__ == "__main__":
    main()
