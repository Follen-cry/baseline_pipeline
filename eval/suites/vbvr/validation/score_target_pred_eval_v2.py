#!/usr/bin/env python3
"""Score an InternVL-U inference run with the REWRITTEN rule-based scorers
(Evaluation/VBVR-CustomEval/scorers/), writing `scored_new.json` alongside the
existing `scored.json` so before/after comparison stays possible.

Same output schema as validation/score_target_pred_eval.py, plus a `sub` field
per record carrying the individual sub-criterion scores (the rewritten scorers
expose these directly, and the comparison artifact uses them).

Covers only the four tasks that have been rewritten so far; any other task in
the run is skipped and reported in `errors`.

    python score_target_pred_eval_v2.py --run-dir <run>/s0_4task500sft
    python score_target_pred_eval_v2.py --all          # every 4task500sft run
"""
import argparse
import importlib
import json
import os
import sys

SCORERS = "/scratch/network/ssd2/junlin/ssl_mllm/Evaluation/VBVR-CustomEval/scorers"
sys.path.insert(0, SCORERS)

RUN_ROOT = "/scratch/network/ssd2/junlin/ssl_mllm/results/vbvr_target_pred_eval"
ID_JSONL = "/scratch/network/ssd2/junlin/ssl_mllm/data/datasets/vbvr_target_pred_id/target_pred_id_eval.jsonl"
OOD_JSONL = "/scratch/network/ssd2/junlin/ssl_mllm/data/datasets/vbvr_target_pred_ood/target_pred_ood_eval.jsonl"
DEFAULT_RUNS = ["base_4task500sft", "s0_4task500sft", "s1_4task500sft",
                "s2_4task500sft", "s3_4task500sft"]

TASK_MODULES = {
    "multi_object_placement": "multi_object_placement",
    "rotation_puzzle": "rotation_puzzle",
    "shape_color_then_move": "shape_color_then_move",
    "2d_geometric_transformation": "2d_geometric_transformation",
}
_LOADED = {}


def scorer_for(task_name):
    if task_name not in TASK_MODULES:
        return None
    if task_name not in _LOADED:
        _LOADED[task_name] = importlib.import_module(TASK_MODULES[task_name])
    return _LOADED[task_name]


def load_source_rows():
    rows = [json.loads(l) for l in open(ID_JSONL)]
    rows += [json.loads(l) for l in open(OOD_JSONL)]
    return {r["id"]: r for r in rows}


def score_run(run_dir, out_name="scored_new.json"):
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    source_by_id = load_source_rows()
    per_task, all_records, errors = {}, [], []

    for row in meta["results"]:
        task_name = row["task_name"]
        mod = scorer_for(task_name)
        if mod is None:
            errors.append(f"{row['id']}: no rewritten scorer for task '{task_name}' (skipped)")
            continue
        # The input frame is a ground-truth-side reference (where the scene
        # started: the source pose for 2d_geometric_transformation's required
        # displacement, the unmodified object for multi_object_placement's
        # fidelity check), so it must be the canonical eval source, not
        # meta.json's `input_image` -- that points at the prescaled 512px copy
        # fed to the model, and resampling it twice shifts scores by ~0.005.
        # The rewritten scorers were calibrated against the canonical source.
        src = source_by_id.get(row["id"])
        input_path = src["image"][0] if src else row["input_image"]

        sub, score, err = None, None, None
        try:
            sub = mod.score(input_path, row["generated_image"], row["target_image"])
            score = mod.overall(sub)
        except Exception as e:                      # noqa: BLE001 - recorded, not raised
            err = str(e)
        rec = {"id": row["id"], "task_name": task_name, "category": row["category"],
               "domain": row["domain"], "score": score, "error": err,
               "sub": {k: round(v, 4) for k, v in sub.items()} if sub else None}
        all_records.append(rec)
        if err:
            errors.append(f"{row['id']}: {err}")
        per_task.setdefault(task_name, []).append(rec)

    summary = {}
    for task_name, recs in per_task.items():
        scores = [r["score"] for r in recs if r["score"] is not None]
        summary[task_name] = {
            "category": recs[0]["category"], "domain": recs[0]["domain"],
            "n": len(recs), "n_scored": len(scores),
            "mean_score": (sum(scores) / len(scores)) if scores else None,
            "min_score": min(scores) if scores else None,
            "max_score": max(scores) if scores else None,
        }
    overall_scores = [r["score"] for r in all_records if r["score"] is not None]
    overall = {"n": len(all_records), "n_scored": len(overall_scores),
               "mean_score": (sum(overall_scores) / len(overall_scores)) if overall_scores else None}

    out_path = os.path.join(run_dir, out_name)
    with open(out_path, "w") as f:
        json.dump({"run_meta": {k: v for k, v in meta.items() if k != "results"},
                   "scorer": "VBVR-CustomEval/scorers (rewritten rule-based, deterministic CV)",
                   "overall": overall, "per_task": summary,
                   "records": all_records, "errors": errors}, f, indent=2)
    return out_path, overall, summary, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir")
    ap.add_argument("--all", action="store_true", help=f"score all of {DEFAULT_RUNS}")
    ap.add_argument("--out-name", default="scored_new.json")
    args = ap.parse_args()

    runs = ([os.path.join(RUN_ROOT, r) for r in DEFAULT_RUNS] if args.all
            else [args.run_dir])
    if not runs or runs == [None]:
        ap.error("pass --run-dir or --all")

    for run_dir in runs:
        path, overall, summary, errors = score_run(run_dir, args.out_name)
        name = os.path.basename(run_dir)
        means = "  ".join(f"{t.split('_')[0][:9]}={summary[t]['mean_score']:.3f}" for t in sorted(summary))
        print(f"{name:22s} overall={overall['mean_score']:.4f} ({overall['n_scored']}/{overall['n']})  {means}")
        if errors:
            print(f"  {len(errors)} errors, first: {errors[0]}")
    print("\nwrote", args.out_name, "into each run dir")


if __name__ == "__main__":
    main()
