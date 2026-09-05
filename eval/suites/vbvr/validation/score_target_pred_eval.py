#!/usr/bin/env python3
"""Score a real InternVL-U inference run (built by
inference/run_target_pred_eval.py) with the current image-only eval pipeline
(evaluators/image_evaluator.py), task_specific_only=True -- same convention
as validation/test_harness.py's 20-pair sanity checks, just against real
model outputs instead of GT-vs-GT/mismatched-GT pairs.

    python score_target_pred_eval.py \
        --run-dir /scratch/network/ssd2/junlin/ssl_mllm/results/vbvr_target_pred_eval/base_prescale512_vaecond \
        --out scored_target_pred_eval.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, "/scratch/network/ssd2/junlin/ssl_mllm/Evaluation/VBVR-EvalKit")
sys.path.insert(0, "/scratch/network/ssd2/junlin/ssl_mllm/Evaluation/VBVR-CustomEval/evaluators")
from vbvr_bench.evaluators import get_task_category, get_split
from image_evaluator import TASK_IMAGE_EVALUATOR_MAP

ID_JSONL = "/scratch/network/ssd2/junlin/ssl_mllm/data/datasets/vbvr_target_pred_id/target_pred_id_eval.jsonl"
OOD_JSONL = "/scratch/network/ssd2/junlin/ssl_mllm/data/datasets/vbvr_target_pred_ood/target_pred_ood_eval.jsonl"

# short task_name (as used in the target_pred jsonl / result meta) -> full
# generator id (TASK_IMAGE_EVALUATOR_MAP key), derived once so this doesn't
# hardcode a second copy of the name list that could drift from LOCKED_TASKS.
NAME_TO_FULL = {
    full.split("_data-generator")[0].split("_", 1)[1]: full
    for full in TASK_IMAGE_EVALUATOR_MAP
}


def load_source_rows():
    rows = [json.loads(l) for l in open(ID_JSONL)]
    rows += [json.loads(l) for l in open(OOD_JSONL)]
    return {r["id"]: r for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None, help="defaults to <run-dir>/scored.json")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.run_dir, "meta.json")))
    source_by_id = load_source_rows()

    per_task = {}
    all_records = []
    errors = []

    for row in meta["results"]:
        task_name = row["task_name"]
        full = NAME_TO_FULL.get(task_name)
        if full is None:
            errors.append(f"{row['id']}: no evaluator mapped for task_name '{task_name}'")
            continue

        source_row = source_by_id.get(row["id"])
        gt_first_frame = source_row["image"][0] if source_row else row["input_image"]

        evaluator = TASK_IMAGE_EVALUATOR_MAP[full](device="cpu", task_name=full)
        try:
            result = evaluator.evaluate({
                "final_frame_path": row["generated_image"],
                "first_frame_path": row["input_image"],
                "gt_first_frame": gt_first_frame,
                "gt_final_frame": row["target_image"],
            }, task_specific_only=True)
            score = result["score"]
            err = result.get("error")
        except Exception as e:
            score = None
            err = str(e)

        rec = {"id": row["id"], "task_name": task_name, "category": row["category"],
               "domain": row["domain"], "score": score, "error": err}
        all_records.append(rec)
        if err:
            errors.append(f"{row['id']}: {err}")
        per_task.setdefault(task_name, []).append(rec)

    summary = {}
    for task_name, recs in per_task.items():
        scores = [r["score"] for r in recs if r["score"] is not None]
        summary[task_name] = {
            "category": recs[0]["category"],
            "domain": recs[0]["domain"],
            "n": len(recs),
            "n_scored": len(scores),
            "mean_score": (sum(scores) / len(scores)) if scores else None,
            "min_score": min(scores) if scores else None,
            "max_score": max(scores) if scores else None,
        }

    overall_scores = [r["score"] for r in all_records if r["score"] is not None]
    overall = {
        "n": len(all_records),
        "n_scored": len(overall_scores),
        "mean_score": (sum(overall_scores) / len(overall_scores)) if overall_scores else None,
    }

    out_path = args.out or os.path.join(args.run_dir, "scored.json")
    with open(out_path, "w") as f:
        json.dump({
            "run_meta": {k: v for k, v in meta.items() if k != "results"},
            "overall": overall,
            "per_task": summary,
            "records": all_records,
            "errors": errors,
        }, f, indent=2)

    print(json.dumps({"overall": overall, "per_task": summary}, indent=2, default=str))
    print(f"\nTotal errors: {len(errors)}")
    for e in errors[:20]:
        print("  ERR:", e)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
