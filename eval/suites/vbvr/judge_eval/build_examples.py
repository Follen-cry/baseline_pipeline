#!/usr/bin/env python3
"""Run the new pure-judge system (evaluators/llm_judge_full.py) on real
InternVL-U generations from the target_pred_eval run, for 2 examples per
task -- the SAME "Representative" (closest to task mean) and "Worst-scoring"
picks used in inference/target_pred_results.html's gallery, so the judge
score and the rule-based score can be compared side by side on identical
images, not different ones.

    python build_examples.py --out full_judge_examples.json
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "evaluators"))
from llm_judge_full import FullVLMJudge, TASK_CRITERIA  # noqa: E402

RUN_DIR = "/scratch/network/ssd2/junlin/ssl_mllm/results/vbvr_target_pred_eval/base_prescale512_vaecond"
SCORED_PATH = os.path.join(RUN_DIR, "scored.json")
META_PATH = os.path.join(RUN_DIR, "meta.json")


def pick_examples(records):
    scored_recs = [r for r in records if r["score"] is not None]
    scored_recs = sorted(scored_recs, key=lambda r: r["score"])
    mean = sum(r["score"] for r in scored_recs) / len(scored_recs)
    rep = min(scored_recs, key=lambda r: abs(r["score"] - mean))
    worst = scored_recs[0]
    picks = [("Representative (closest to task mean)", rep)]
    if worst["id"] != rep["id"]:
        picks.append(("Worst-scoring", worst))
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "full_judge_examples.json"))
    args = ap.parse_args()

    scored = json.load(open(SCORED_PATH))
    meta = json.load(open(META_PATH))
    by_id = {r["id"]: r for r in meta["results"]}

    records_by_task = {}
    for r in scored["records"]:
        records_by_task.setdefault(r["task_name"], []).append(r)

    judge = FullVLMJudge()
    results = {}
    t0 = time.time()
    total = sum(len(pick_examples(records_by_task[t])) for t in TASK_CRITERIA if t in records_by_task)
    done = 0

    for task_name in TASK_CRITERIA:
        if task_name not in records_by_task:
            print(f"SKIP {task_name}: no scored records (no eval split)", flush=True)
            continue
        picks = pick_examples(records_by_task[task_name])
        task_results = []
        for label, rec in picks:
            row = by_id[rec["id"]]
            t_start = time.time()
            j = judge.score(
                task_name=task_name,
                prompt_text=row["prompt"],
                first_frame_path=row["input_image"],
                gen_final_path=row["generated_image"],
                gt_final_path=row["target_image"],
            )
            elapsed = round(time.time() - t_start, 2)
            done += 1
            print(f"[{done}/{total}] {task_name} / {label}: "
                  f"rule={rec['score']:.3f} judge={j['score']:.3f} ({elapsed}s)", flush=True)
            task_results.append({
                "label": label,
                "id": rec["id"],
                "rule_score": rec["score"],
                "input_image": row["input_image"],
                "generated_image": row["generated_image"],
                "target_image": row["target_image"],
                "prompt": row["prompt"],
                "judge_score": j["score"],
                "judge_criteria": j.get("criteria"),
                "judge_reasoning": j.get("reasoning"),
                "judge_error": j.get("error"),
            })
        results[task_name] = task_results

    with open(args.out, "w") as f:
        json.dump({"results": results, "elapsed_s": round(time.time() - t0, 1)}, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
