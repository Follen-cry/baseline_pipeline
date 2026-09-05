#!/usr/bin/env python3
"""Run the pure-judge system (evaluators/llm_judge_full.py) on ALL 900 real
generations from a target_pred_eval run (not just the 18 hand-picked
examples in build_examples.py) -- so the results artifact can show the
judge's score for the same random samples it displays the rule-based score
for.

Generalized 2026-08-27 to take --run-dir, so the identical judge pipeline
can be pointed at any model's run (bagel, sensenova, ...), not just
InternVL-U's -- all of them share the same meta.json record schema by
construction (each model's run script reads InternVL-U's own meta.json as
the source of truth for prompts/task list/ids, see
Evaluation/vbvr_baseline_runners/run_target_pred_eval_{bagel,sensenova}.py). Default
--run-dir/--out are unchanged from before this generalization, so existing
callers (gen_artifact.py, build_examples.py, gen_target_pred_results_artifact.py)
that hardcode the historical judge_eval/judge_scored_900.json path keep
working without modification.

    python score_all_with_judge.py --out judge_scored_900.json [--workers 8]
    python score_all_with_judge.py --run-dir .../results/vbvr_target_pred_eval/bagel [--workers 8]
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "evaluators"))
from llm_judge_full import FullVLMJudge, TASK_CRITERIA  # noqa: E402

DEFAULT_RUN_DIR = "/scratch/network/ssd2/junlin/ssl_mllm/results/vbvr_target_pred_eval/base_prescale512_vaecond"
DEFAULT_OUT = os.path.join(HERE, "judge_scored_900.json")


def score_one(judge, row):
    j = judge.score(
        task_name=row["task_name"],
        prompt_text=row["prompt"],
        first_frame_path=row["input_image"],
        gen_final_path=row["generated_image"],
        gt_final_path=row["target_image"],
    )
    return {
        "id": row["id"],
        "task_name": row["task_name"],
        "judge_score": j["score"],
        "judge_criteria": j.get("criteria"),
        "judge_reasoning": j.get("reasoning"),
        "judge_error": j.get("error"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=DEFAULT_RUN_DIR,
                     help="target_pred_eval run dir containing meta.json (default: InternVL-U's)")
    ap.add_argument("--out", default=None,
                     help="defaults to <run-dir>/judge_scored.json, except for the default "
                          "--run-dir, which keeps the historical judge_eval/judge_scored_900.json "
                          "path other scripts hardcode")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out_path = args.out or (DEFAULT_OUT if args.run_dir == DEFAULT_RUN_DIR
                             else os.path.join(args.run_dir, "judge_scored.json"))

    meta = json.load(open(os.path.join(args.run_dir, "meta.json")))
    rows = [r for r in meta["results"] if r["task_name"] in TASK_CRITERIA]
    print(f"Scoring {len(rows)} rows across {len(TASK_CRITERIA)} tasks with {args.workers} workers", flush=True)

    judge = FullVLMJudge()
    results = {}
    t0 = time.time()
    errors = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(score_one, judge, row): row for row in rows}
        done = 0
        for fut in as_completed(futures):
            row = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"id": row["id"], "task_name": row["task_name"], "judge_score": 0.0,
                     "judge_criteria": None, "judge_reasoning": None, "judge_error": str(e)}
            results[r["id"]] = r
            if r.get("judge_error"):
                errors.append(f"{r['id']}: {r['judge_error']}")
            done += 1
            if done % 25 == 0 or done == len(rows):
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(rows) - done) / rate if rate > 0 else float("nan")
                print(f"[{done}/{len(rows)}] ({elapsed:.0f}s elapsed, eta {eta:.0f}s, {len(errors)} errors)", flush=True)

    with open(out_path, "w") as f:
        json.dump({"results": results, "n": len(results), "n_errors": len(errors),
                    "errors": errors[:50], "elapsed_s": round(time.time() - t0, 1)}, f, indent=2)
    print(f"\nSaved to {out_path}. Errors: {len(errors)}")
    for e in errors[:20]:
        print("  ERR:", e)


if __name__ == "__main__":
    main()
