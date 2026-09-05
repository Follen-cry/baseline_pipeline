#!/usr/bin/env python3
"""Score the rotation_puzzle samples of the base/s0/s1/s2/s3 4task500sft runs
with the yes/no checklist judge (evaluators/judge_yesno.py), writing
<run_dir>/judge_scored_yesno.json alongside the existing judge_scored.json so
the old rubric judge, the rewritten rule-based scorer, and this one can all be
compared per sample.

Resumable: any id already present in the output file is skipped, so a killed
run loses no completed work.

    python score_rp_yesno.py                 # all five runs
    python score_rp_yesno.py --runs s0 s3
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "evaluators"))
from judge_yesno import YesNoJudge  # noqa: E402

RUN = "/scratch/network/ssd2/junlin/ssl_mllm/results/vbvr_target_pred_eval"
DEFAULT_RUNS = ["base", "s0", "s1", "s2", "s3"]
TASK = "rotation_puzzle"
OUT_NAME = "judge_scored_yesno.json"
CONCURRENCY = 8


def score_run(name, judge):
    run_dir = os.path.join(RUN, f"{name}_4task500sft")
    out_path = os.path.join(run_dir, OUT_NAME)
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    rows = [r for r in meta["results"] if r["task_name"] == TASK]

    results = {}
    if os.path.exists(out_path):
        results = json.load(open(out_path)).get("results", {})

    t0 = time.time()
    todo = [r for r in rows
            if not (r["id"] in results and results[r["id"]].get("score") is not None)]

    def one(row):
        out = judge.score(TASK, row["generated_image"], row["target_image"])
        return row["id"], {
            "id": row["id"], "task_name": TASK, "domain": row["domain"],
            "score": out["score"], "n_yes": out["n_yes"],
            "n_questions": out["n_questions"], "answers": out["answers"],
            "readout": out.get("readout"), "reasoning": out.get("reasoning"),
            "error": out.get("error"),
        }

    # The judge is a local vLLM server, which batches concurrent requests, so
    # the 500-sample corpus takes ~15 min instead of ~100 serial. Temperature
    # is 0, so concurrency changes no individual verdict.
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        for n, (rid, rec) in enumerate(ex.map(one, todo), 1):
            results[rid] = rec
            if n % 25 == 0:
                print(f"  {name}: {n}/{len(todo)}  ({time.time()-t0:.0f}s)", flush=True)

    scored = [r["score"] for r in results.values() if r.get("score") is not None]
    payload = {
        "judge": "judge_yesno.YesNoJudge (11 yes/no questions, candidate+GT only)",
        "task": TASK, "n": len(results), "n_scored": len(scored),
        "mean_score": (sum(scored) / len(scored)) if scored else None,
        "n_errors": sum(1 for r in results.values() if r.get("error")),
        "elapsed_s": round(time.time() - t0, 1),
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path, payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    args = ap.parse_args()

    judge = YesNoJudge()
    for name in args.runs:
        print(f"=== {name} ===", flush=True)
        path, p = score_run(name, judge)
        print(f"  -> {path}  mean={p['mean_score']:.4f}  "
              f"n={p['n_scored']}/{p['n']}  errors={p['n_errors']}  "
              f"{p['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
