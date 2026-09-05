#!/usr/bin/env python3
"""
Validate the VLM judge (llm_judge.py) the same way test_harness.py validates
the rule-based evaluators: 20 pairs per task (5 correct + 15 index-shifted
mismatches) built from the 5 downloaded GT samples, scored, and summarized
by discriminative gap (mean correct score - mean wrong score).

Only covers the tasks the judge is meant to supplement (rule-based score
known weak) that are still in the locked 10-task set (see image_evaluator.py
LOCKED_TASKS). As of 2026-08-25 TASK_DIRS is empty: ball_bounces_given_time
(the last task using the judge) had its rule-based score simplified to a
plain final_position distance check once its path_shape metric turned out to
depend on a generator visual style that's since changed upstream (see
BallBounceImageEvaluator's docstring) -- that deterministic check doesn't
need judge supplementation. key_door_matching was dropped entirely from the
locked set earlier (its GT data itself was found ambiguous, not just its
evaluator -- see README's Known Issues), and gravity_physics was replaced by
glass_refraction earlier too -- see llm_judge.py's module docstring for both.
This file is kept as the harness to reuse if/when another active task needs
judge supplementation in the future.
"""
import json
import os
import sys
import time

sys.path.insert(0, "/scratch/network/ssd2/junlin/ssl_mllm/Evaluation/VBVR-CustomEval/evaluators")
from llm_judge import VLMJudge

GT_BASE = "/scratch/network/ssd2/junlin/ssl_mllm/Evaluation/VBVR-EvalKit/VBVR-Bench"
OUT_PATH = "/scratch/network/ssd2/junlin/ssl_mllm/Evaluation/VBVR-CustomEval/validation/judge_validation_results.json"

TASK_DIRS = {}


def build_pairs(n_samples=5, n_shifts=3):
    pairs = []
    for i in range(n_samples):
        pairs.append(("correct", i, i))
    for shift in range(1, n_shifts + 1):
        for i in range(n_samples):
            pairs.append(("wrong", i, (i + shift) % n_samples))
    return pairs


def main():
    judge = VLMJudge()
    results = {}

    for task_name, (split, gen_name) in TASK_DIRS.items():
        tdir = os.path.join(GT_BASE, split, gen_name)
        records = []
        print(f"=== {task_name} ===", flush=True)

        for kind, src_idx, tgt_idx in build_pairs():
            src = os.path.join(tdir, f"{src_idx:05d}")
            tgt = os.path.join(tdir, f"{tgt_idx:05d}")
            prompt = open(os.path.join(tgt, "prompt.txt")).read()

            t0 = time.time()
            r = judge.score(
                task_name=task_name,
                prompt_text=prompt,
                first_frame_path=os.path.join(tgt, "first_frame.png"),
                gen_final_path=os.path.join(src, "final_frame.png"),
                gt_final_path=os.path.join(tgt, "final_frame.png"),
            )
            elapsed = round(time.time() - t0, 2)
            rec = {"kind": kind, "src": src_idx, "tgt": tgt_idx, "score": r["score"],
                   "reasoning": r.get("reasoning"), "elapsed_s": elapsed, "error": r.get("error")}
            records.append(rec)
            print(f"  {kind:8} {src_idx}->{tgt_idx}  score={r['score']:.2f}  ({elapsed}s)  {r.get('reasoning','')[:80]}", flush=True)

        correct = [r["score"] for r in records if r["kind"] == "correct"]
        wrong = [r["score"] for r in records if r["kind"] == "wrong"]
        n_errors = sum(1 for r in records if r.get("error"))
        summary = {
            "judge_mean_correct": sum(correct) / len(correct),
            "judge_mean_wrong": sum(wrong) / len(wrong),
            "judge_discriminative_gap": sum(correct) / len(correct) - sum(wrong) / len(wrong),
            "n_errors": n_errors,
        }
        print(f"  SUMMARY: {json.dumps(summary, indent=2)}\n", flush=True)
        results[task_name] = {"summary": summary, "records": records}

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to", OUT_PATH)


if __name__ == "__main__":
    main()
