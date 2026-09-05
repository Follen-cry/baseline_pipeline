#!/usr/bin/env python3
"""Re-score the exact same shared-sample subset that
artifacts/gen_9task_sft_comparison.py displays (10-20 per task x 9 tasks x
3 models = 330 samples) with a second judge -- Qwen3.8-27B run locally via
`transformers` (llm_judge_transformers.TransformersVLMJudge) -- instead of
the original qwen3-vl-30b-fp8 vLLM judge.

Scoped to the shown subset, not the full 900x3 corpus: a single call
measured at 40-70s (task-criteria-count dependent, already using
enable_thinking=False, the fastest working config found -- see
llm_judge_transformers.py's docstring) makes the full 2,700-call corpus a
~35-40 hour job, infeasible in one sitting; the 330-call shared subset is
~4 hours and directly covers everything the artifact actually displays
per-sample. User-confirmed scope choice, 2026-08-31.

Sampling logic (SEED, N_SHARED_PER_TASK, EXTRA_SAMPLES, the by-task-id
grouping) is copy-identical to gen_9task_sft_comparison.py's -- so
shared_picks here is bit-identical to what that script computes, and the
resulting judge_scored_qwen38.json files line up 1:1 with the sample rows
already on the page.

Writes <run_dir>/judge_scored_qwen38.json per model, same shape as the
existing judge_scored.json but only for the 330-id subset, plus a
"scope": "shared_subset" marker so downstream readers don't mistake it for
a full-corpus score file. Resumable: skips any id already present in an
existing output file, so a killed/restarted run loses no completed work.

    python score_shared_with_qwen38.py
    python score_shared_with_qwen38.py --models frombase   # just one model
"""
import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "evaluators"))
from llm_judge_transformers import TransformersVLMJudge  # noqa: E402

RUN = "/scratch/network/ssd2/junlin/ssl_mllm/results/vbvr_target_pred_eval"
SEED = 92820260828
N_SHARED_PER_TASK = 10
EXTRA_SAMPLES = {"maze": 10, "2d_geometric_transformation": 10}
CATEGORY_OF = {
    "ball_bounces_given_time": "Knowledge", "stable_sort": "Perception",
    "multi_object_placement": "Perception", "grid_shift": "Transformation",
    "rotation_puzzle": "Transformation", "shape_color_then_move": "Abstraction",
    "animal_size_sorting": "Perception", "maze": "Spatiality",
    "2d_geometric_transformation": "Transformation",
}
TASK_ORDER = list(CATEGORY_OF.keys())
MODELS = [
    ("frombase", f"{RUN}/frombase_9task4500"),
    ("from100k", f"{RUN}/from100k_9task4500"),
    ("from140k", f"{RUN}/from140k_9task4500"),
]


def compute_shared_picks(base_meta):
    by_task_ids = {}
    for rid, row in base_meta.items():
        by_task_ids.setdefault(row["task_name"], []).append(rid)
    rng = random.Random(SEED)
    shared_picks = {t: rng.sample(ids, min(N_SHARED_PER_TASK, len(ids))) for t, ids in by_task_ids.items()}
    for t, n_extra in EXTRA_SAMPLES.items():
        ids = by_task_ids.get(t, [])
        already = set(shared_picks.get(t, []))
        remaining = [i for i in ids if i not in already]
        extra_rng = random.Random(f"{SEED}:extra:{t}")
        extra_picks = extra_rng.sample(remaining, min(n_extra, len(remaining)))
        shared_picks[t] = shared_picks.get(t, []) + extra_picks
    return shared_picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                     help="subset of {frombase,from100k,from140k}; default all 3")
    args = ap.parse_args()
    models = [(k, d) for k, d in MODELS if args.models is None or k in args.models]

    base_meta = json.load(open(os.path.join(MODELS[0][1], "meta.json")))["results"]
    base_meta_by_id = {r["id"]: r for r in base_meta}
    shared_picks = compute_shared_picks(base_meta_by_id)
    all_ids_ordered = [(t, rid) for t in TASK_ORDER for rid in shared_picks[t]]
    print(f"Shared subset: {len(all_ids_ordered)} ids across {len(TASK_ORDER)} tasks", flush=True)

    print("Loading Qwen3.8-27B judge (transformers, ~10s)...", flush=True)
    judge = TransformersVLMJudge()
    print("Judge loaded.", flush=True)

    for key, run_dir in models:
        out_path = os.path.join(run_dir, "judge_scored_qwen38.json")
        meta = json.load(open(os.path.join(run_dir, "meta.json")))
        meta_by_id = {r["id"]: r for r in meta["results"]}

        existing = {}
        if os.path.exists(out_path):
            existing = json.load(open(out_path)).get("results", {})
            print(f"[{key}] resuming: {len(existing)} already scored", flush=True)

        results = dict(existing)
        todo = [(t, rid) for t, rid in all_ids_ordered if rid not in existing]
        print(f"[{key}] scoring {len(todo)} remaining of {len(all_ids_ordered)}", flush=True)

        t0 = time.time()
        for i, (t, rid) in enumerate(todo):
            row = meta_by_id.get(rid)
            base_row = base_meta_by_id[rid]
            if row is None:
                results[rid] = {"id": rid, "task_name": t, "judge_score": None,
                                 "judge_error": "id not found in this model's meta.json"}
            else:
                j = judge.score(
                    task_name=t, prompt_text=base_row["prompt"],
                    first_frame_path=base_row["input_image"],
                    gen_final_path=row["generated_image"],
                    gt_final_path=base_row["target_image"],
                )
                results[rid] = {
                    "id": rid, "task_name": t, "judge_score": j["score"],
                    "judge_criteria": j.get("criteria"), "judge_reasoning": j.get("reasoning"),
                    "judge_error": j.get("error"),
                }
            if (i + 1) % 5 == 0 or (i + 1) == len(todo):
                with open(out_path, "w") as f:
                    json.dump({"results": results, "n": len(results),
                               "scope": "shared_subset", "shared_subset_size": len(all_ids_ordered),
                               "judge_model": "Qwen3.8-27B-bf16 (transformers, enable_thinking=False)"},
                              f, indent=2)
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(todo) - i - 1) / rate if rate > 0 else float("nan")
                print(f"[{key}] [{i+1}/{len(todo)}] {elapsed:.0f}s elapsed, eta {eta:.0f}s", flush=True)

        print(f"[{key}] done -> {out_path}", flush=True)

    print("\nAll models scored.")


if __name__ == "__main__":
    main()
