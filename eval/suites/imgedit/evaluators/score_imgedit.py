#!/usr/bin/env python
"""Score a completed ImgEdit-Bench generation run with the local ImgEdit_Judge.

Folds upstream's 3-stage basic_bench.py -> step1_get_avgscore.py ->
step2_typescore.py pipeline into one script (no GPT-4o API calls, no
ThreadPoolExecutor -- one local GPU-bound model, called sequentially): for
every item in a --run-dir's gen_manifest.json, judges (original, generated)
against the matching judge_prompt.json rubric, then averages per-item scores
and per-edit_type scores.

Resumable like inference/gen_imgedit_internvlu.py: skips items already
present in --out's raw-response dict on re-run.

Usage (env: imgedit-eval-env or internvlu, both have qwen_vl_utils + a
transformers new enough for Qwen2_5_VLForConditionalGeneration):
  CUDA_VISIBLE_DEVICES=<g> python evaluators/score_imgedit.py \
      --data_root /scratch/local/ssd/junlin/data/ImgEdit/Benchmark/singleturn \
      --run_dir /scratch/local/ssd/junlin/results/ImgEdit/InternVL-U \
      --limit 0

Writes, inside --run_dir (or --out if given):
  judge_raw.json    # {key: {edit_type, raw_response, score}, ...}
  judge_typescore.json   # {edit_type: avg_score, ..., "overall": avg_score}
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SUITE_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _SUITE_ROOT)
sys.path.insert(0, _HERE)
from dataset import load_items, filter_missing, item_key  # noqa: E402
from imgedit_judge import ImgEditJudge, load_prompts  # noqa: E402


def compute_typescores(raw: dict) -> dict:
    by_type = {}
    for entry in raw.values():
        s = entry.get("score")
        if s is None:
            continue
        by_type.setdefault(entry["edit_type"], []).append(s)
    result = {et: round(sum(v) / len(v), 2) for et, v in by_type.items()}
    all_scores = [s for v in by_type.values() for s in v]
    if all_scores:
        result["overall"] = round(sum(all_scores) / len(all_scores), 2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Benchmark/singleturn/ (singleturn.json + judge_prompt.json + images)")
    ap.add_argument("--run_dir", required=True, help="A completed inference/gen_imgedit_internvlu.py --out dir (must contain gen_manifest.json)")
    ap.add_argument("--out", default=None, help="Defaults to --run_dir")
    ap.add_argument("--model_path", default=None, help="Defaults to ImgEditJudge's DEFAULT_MODEL_PATH")
    ap.add_argument("--limit", type=int, default=0, help="0 = all items")
    args = ap.parse_args()

    out_dir = args.out or args.run_dir
    os.makedirs(out_dir, exist_ok=True)

    manifest_path = os.path.join(args.run_dir, "gen_manifest.json")
    manifest = json.load(open(manifest_path))

    prompts = load_prompts(os.path.join(args.data_root, "judge_prompt.json"))
    items = {item_key(it): it for it in filter_missing(load_items(args.data_root))}

    keys = [k for k in manifest if k in items]
    if args.limit:
        keys = keys[: args.limit]

    raw_path = os.path.join(out_dir, "judge_raw.json")
    raw = json.load(open(raw_path)) if os.path.exists(raw_path) else {}

    todo = [k for k in keys if k not in raw]
    print(f"[judge][imgedit] {len(todo)}/{len(keys)} items to score (resuming {len(keys) - len(todo)} already done)", flush=True)
    if todo:
        judge_kwargs = {"model_path": args.model_path} if args.model_path else {}
        judge = ImgEditJudge(**judge_kwargs)
        for idx, key in enumerate(todo):
            it = items[key]
            gen_path = manifest[key]
            try:
                result = judge.score(
                    edit_type=it.edit_type,
                    prompt=prompts[it.edit_type],
                    edit_prompt=it.instruction,
                    original_path=it.input_path,
                    result_path=gen_path,
                )
                raw[key] = result
            except Exception as e:
                print(f"[judge][imgedit][FAIL] {key}: {e}", flush=True)
                raw[key] = {"edit_type": it.edit_type, "raw_response": None, "score": None, "error": str(e)}
            if (idx + 1) % 10 == 0 or idx == len(todo) - 1:
                json.dump(raw, open(raw_path, "w"), indent=2)
                print(f"[judge][imgedit] {idx + 1}/{len(todo)} scored", flush=True)

    json.dump(raw, open(raw_path, "w"), indent=2)
    typescores = compute_typescores(raw)
    json.dump(typescores, open(os.path.join(out_dir, "judge_typescore.json"), "w"), indent=2)
    print(f"[judge][imgedit] FINISHED -> {raw_path}", flush=True)
    print(json.dumps(typescores, indent=2))


if __name__ == "__main__":
    main()
