"""Run the BASE InternVL-U model over the real 1-step-target-frame-prediction
eval split (data/vbvr_target_pred_{id,ood}/target_pred_{id,ood}_eval.jsonl --
100 samples per active task, 9 of the 10 LOCKED_TASKS; glass_refraction has
no eval split because its DataFactory generator repo is access-restricted,
same known limitation documented in VBVR-CustomEval/README.md).

Single conditioning image per sample (unlike Evaluation/vbvr_baseline_runners/run_vbvr_smoke_infer.py's
3-in-1-out smoke split), so this reuses that script's building blocks
(prescale_rows / build_dataset / generate_one) directly rather than
duplicating them -- with only 1 input frame, its --vae-cond-last-frame path's
_keep_last_true_per_row masking hack is a no-op (nothing to restrict: the
default image-hidden-state mask already has just the one column), so the
same function is correct here unmodified.

--vae-cond-last-frame is the default in this script (unlike the smoke
script, where it's opt-in) because it reproduces true generation_mode="image"
semantics -- the conditioning image goes through both the ViT (ImgConTEXT
tokens / VLM hidden states) AND the VAE (pixel-level diffusion condition),
which is what InternVLUProcessor.__call__ always does whenever an image is
passed with generation_mode != "text" (see processing_internvlu.py
lines ~1176-1220). Pass --no-vae-cond to fall back to VLM-hidden-states-only
conditioning for comparison.

    python run_target_pred_eval.py --gpu 2 --limit-per-task 2   # timing/smoke test
    python run_target_pred_eval.py --gpu 2                      # full 900-sample run
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

SMOKE_DIR = "/scratch/network/ssd2/junlin/ssl_mllm/Evaluation/vbvr"
PKG = "/scratch/network/ssd2/junlin/ssl_mllm/Model_Related/InternVLU/InternVL-U"

ID_JSONL = "/scratch/network/ssd2/junlin/ssl_mllm/data/datasets/vbvr_target_pred_id/target_pred_id_eval.jsonl"
OOD_JSONL = "/scratch/network/ssd2/junlin/ssl_mllm/data/datasets/vbvr_target_pred_ood/target_pred_ood_eval.jsonl"
OUT_ROOT = "/scratch/network/ssd2/junlin/ssl_mllm/results/vbvr_target_pred_eval"
CKPT_BASE = "/scratch/network/ssd2/junlin/models/InternVL-U"
SEED = 42
GEN_SIZE = 512


def load_rows():
    rows = [json.loads(l) for l in open(ID_JSONL)]
    rows += [json.loads(l) for l in open(OOD_JSONL)]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--limit-per-task", type=int, default=None)
    ap.add_argument("--gen-size", type=int, default=GEN_SIZE)
    ap.add_argument("--prescale-input", type=int, default=512,
                     help="matches inference/settings.py's IMAGE_PREPROCESSING "
                          "convention (default: 512, pass 0 to disable)")
    ap.add_argument("--no-vae-cond", action="store_true",
                     help="disable VAE pixel-level conditioning, use VLM "
                          "hidden-states-only conditioning instead (default: "
                          "on, matching real generation_mode='image' semantics)")
    ap.add_argument("--tag", default="base_prescale512_vaecond",
                     help="output subdirectory name under results/vbvr_target_pred_eval/")
    ap.add_argument("--ckpt", default=CKPT_BASE,
                     help="InternVL-U pipeline snapshot to evaluate (default: the "
                          "base checkpoint, CKPT_BASE) -- point this at a merged "
                          "fine-tuned pipeline (e.g. an internvlu-*-merged dir) to "
                          "eval that checkpoint on the same 900-sample split instead")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from transformers import AutoTokenizer
    sys.path.insert(0, SMOKE_DIR)
    sys.path.insert(0, PKG)
    from run_vbvr_smoke_infer import prescale_rows, build_dataset, generate_one
    from internvlu import InternVLUPipeline

    rows = load_rows()
    if args.tasks:
        rows = [r for r in rows if r["task_name"] in args.tasks]
    if args.limit_per_task:
        by_task = defaultdict(list)
        for r in rows:
            by_task[r["task_name"]].append(r)
        rows = [r for task_rows in by_task.values() for r in task_rows[:args.limit_per_task]]

    print(f"[target_pred_eval] {len(rows)} samples across {len({r['task_name'] for r in rows})} tasks", flush=True)

    if args.prescale_input:
        rows = prescale_rows(rows, args.prescale_input)
        print(f"[target_pred_eval] pre-scaled input frames to {args.prescale_input}x{args.prescale_input}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.ckpt + "/vlm", trust_remote_code=True, use_fast=False)
    ds = build_dataset(tok, args.gen_size, rows)

    out_dir = os.path.join(OUT_ROOT, args.tag)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[target_pred_eval] loading {args.ckpt}", flush=True)
    pipe = InternVLUPipeline.from_pretrained(args.ckpt, torch_dtype=torch.bfloat16).to("cuda")

    vae_cond = not args.no_vae_cond
    t_start = time.time()
    results_meta = []
    for i, row in enumerate(rows):
        task_dir = os.path.join(out_dir, row["task_name"])
        os.makedirs(task_dir, exist_ok=True)
        dst = os.path.join(task_dir, row["id"] + ".png")
        if os.path.exists(dst):
            print(f"[target_pred_eval] {i+1}/{len(rows)} skip {row['id']}", flush=True)
        else:
            sample = ds[i]
            img = generate_one(pipe, sample, SEED + i, torch, args.gen_size,
                                input_frame_paths=row["image"],
                                vae_cond_last_frame=vae_cond)
            img.save(dst)
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(rows) - i - 1) / rate if rate > 0 else float("nan")
            print(f"[target_pred_eval] {i+1}/{len(rows)} {row['id']} {img.size} "
                  f"({elapsed:.0f}s elapsed, eta {eta:.0f}s)", flush=True)

        prompt_text = row["conversations"][0]["value"]
        if prompt_text.startswith("<image>\n"):
            prompt_text = prompt_text[len("<image>\n"):]
        results_meta.append({"id": row["id"], "task_name": row["task_name"],
                              "category": row["category"], "domain": row["domain"],
                              "prompt": prompt_text,
                              "input_image": row["image"][0], "target_image": row["target_image"],
                              "generated_image": dst})

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"model": "InternVL-U (base)" if args.ckpt == CKPT_BASE else args.ckpt,
                    "model_path": args.ckpt, "gen_size": args.gen_size,
                    "prescale_input": args.prescale_input, "vae_cond_last_frame": vae_cond,
                    "seed": SEED, "n": len(rows), "results": results_meta}, f, indent=2)
    print(f"[target_pred_eval] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
