"""Build VBVR's S3 (VC2IA-F: 3 context frames + caption -> target image + MCQ
answer) training rows -- the "dual" MCQ + generation joint-supervision format,
same shape as NWM/EPIC/panda70m's *_dual_noaction jsonls (imgen_form=
"multimodal", is_option, cond_image, single-letter GPT answer -- see e.g.
data/datasets/epic_ssl/epic_dual_action/epic_dual_action_train.jsonl for the reference
row shape this matches byte-for-byte in structure).

Unlike NWM/EPIC/panda70m, VBVR has no pose/temporal-distance infrastructure to
mine hard negatives from, and no pre-existing "ffs" (selection-only) task to
convert via tools/make_nwm_dual_ssl.py (that converter is also hardcoded to
N_CTX=4; VBVR uses 3 context frames, so it isn't reused here -- see this
project's task-preset/pipeline planning notes). Distractor scheme (v1, see
module docstring caveat below): for a given window, the 3 wrong options are
OTHER windows' target frames from the SAME task -- these already exist on
disk (zero new rendering), and are visually task-consistent (same rendering
style) but semantically wrong (different source video / different generator
parameters), so the task requires reading the specific 3 context frames, not
just recognizing "which image looks like this task."

CAVEAT (flagged, not resolved here): because distractors come from
DIFFERENT videos/parameter instances of the same task rather than
DIFFERENT TIMES of the SAME video, the MCQ may be easier than NWM/EPIC's
pose/temporal-distance negatives (a distractor can differ in object count,
color, or layout in a way that's visually obvious without truly reasoning
about the 3 context frames' motion). Fine for a pipeline smoke test; revisit
before scaling if the trained model saturates the MCQ trivially.

Input: already-extracted window dirs from vbvr_next_frame_sample.py (this
script's own SAMPLES_ROOT/<task>/NNNNN/ layout: frame_0/1/2.png + target.png
+ window_meta.json carrying "raw_parameters" -- same requirement as that
script's --reuse-samples path). Needs >=4 windows for a task to build even
one MCQ row (1 anchor + 3 distractor donors).

Usage:
    python vbvr_next_frame_make_dual.py --task-preset pilot5 --limit 3
    python vbvr_next_frame_make_dual.py --tasks rotation_puzzle --samples-root /scratch/.../smoke_samples --limit 2
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from vbvr_next_frame_generate import TASKS, ALL_TASKS
from vbvr_next_frame_prompt_adapt import scene_description

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vbvr_task_presets import TASK_PRESETS, resolve_tasks

REPO_ROOT = Path("/scratch/network/ssd2/junlin/ssl_mllm")
SAMPLES_ROOT = Path("/scratch/network/ssd/junlin/vbvr_next_frame/samples")
DATA_DIR = REPO_ROOT / "data" / "vbvr_next_frame"
META_DIR = REPO_ROOT / "data" / "meta"

LETTERS = "ABCD"
NUM_OPTIONS = 4
EVAL_COUNT_PER_TASK = 200

MCQ_QUESTION = (
    "You are shown 3 consecutive frames sampled evenly in time from this "
    "sequence, followed by {n} candidate next frames. Based on the "
    "direction and rate of motion or transformation established across the "
    "3 context frames, identify which candidate is the true next frame.\n"
    "{options}\n"
    "Which option is the true next frame? Answer with a single letter ({letters})."
).format(n=NUM_OPTIONS,
         options="\n".join(f"Option {L}: <image>" for L in LETTERS),
         letters=", ".join(LETTERS[:-1]) + f", or {LETTERS[-1]}")


def build_mcq_prompt(task_name, raw_parameters, include_scene=True):
    head = "<image>\n<image>\n<image>\n"
    if not include_scene:
        return head + MCQ_QUESTION
    scene = scene_description(task_name, raw_parameters)
    return head + scene + "\n\n" + MCQ_QUESTION


def _load_window(win_dir):
    meta_path = win_dir / "window_meta.json"
    target_path = win_dir / "target.png"
    frame_paths = [win_dir / f"frame_{k}.png" for k in range(3)]
    if not meta_path.exists() or not target_path.exists() or not all(p.exists() for p in frame_paths):
        return None
    meta = json.loads(meta_path.read_text())
    if meta.get("raw_parameters") is None:
        raise ValueError(
            f"{meta_path} has no 'raw_parameters' (pre-dates --direction/--no-caption "
            "support in vbvr_next_frame_sample.py) -- re-extract this task without "
            "--reuse-samples first."
        )
    return {"dir": win_dir, "meta": meta, "frame_paths": frame_paths, "target_path": target_path}


def split_window_dirs(window_dirs, eval_count, seed):
    """Window-level (not row-level) train/eval split: an eval-holdout window's
    distractors are drawn ONLY from other eval-holdout windows (never from
    train), and vice versa, so no eval image ever appears -- even as a wrong
    MCQ option -- in a training row. Returns (train_dirs, eval_dirs), both
    sorted for determinism downstream."""
    dirs = list(window_dirs)
    rng = random.Random(f"{seed}:holdout")
    rng.shuffle(dirs)
    eval_dirs = sorted(dirs[:eval_count])
    train_dirs = sorted(dirs[eval_count:])
    return train_dirs, eval_dirs


def build_dual_rows(task_name, category, window_dirs, rng, include_scene=True, setting_label="S3",
                     eval_holdout=False):
    windows = [w for w in (_load_window(d) for d in sorted(window_dirs)) if w is not None]
    if len(windows) < NUM_OPTIONS:
        raise ValueError(
            f"{task_name}: only {len(windows)} usable windows, need >= {NUM_OPTIONS} "
            "(1 anchor + 3 distractor donors) to build even one MCQ row"
        )

    rows = []
    for idx, w in enumerate(windows):
        others = [o for o in windows if o is not w]
        distractors = rng.sample(others, NUM_OPTIONS - 1)

        candidates = [(w["target_path"], "ground_truth")] + [(d["target_path"], "distractor") for d in distractors]
        order = list(range(NUM_OPTIONS))
        rng.shuffle(order)
        candidates = [candidates[i] for i in order]
        option_paths = [c[0] for c in candidates]
        answer_idx = next(i for i, (_, kind) in enumerate(candidates) if kind == "ground_truth")
        answer_letter = LETTERS[answer_idx]

        raw_parameters = w["meta"]["raw_parameters"]
        prompt = build_mcq_prompt(task_name, raw_parameters, include_scene=include_scene)

        ctx_paths = w["frame_paths"]  # forward-only: [f0, f1, f2], matches S2's ctx order
        cond_path = ctx_paths[-1]  # last context frame -- never equals any option (verified below)
        image_paths = [str(p) for p in ctx_paths] + [str(p) for p in option_paths]
        is_option = [0] * len(ctx_paths) + [1] * NUM_OPTIONS

        # cond_image / target_image leak check: cond must never be a candidate
        # (would expose the ground truth, or a distractor, through the VAE
        # conditioning channel) -- context frames and target frames are always
        # different physical files by construction, but assert it anyway since
        # this is new code, not a battle-tested shared converter.
        assert str(cond_path) not in {str(p) for p in option_paths}, \
            f"{task_name} idx={idx}: cond_image collides with an MCQ option"

        row = {
            "id": f"vbvr_{task_name}_dual_{'eval' if eval_holdout else 'train'}_{idx:05d}",
            "task_type": "imgen",
            "imgen_form": "multimodal",
            "task_name": task_name,
            "category": category,
            "split": "ID",
            "eval_holdout": bool(eval_holdout),
            "direction": "forward",
            "has_caption": bool(include_scene),
            "image": image_paths,
            "target_image": str(option_paths[answer_idx]),
            "cond_image": str(cond_path),
            "is_option": is_option,
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": answer_letter},
            ],
        }
        if setting_label:
            row["setting"] = setting_label
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--task-preset", default=None,
                     help=f"named --tasks shortcut from vbvr_task_presets.py (valid: {list(TASK_PRESETS)})")
    ap.add_argument("--samples-root", type=str, default=None,
                     help="Root containing <task>/<NNNNN>/ window dirs already built by "
                          "vbvr_next_frame_sample.py (default: the shared production location).")
    ap.add_argument("--limit", type=int, default=None,
                     help="Use only the first N window dirs per task (sorted order) as MCQ "
                          "anchors -- distractors are still drawn from this same limited pool, "
                          "so pass at least 4.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-caption", action="store_true",
                     help="Build the MCQ prompt without the per-task scene description "
                          "(not a named S0-S3 setting on its own -- S3 requires caption "
                          "by definition -- but kept available for experimentation).")
    ap.add_argument("--setting-label", type=str, default="S3")
    ap.add_argument("--eval-count", type=int, default=EVAL_COUNT_PER_TASK,
                     help=f"Windows held out PER TASK for eval (default: {EVAL_COUNT_PER_TASK}). "
                          "Window-level split -- an eval row's distractors are only ever drawn "
                          "from other eval windows, never from train. Pass a smaller number for "
                          "smoke-scale runs (needs >= 4 windows in EACH pool to build any MCQ row).")
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--out-stem", type=str, default="next_frame_dual")
    args = ap.parse_args()

    task_names = resolve_tasks(args.tasks, args.task_preset, ALL_TASKS, list(TASKS.keys()))
    unknown = [t for t in task_names if t not in ALL_TASKS]
    if unknown:
        raise SystemExit(f"unknown task names: {unknown} (valid: {list(ALL_TASKS.keys())})")

    samples_root = Path(args.samples_root).resolve() if args.samples_root else SAMPLES_ROOT
    out_dir = Path(args.out_dir).resolve() if args.out_dir else DATA_DIR

    all_train_rows, all_eval_rows = [], []
    for task_name in task_names:
        _, category = ALL_TASKS[task_name]
        task_dir = samples_root / task_name
        window_dirs = sorted(task_dir.glob("[0-9]" * 5))
        if args.limit is not None:
            window_dirs = window_dirs[:args.limit]

        train_dirs, eval_dirs = split_window_dirs(window_dirs, args.eval_count, args.seed)

        rng_train = random.Random(f"{args.seed}:{task_name}:dual:train")
        train_rows = build_dual_rows(task_name, category, train_dirs, rng_train,
                                      include_scene=not args.no_caption, setting_label=args.setting_label,
                                      eval_holdout=False)
        rng_eval = random.Random(f"{args.seed}:{task_name}:dual:eval")
        eval_rows = build_dual_rows(task_name, category, eval_dirs, rng_eval,
                                     include_scene=not args.no_caption, setting_label=args.setting_label,
                                     eval_holdout=True)
        all_train_rows.extend(train_rows)
        all_eval_rows.extend(eval_rows)
        print(f"[OK] {task_name:28s} dual rows: train={len(train_rows):6d} eval={len(eval_rows):4d} "
              f"(from {len(train_dirs)}+{len(eval_dirs)} windows, {task_dir})", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_nocap" if args.no_caption else ""
    max_dynamic_patch = 3 + NUM_OPTIONS  # 3 context + 4 options, matches NWM/EPIC/panda70m dual convention

    for split_name, rows in (("train", all_train_rows), ("eval", all_eval_rows)):
        jsonl_path = out_dir / f"{args.out_stem}_{split_name}{suffix}.jsonl"
        with open(jsonl_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        meta_name = f"vbvr_{args.out_stem}{suffix}" + ("" if split_name == "train" else "_eval")
        meta_path = META_DIR / f"{meta_name}_meta.json"
        meta = {
            meta_name: {
                "root": "/",
                "annotation": str(jsonl_path),
                "data_augment": False,
                "max_dynamic_patch": max_dynamic_patch,
                "repeat_time": 1,
                "length": len(rows),
                "task_type": "imgen",
            }
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"{split_name}: {len(rows)} rows -> {jsonl_path}  (meta -> {meta_path})")


if __name__ == "__main__":
    main()
