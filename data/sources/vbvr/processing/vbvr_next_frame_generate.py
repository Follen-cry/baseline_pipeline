"""Run each VBVR-DataFactory generator to produce raw source videos for the
production next-frame (3-in -> 1-out) training split.

Production copy of vbvr_generate_raw.py, scoped down and re-parameterized per
the confirmed data-construction spec (2026-08-25). The smoke-test script
stays untouched as a reference; this file does not import it.

What's different from the smoke-test version
----------------------------------------------
- TASKS restricted to the 10 canonical VBVR in-domain generators. The 5 OOD
  tasks in vbvr_generate_raw.py's TASKS dict (gravity_physics/glass_refraction,
  shape_color_then_move, maze, 2d_geometric_transformation, animal_size_sorting)
  are dropped -- Task 1 only trains on in-domain data. OOD (and ID) generation
  for the separate 1-step target-prediction task lives in
  vbvr_target_pred_generate.py, not here.
- Per-task video count is NOT uniform. Measured against the 25 smoke-test
  videos/task already on disk (data/datasets/vbvr/smoke_test/raw/): 7 of the 10 tasks
  average ~1 sampling window/video (short clips -> small delta -> only one
  3-in-1-out window fits per video), so a flat 7,000 videos/task would fall
  short of the 10,000-window/task target in vbvr_next_frame_sample.py. Those
  7 tasks are raised to 11,000; the other 3 (key_door_matching @ ~2.5
  windows/video, rotation_puzzle and stable_sort @ ~2.0) clear 10,000 windows
  already at 7,000. See NUM_RAW_BY_TASK.
- Output goes to /scratch/network/ssd/junlin (1.7TB free), not this repo's
  ssd2 volume (36GB free against an estimated ~51GB of new data across both
  task types -- see the data-construction spec artifact).
- Seed convention is unchanged from the smoke test: a single --seed value
  shared across every task/generator in a run. Task 2's target-prediction
  generation (vbvr_target_pred_generate.py) uses a different fixed seed so
  the 5 tasks it shares with Task 1 (ball_bounces_given_time, grid_shift,
  rotation_puzzle, stable_sort, multi_object_placement) never draw the same
  source videos.
- Generation is CHUNKED, not one `--num-samples 11000` call per task. Found
  out why the hard way: a first production run (2026-08-25) launched all 10
  tasks as one `--num-samples <target>` call each, in parallel, and the
  system ran out of memory within ~2 hours -- one task (grid_shift) was
  OOM-killed by the kernel at 957/11000 videos with one of its
  `examples/generate.py` subprocesses sitting at ~85GB resident memory, and
  the other 9 were independently climbing toward similar sizes (collectively
  ~430GB of the machine's 503GB RAM) when they were killed by hand to stop
  the bleeding. Root cause: `core/base_generator.py`'s `generate_dataset()`
  (the default, un-overridden path for most of these 10 generators) builds
  the ENTIRE requested batch as a list of in-memory `TaskPair` objects --
  each holding full-resolution `first_image`/`final_image` PIL Images --
  and only calls `OutputWriter.write_dataset()` once ALL of them exist in
  memory; nothing is freed until the whole `--num-samples` batch is done.
  Peak memory scales with the requested count, so `--num-samples 11000` is
  not safe to request in one process. Fix: each task's target count is split
  into CHUNK_SIZE-sample sub-batches, each run as its OWN
  `examples/generate.py` subprocess (own process, own memory, freed on
  exit), written to its own `raw/<task>/chunk_NNNN/` subdirectory --
  vbvr_next_frame_sample.py's `rglob("ground_truth.*")` already finds videos
  at any depth, so no merging/renaming is needed afterward. Each chunk uses
  a distinct derived seed (`base_seed * 100000 + chunk_index`) so chunks
  don't reproduce identical content -- reusing the same seed across fresh
  processes reproduces the exact same sample sequence, since generation is
  deterministic given `random.seed(seed)`.

Usage:
    python VBVR-DataGeneration/next_frame/vbvr_next_frame_generate.py
    python VBVR-DataGeneration/next_frame/vbvr_next_frame_generate.py --tasks mirror_reflection grid_shift --num-raw 5 --timeout 300  # smoke check
    python VBVR-DataGeneration/next_frame/vbvr_next_frame_generate.py --seed 42
    python VBVR-DataGeneration/next_frame/vbvr_next_frame_generate.py --chunk-size 500  # override the default chunk size
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vbvr_task_presets import TASK_PRESETS, resolve_tasks

REPO_ROOT = Path("/scratch/network/ssd2/junlin/ssl_mllm")
FACTORY_ROOT = REPO_ROOT / "VBVR-DataFactory"
RAW_ROOT = Path("/scratch/network/ssd/junlin/vbvr_next_frame/raw")

# canonical task_name -> (generator repo dir, category)
# The 10 canonical in-domain tasks only -- see VBVR-CustomEval/README.md's
# "10 ID / 5 OOD overall" split and the data-construction spec artifact's
# Task taxonomy section. All 15 generators (with category + ID/OOD split)
# stay recorded in vbvr_generate_raw.py's TASKS dict for reference.
TASKS = {
    "ball_bounces_given_time":   ("O-15_ball_bounces_given_time_data-generator", "Knowledge"),
    "mirror_reflection":         ("O-19_mirror_reflection_data-generator", "Knowledge"),
    "shape_color_then_scale":    ("O-12_shape_color_then_scale_data-generator", "Abstraction"),
    "shape_outline_then_move":   ("O-13_shape_outline_then_move_data-generator", "Abstraction"),
    "grid_shortest_path":        ("G-18_grid_shortest_path_data-generator", "Spatiality"),
    "key_door_matching":         ("G-45_key_door_matching_data-generator", "Spatiality"),
    "grid_shift":                ("O-36_grid_shift_data-generator", "Transformation"),
    "rotation_puzzle":           ("O-44_rotation_puzzle_data-generator", "Transformation"),
    "stable_sort":               ("G-3_stable_sort_data-generator", "Perception"),
    "multi_object_placement":    ("G-5_multi_object_placement_data-generator", "Perception"),
}

# Per-task raw video count. Default 7,000; raised to 11,000 for the 7 tasks
# measured to average ~1 sampling window/video (see module docstring). Tasks
# not listed here fall back to DEFAULT_NUM_RAW.
DEFAULT_NUM_RAW = 7000
NUM_RAW_BY_TASK = {
    "ball_bounces_given_time": 11000,
    "mirror_reflection": 11000,
    "shape_color_then_scale": 11000,
    "shape_outline_then_move": 11000,
    "grid_shortest_path": 11000,
    "grid_shift": 11000,
    "multi_object_placement": 11000,
}

# --- 2026-08-29 extension --------------------------------------------------
# 4 of the 5 originally-excluded OOD tasks (all but glass_refraction, which
# has no train jsonl yet -- see Evaluation/VBVR-CustomEval/README.md's Known
# Issues), now folded in as ordinary next-frame training data (not held out)
# for a 100k(ID) + 40k(these 4) = 140k-sample continuation-SFT run. Kept as a
# separate dict rather than merged into TASKS so the module's DEFAULT task
# selection (no --tasks) is unchanged; select these explicitly via
# `--tasks shape_color_then_move maze 2d_geometric_transformation animal_size_sorting`.
# repo dirs/categories match data/scripts/vbvr_generate_raw.py's TASKS dict.
# num-raw counts derived the same way as NUM_RAW_BY_TASK above (measured
# windows/video against data/datasets/vbvr/smoke_test/raw/): shape_color_then_move,
# 2d_geometric_transformation, animal_size_sorting average ~1 window/video ->
# 11,000 raw needed for a 10,000-window target; maze averages ~1.6
# windows/video -> 7,000 raw clears the target already.
OOD_EXT_TASKS = {
    "shape_color_then_move":       ("O-11_shape_color_then_move_data-generator", "Abstraction"),
    "maze":                        ("O-39_maze_data-generator", "Spatiality"),
    "2d_geometric_transformation": ("O-6_2d_geometric_transformation_data-generator", "Transformation"),
    "animal_size_sorting":         ("O-65_animal_size_sorting_data-generator", "Perception"),
}
NUM_RAW_BY_TASK_OOD_EXT = {
    "shape_color_then_move": 11000,
    "maze": 7000,
    "2d_geometric_transformation": 11000,
    "animal_size_sorting": 11000,
}
ALL_TASKS = {**TASKS, **OOD_EXT_TASKS}
ALL_NUM_RAW_BY_TASK = {**NUM_RAW_BY_TASK, **NUM_RAW_BY_TASK_OOD_EXT}

# Samples per `examples/generate.py` subprocess call. Bounds peak memory per
# process regardless of the total per-task target -- see the OOM incident in
# the module docstring. Not tuned precisely (no clean count->memory model was
# established, since two tasks hit similar RSS at very different sample
# counts before the fix) -- picked conservatively small and verified safe by
# actually watching memory across a multi-chunk run, not derived analytically.
DEFAULT_CHUNK_SIZE = 500


def run_chunk(gen_dir, chunk_out_dir, this_chunk, chunk_seed, timeout):
    cmd = [sys.executable, "examples/generate.py",
           "--num-samples", str(this_chunk),
           "--output", str(chunk_out_dir),
           "--seed", str(chunk_seed)]
    try:
        proc = subprocess.run(cmd, cwd=str(gen_dir), capture_output=True, text=True, timeout=timeout)
        return proc.returncode == 0, proc.stderr
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {timeout}s"


def run_one(task_name, repo_dir, num_raw, seed, timeout, chunk_size):
    gen_dir = FACTORY_ROOT / repo_dir
    task_out_dir = RAW_ROOT / task_name
    task_out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    remaining, chunk_idx, ok_all, last_stderr = num_raw, 0, True, ""
    while remaining > 0:
        this_chunk = min(chunk_size, remaining)
        chunk_out_dir = task_out_dir / f"chunk_{chunk_idx:04d}"
        chunk_seed = seed * 100000 + chunk_idx  # distinct per chunk, still reproducible
        ok, stderr = run_chunk(gen_dir, chunk_out_dir, this_chunk, chunk_seed, timeout)
        if not ok:
            ok_all, last_stderr = False, stderr
            break
        remaining -= this_chunk
        chunk_idx += 1
    dt = time.time() - t0

    n_videos = len(list(task_out_dir.rglob("ground_truth.*")))
    ok_final = ok_all and n_videos >= num_raw
    status = "OK" if ok_final else "FAIL"
    rate = dt / max(n_videos, 1)
    print(f"[{status}] {task_name:28s} videos={n_videos:6d}/{num_raw:<6d} {dt:7.1f}s  ({rate:.2f}s/video, {chunk_idx} chunks)", flush=True)
    if not ok_all:
        print(f"  stderr tail: {last_stderr.strip()[-800:]}", flush=True)
    return ok_final, n_videos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=None, help="subset of task names (default: all 10)")
    ap.add_argument("--task-preset", default=None,
                     help=f"named --tasks shortcut from vbvr_task_presets.py (valid: {list(TASK_PRESETS)}); "
                          "mutually exclusive with --tasks")
    ap.add_argument("--num-raw", type=int, default=None,
                     help="override video count for ALL selected tasks (default: per-task NUM_RAW_BY_TASK / 7,000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=int, default=6 * 3600,
                     help="per-CHUNK subprocess timeout in seconds (default: 6h -- generous, chunks are small)")
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                     help=f"samples per examples/generate.py subprocess call (default: {DEFAULT_CHUNK_SIZE} -- bounds peak memory, see module docstring)")
    args = ap.parse_args()

    task_names = resolve_tasks(args.tasks, args.task_preset, ALL_TASKS, list(TASKS.keys()))
    unknown = [t for t in task_names if t not in ALL_TASKS]
    if unknown:
        raise SystemExit(f"unknown task names: {unknown} (valid: {list(ALL_TASKS.keys())})")

    print(f"Generating raw videos for {len(task_names)} task(s) -> {RAW_ROOT}")
    results = {}
    for task_name in task_names:
        repo_dir, category = ALL_TASKS[task_name]
        num_raw = args.num_raw if args.num_raw is not None else ALL_NUM_RAW_BY_TASK.get(task_name, DEFAULT_NUM_RAW)
        ok, n_videos = run_one(task_name, repo_dir, num_raw, args.seed, args.timeout, args.chunk_size)
        results[task_name] = {"ok": ok, "n_videos": n_videos, "category": category, "num_raw_requested": num_raw}

    failed = [t for t, r in results.items() if not r["ok"]]
    total_videos = sum(r["n_videos"] for r in results.values())
    print(f"\nDone. {len(task_names) - len(failed)}/{len(task_names)} tasks OK. Total videos: {total_videos}")
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
