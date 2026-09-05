#!/usr/bin/env python3
"""
Validate all 15 image-only evaluators against 20 test pairs each:
  - 5 "correct" pairs: sample i's own frames scored against sample i's GT
  - 15 "wrong" pairs: sample i's frames scored against sample (i+1)/(i+2)/(i+3) mod 5's GT
    (simulates a model that generated something unrelated to the prompt)

For each pair, both the image-only evaluator AND the original unmodified video
evaluator are run (the video evaluator uses the *source* sample's real
ground_truth.mp4 as "model output" against the *target* sample's GT -- same
mismatch trick, just in video form) so we get a same-condition reference score.

Outputs one JSON with, per task: mean score on correct vs wrong pairs (for both
image and video mode), the gap between them (discriminative power), and for
Tier-1 tasks specifically, the exact-match rate against the video evaluator.
"""
import json
import os
import sys
import traceback

sys.path.insert(0, "/scratch/network/ssd2/junlin/ssl_mllm/Evaluation/VBVR-EvalKit")
sys.path.insert(0, "/scratch/network/ssd2/junlin/ssl_mllm/Evaluation/VBVR-CustomEval/evaluators")

from vbvr_bench.evaluators import TASK_EVALUATOR_MAP, get_task_category, get_split
from image_evaluator import TASK_IMAGE_EVALUATOR_MAP, LOCKED_TASKS

GT_BASE = "/scratch/network/ssd2/junlin/ssl_mllm/Evaluation/VBVR-EvalKit/VBVR-Bench"
OUT_PATH = "/scratch/network/ssd2/junlin/ssl_mllm/Evaluation/VBVR-CustomEval/validation/validation_results.json"

TIER = {
    'O-12_shape_color_then_scale_data-generator': 1,
    'O-13_shape_outline_then_move_data-generator': 1,
    'O-19_mirror_reflection_data-generator': 1,
    'O-44_rotation_puzzle_data-generator': 1,
    'G-3_stable_sort_data-generator': 1,
    'G-18_grid_shortest_path_data-generator': 1,
    'O-11_shape_color_then_move_data-generator': 1,
    'O-39_maze_data-generator': 1,
    'O-65_animal_size_sorting_data-generator': 1,
    'O-36_grid_shift_data-generator': 2,
    'G-5_multi_object_placement_data-generator': 2,
    'O-6_2d_geometric_transformation_data-generator': 2,
    'O-15_ball_bounces_given_time_data-generator': 3,
    'G-45_key_door_matching_data-generator': 3,
    'O-18_glass_refraction_data-generator': 1,
}


def task_dir(task_name):
    split_folder = 'Out-of-Domain_50' if get_split(task_name) == 'Out_of_Domain' else 'In-Domain_50'
    return os.path.join(GT_BASE, split_folder, task_name)


def build_pairs(task_name, n_samples=5, n_shifts=3):
    """Returns list of (kind, source_idx, target_idx) -- 5 correct + 15 wrong."""
    pairs = []
    for i in range(n_samples):
        pairs.append(('correct', i, i))
    for shift in range(1, n_shifts + 1):
        for i in range(n_samples):
            pairs.append(('wrong', i, (i + shift) % n_samples))
    return pairs


def run_task(task_name, device='cpu'):
    tdir = task_dir(task_name)
    video_eval = TASK_EVALUATOR_MAP[task_name](device=device, task_name=task_name)
    image_eval = TASK_IMAGE_EVALUATOR_MAP[task_name](device=device, task_name=task_name)

    pairs = build_pairs(task_name)
    records = []
    errors = []

    for kind, src_idx, tgt_idx in pairs:
        src = os.path.join(tdir, f"{src_idx:05d}")
        tgt = os.path.join(tdir, f"{tgt_idx:05d}")

        gt_video = os.path.join(tgt, "ground_truth.mp4")
        gt_first = os.path.join(tgt, "first_frame.png")
        gt_final = os.path.join(tgt, "final_frame.png")

        src_video = os.path.join(src, "ground_truth.mp4")
        src_first = os.path.join(src, "first_frame.png")
        src_final = os.path.join(src, "final_frame.png")

        rec = {'kind': kind, 'src': src_idx, 'tgt': tgt_idx}

        try:
            r_video = video_eval.evaluate({
                'video_path': src_video,
                'gt_video_path': gt_video,
                'gt_first_frame': gt_first,
                'gt_final_frame': gt_final,
            }, task_specific_only=True)
            rec['video_score'] = r_video['score']
            if r_video.get('error'):
                errors.append(f"{task_name} video {kind} {src_idx}->{tgt_idx}: {r_video['error']}")
        except Exception as e:
            rec['video_score'] = None
            errors.append(f"{task_name} video {kind} {src_idx}->{tgt_idx}: EXC {e}")

        try:
            r_image = image_eval.evaluate({
                'final_frame_path': src_final,
                'first_frame_path': src_first,
                'gt_video_path': gt_video,
                'gt_first_frame': gt_first,
                'gt_final_frame': gt_final,
            }, task_specific_only=True)
            rec['image_score'] = r_image['score']
            if r_image.get('error'):
                errors.append(f"{task_name} image {kind} {src_idx}->{tgt_idx}: {r_image['error']}")
        except Exception as e:
            rec['image_score'] = None
            errors.append(f"{task_name} image {kind} {src_idx}->{tgt_idx}: EXC {e}")
            traceback.print_exc()

        records.append(rec)

    return records, errors


def spearman(xs, ys):
    n = len(xs)
    if n < 2:
        return None

    def rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks

    rx = rank(xs)
    ry = rank(ys)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    cov = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    var_x = sum((rx[i] - mean_rx) ** 2 for i in range(n))
    var_y = sum((ry[i] - mean_ry) ** 2 for i in range(n))
    if var_x == 0 or var_y == 0:
        return None
    return cov / (var_x ** 0.5 * var_y ** 0.5)


def summarize(task_name, records):
    correct = [r for r in records if r['kind'] == 'correct']
    wrong = [r for r in records if r['kind'] == 'wrong']

    def mean(lst, key):
        vals = [r[key] for r in lst if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    img_correct = mean(correct, 'image_score')
    img_wrong = mean(wrong, 'image_score')
    vid_correct = mean(correct, 'video_score')
    vid_wrong = mean(wrong, 'video_score')

    both = [r for r in records if r['image_score'] is not None and r['video_score'] is not None]
    exact_matches = sum(1 for r in both if abs(r['image_score'] - r['video_score']) < 1e-6)
    exact_match_rate = exact_matches / len(both) if both else None

    corr = spearman([r['image_score'] for r in both], [r['video_score'] for r in both]) if both else None

    return {
        'task_name': task_name,
        'tier': TIER[task_name],
        'category': get_task_category(task_name),
        'split': get_split(task_name),
        'n_pairs': len(records),
        'image_mean_correct': img_correct,
        'image_mean_wrong': img_wrong,
        'image_discriminative_gap': (img_correct - img_wrong) if (img_correct is not None and img_wrong is not None) else None,
        'video_mean_correct': vid_correct,
        'video_mean_wrong': vid_wrong,
        'video_discriminative_gap': (vid_correct - vid_wrong) if (vid_correct is not None and vid_wrong is not None) else None,
        'image_vs_video_exact_match_rate': exact_match_rate,
        'image_vs_video_spearman': corr,
    }


def main():
    all_results = {}
    all_errors = []
    for task_name in LOCKED_TASKS:
        print(f"=== {task_name} ===", flush=True)
        records, errors = run_task(task_name)
        summary = summarize(task_name, records)
        all_results[task_name] = {'summary': summary, 'records': records}
        all_errors.extend(errors)
        print(json.dumps(summary, indent=2, default=str), flush=True)

    with open(OUT_PATH, "w") as f:
        json.dump({'results': all_results, 'errors': all_errors}, f, indent=2, default=str)

    print(f"\n\nSaved to {OUT_PATH}")
    print(f"Total errors: {len(all_errors)}")
    for e in all_errors[:30]:
        print("  ERR:", e)


if __name__ == "__main__":
    main()
