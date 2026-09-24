"""Recipe: "uniform data split regarding data source".

Rebuilds WorldPrediction-WM's train/eval split, replacing the original
source-level split (data/v1/sources/worldprediction/ -- COIN=eval, CrossTask+
EPIC-KITCHENS-100+IKEAASM=train) with a row-level 80/20 split applied
independently WITHIN each of the 4 usable data sources (COIN, CrossTask,
EPIC-KITCHENS-100, IKEAASM), so both train and eval contain samples from
every source in the same proportion. Motivated by the prior split's train/
eval domain gap (train was egocentric-kitchen/furniture-assembly-heavy,
eval was COIN's third-person how-to video) possibly limiting transfer.

This is a pure recombination of already-processed rows -- no video decoding,
no frame re-extraction. Supports --frames to build off of either the 4-frame
(default, wm_train.jsonl/wm_eval.jsonl) or N-frame (wm_train_{n}f.jsonl/
wm_eval_{n}f.jsonl) builds from
data/v1/sources/worldprediction/processing/build_worldprediction_wm.py --frames N.
Note the split assignment (which sample_uid lands in train vs eval) is NOT
guaranteed identical across frame counts even with the same SPLIT_SEED --
a different frame count can drop a different set of too-short clips (the
min-duration threshold scales with frame count), changing each source's row
list before the seeded shuffle runs. This is an accepted consequence of the
frame-count change, not compensated for.

Row-level (not video-level) split: unlike every other source in this repo's
convention (video/trajectory/clip-level, to avoid train/eval sharing a
source video), this recipe was explicitly requested as a uniform per-source
mix, and does not check for shared source videos between a train row and an
eval row. Caveat, not a bug: a small MCQ-adjacent leakage risk exists if two
rows from the same source video land on opposite sides of the split.

Usage:
    python build_uniform_split.py               # 4 frames (default)
    python build_uniform_split.py --frames 8     # 8 frames
"""
import argparse
import json
import os
import random

WP_DATASETS_DIR = "/scratch/network/ssd2/junlin/ssl_mllm/baseline_pipeline/data/v1/datasets/worldprediction"
OUT_DIR_BASE = "/scratch/network/ssd2/junlin/ssl_mllm/baseline_pipeline/data/v1/datasets/uniform_data_split_regarding_data_source"
SUITE_DIR = "/scratch/network/ssd2/junlin/ssl_mllm/baseline_pipeline/eval/suites/worldprediction"

SPLIT_SEED = 42
EVAL_FRACTION = 0.2  # train:eval = 4:1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=4,
                         help="frame count to build off of (matches build_worldprediction_wm.py's "
                              "--frames). Default 4 reads/writes unsuffixed paths; any other value "
                              "reads wm_{train,eval}_{n}f.jsonl and writes to a _{n}f output dir.")
    args = parser.parse_args()

    suffix = "" if args.frames == 4 else f"_{args.frames}f"
    out_dir = f"{OUT_DIR_BASE}{suffix}"

    rows = []
    for split in ["train", "eval"]:
        fname = f"wm_{split}{suffix}.jsonl"
        with open(os.path.join(WP_DATASETS_DIR, fname)) as f:
            for line in f:
                rows.append(json.loads(line))

    by_source = {}
    for row in rows:
        by_source.setdefault(row["dataset_source"], []).append(row)

    train_rows, eval_rows = [], []
    print(f"{'source':<20} {'total':>6} {'train':>6} {'eval':>6}")
    for source in sorted(by_source):
        source_rows = by_source[source]
        rng = random.Random(SPLIT_SEED)
        rng.shuffle(source_rows)
        n_eval = round(len(source_rows) * EVAL_FRACTION)
        eval_rows.extend(source_rows[:n_eval])
        train_rows.extend(source_rows[n_eval:])
        print(f"{source:<20} {len(source_rows):>6} {len(source_rows)-n_eval:>6} {n_eval:>6}")

    print(f"{'TOTAL':<20} {len(rows):>6} {len(train_rows):>6} {len(eval_rows):>6}")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "train.jsonl"), "w") as f:
        for row in train_rows:
            f.write(json.dumps(row) + "\n")
    with open(os.path.join(out_dir, "eval.jsonl"), "w") as f:
        for row in eval_rows:
            f.write(json.dumps(row) + "\n")
    print(f"\nWrote {len(train_rows)} rows -> {out_dir}/train.jsonl")
    print(f"Wrote {len(eval_rows)} rows -> {out_dir}/eval.jsonl")

    # Build the WorldPrediction-annotation-format filtered eval file (needed
    # by the real eval harness, run.py/evaluator.py -- that code reads the
    # original states/candidates/ground_truth annotation schema, not our
    # flat training jsonl).
    wm_full = json.load(open(f"{SUITE_DIR}/data/WorldPrediction-WM.json"))
    eval_uids_by_source = {}
    for row in eval_rows:
        eval_uids_by_source.setdefault(row["dataset_source"], set()).add(row["sample_uid"])

    filtered = {}
    for source, uids in eval_uids_by_source.items():
        # sorted(): set iteration order depends on Python's per-process hash
        # randomization, which would otherwise make this dict's (and thus
        # the dumped JSON's) key order non-reproducible across runs even
        # with an identical SPLIT_SEED and identical input data.
        filtered[source] = {uid: wm_full[source][uid] for uid in sorted(uids) if uid in wm_full[source]}
        missing = uids - set(filtered[source].keys())
        if missing:
            print(f"WARNING: {len(missing)} eval sample_uids for {source} not found in WorldPrediction-WM.json")

    eval_ann_path = os.path.join(out_dir, "WorldPrediction-WM-uniform_eval.json")
    json.dump(filtered, open(eval_ann_path, "w"))
    print(f"Wrote eval-harness annotation file ({sum(len(v) for v in filtered.values())} samples) -> {eval_ann_path}")


if __name__ == "__main__":
    main()
