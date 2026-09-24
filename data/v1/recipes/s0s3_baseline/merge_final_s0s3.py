"""Merge VBVR + the 4 real-world sources (EPIC-Kitchen, NWM, panda70m_epic_ssl,
panda70m_ssl-v1) into the final 4 S0-S3 training datasets.

Source files per setting (S0/S1/S2 = train only, no eval holdout by project
convention; S3 = train + eval):

  VBVR:              data/datasets/vbvr_next_frame/vbvr_{S0,S1,S2,S3}_{train,eval}.jsonl
  EPIC-Kitchen:       data/datasets/epic_ssl/epic_gen_noaction/..., epic_gen_S1/...,
                      epic_gen_action/..., epic_dual_action/...
  NWM/RECON:          data/datasets/nwm_ssl/nwm_gen_noaction/..., nwm_gen_S1/...,
                      nwm_gen_action/..., nwm_dual_action/...
  panda70m_epic_ssl:  data/datasets/panda70m_epic_ssl/panda70m_gen_noaction/...,
                      panda70m_gen_S1/..., panda70m_gen_S2/...,
                      panda70m_dual_noaction/...
  panda70m_ssl v1:    data/datasets/panda70m_ssl_v1/panda70m_v1_gen_{S0,S1,S2}/...,
                      panda3dsr_dual_noaction/...

Usage:
    python data/scripts/merge_final_s0s3.py
"""
import json
from pathlib import Path

ROOT = Path("/scratch/network/ssd2/junlin/ssl_mllm/data")
DATASETS = ROOT / "datasets"
OUT_DIR = DATASETS / "final_s0s3"
META_DIR = ROOT / "meta"

SOURCES = {
    "S0": [
        ("vbvr", DATASETS / "vbvr_next_frame/vbvr_S0_train_nocap.jsonl"),
        ("epic", DATASETS / "epic_ssl/epic_gen_noaction/epic_gen_noaction_train.jsonl"),
        ("nwm", DATASETS / "nwm_ssl/nwm_gen_noaction/nwm_gen_noaction_train.jsonl"),
        ("panda70m_epic", DATASETS / "panda70m_epic_ssl/panda70m_gen_noaction/panda70m_gen_noaction_train.jsonl"),
        ("panda70m_v1", DATASETS / "panda70m_ssl_v1/panda70m_v1_gen_S0/panda70m_v1_gen_S0_train.jsonl"),
    ],
    "S1": [
        ("vbvr", DATASETS / "vbvr_next_frame/vbvr_S1_train_mixed_nocap.jsonl"),
        ("epic", DATASETS / "epic_ssl/epic_gen_S1/epic_gen_S1_train.jsonl"),
        ("nwm", DATASETS / "nwm_ssl/nwm_gen_S1/nwm_gen_S1_train.jsonl"),
        ("panda70m_epic", DATASETS / "panda70m_epic_ssl/panda70m_gen_S1/panda70m_gen_S1_train.jsonl"),
        ("panda70m_v1", DATASETS / "panda70m_ssl_v1/panda70m_v1_gen_S1/panda70m_v1_gen_S1_train.jsonl"),
    ],
    "S2": [
        ("vbvr", DATASETS / "vbvr_next_frame/vbvr_S2_train.jsonl"),
        ("epic", DATASETS / "epic_ssl/epic_gen_action/epic_gen_action_train.jsonl"),
        ("nwm", DATASETS / "nwm_ssl/nwm_gen_action/nwm_gen_action_train.jsonl"),
        ("panda70m_epic", DATASETS / "panda70m_epic_ssl/panda70m_gen_S2/panda70m_gen_S2_train.jsonl"),
        ("panda70m_v1", DATASETS / "panda70m_ssl_v1/panda70m_v1_gen_S2/panda70m_v1_gen_S2_train.jsonl"),
    ],
}

S3_TRAIN = [
    ("vbvr", DATASETS / "vbvr_next_frame/vbvr_S3_train.jsonl"),
    ("epic", DATASETS / "epic_ssl/epic_dual_action/epic_dual_action_train.jsonl"),
    ("nwm", DATASETS / "nwm_ssl/nwm_dual_action/nwm_dual_action_train.jsonl"),
    ("panda70m_epic", DATASETS / "panda70m_epic_ssl/panda70m_dual_noaction/panda70m_dual_noaction_train.jsonl"),
    ("panda70m_v1", DATASETS / "panda70m_ssl_v1/panda3dsr_dual_noaction/panda3dsr_dual_noaction_train.jsonl"),
]
S3_EVAL = [
    ("vbvr", DATASETS / "vbvr_next_frame/vbvr_S3_eval.jsonl"),
    ("epic", DATASETS / "epic_ssl/epic_dual_action/epic_dual_action_eval.jsonl"),
    ("nwm", DATASETS / "nwm_ssl/nwm_dual_action/nwm_dual_action_eval.jsonl"),
    ("panda70m_epic", DATASETS / "panda70m_epic_ssl/panda70m_dual_noaction/panda70m_dual_noaction_eval.jsonl"),
    ("panda70m_v1", DATASETS / "panda70m_ssl_v1/panda3dsr_dual_noaction/panda3dsr_dual_noaction_eval.jsonl"),
]


def load_and_tag(path, source_name):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d["source"] = source_name
            rows.append(d)
    return rows


def write_merged(rows, out_path, meta_key, max_dynamic_patch, task_type="imgen"):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for i, r in enumerate(rows):
            r["orig_id"] = r.get("id")
            r["id"] = f"{meta_key}_{i:06d}"
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    META_DIR.mkdir(parents=True, exist_ok=True)
    meta = {meta_key: {
        "root": "/", "annotation": str(out_path), "data_augment": False,
        "max_dynamic_patch": max_dynamic_patch, "repeat_time": 1,
        "length": len(rows), "task_type": task_type,
    }}
    json.dump(meta, open(META_DIR / f"{meta_key}_meta.json", "w"), indent=2)
    return len(rows)


def main():
    summary = {}

    for setting, sources in SOURCES.items():
        rows = []
        counts = {}
        for name, path in sources:
            src_rows = load_and_tag(path, name)
            rows.extend(src_rows)
            counts[name] = len(src_rows)
        n = write_merged(rows, OUT_DIR / f"{setting}_train.jsonl", f"final_{setting}", max_dynamic_patch=3)
        summary[setting] = {"total": n, "by_source": counts}
        print(f"[{setting}] {n} rows ({counts}) -> {OUT_DIR}/{setting}_train.jsonl")

    # S3: train + eval, is_option rows need max_dynamic_patch = 3 ctx + 4 options = 7
    train_rows, train_counts = [], {}
    for name, path in S3_TRAIN:
        r = load_and_tag(path, name)
        train_rows.extend(r)
        train_counts[name] = len(r)
    n_train = write_merged(train_rows, OUT_DIR / "S3_train.jsonl", "final_S3", max_dynamic_patch=7)

    eval_rows, eval_counts = [], {}
    for name, path in S3_EVAL:
        r = load_and_tag(path, name)
        eval_rows.extend(r)
        eval_counts[name] = len(r)
    n_eval = write_merged(eval_rows, OUT_DIR / "S3_eval.jsonl", "final_S3_eval", max_dynamic_patch=7)

    summary["S3"] = {"train": n_train, "train_by_source": train_counts,
                      "eval": n_eval, "eval_by_source": eval_counts}
    print(f"[S3] train {n_train} ({train_counts}) + eval {n_eval} ({eval_counts})")

    print("\n=== FINAL SUMMARY ===")
    for setting in ("S0", "S1", "S2", "S3"):
        print(setting, summary[setting])


if __name__ == "__main__":
    main()
