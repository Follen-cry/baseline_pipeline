"""Sample-triple lookup shared by the calibration/held-out harnesses.

Maps an eval record id -> (input_path, candidate_path, ground_truth_path) for
any of the 5 model variants. No scoring logic lives here.
"""
import json
import os

ROOT = "/scratch/network/ssd2/junlin/ssl_mllm"
JSONLS = [
    f"{ROOT}/data/datasets/vbvr_target_pred_id/target_pred_id_eval.jsonl",
    f"{ROOT}/data/datasets/vbvr_target_pred_ood/target_pred_ood_eval.jsonl",
]
RESULTS = f"{ROOT}/results/vbvr_target_pred_eval"
VARIANTS = ["base_4task500sft", "s0_4task500sft", "s1_4task500sft",
            "s2_4task500sft", "s3_4task500sft"]
TASKS = ["multi_object_placement", "rotation_puzzle",
         "shape_color_then_move", "2d_geometric_transformation"]


def load_index():
    idx = {}
    for p in JSONLS:
        for line in open(p):
            r = json.loads(line)
            idx[r["id"]] = r
    return idx


_IDX = None


def index():
    global _IDX
    if _IDX is None:
        _IDX = load_index()
    return _IDX


def triple(rec_id, variant):
    r = index()[rec_id]
    cand = f"{RESULTS}/{variant}/{r['task_name']}/{rec_id}.png"
    return r["image"][0], cand, r["target_image"]


def ids_for(task):
    return sorted(k for k, v in index().items() if v["task_name"] == task)


def judge_scores(variant):
    p = f"{RESULTS}/{variant}/judge_scored.json"
    if not os.path.exists(p):
        return {}
    return json.load(open(p)).get("results", {})


def old_scores(variant):
    p = f"{RESULTS}/{variant}/scored.json"
    if not os.path.exists(p):
        return {}
    return {r["id"]: r for r in json.load(open(p))["records"]}
