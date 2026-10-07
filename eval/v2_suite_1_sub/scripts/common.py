"""Shared config loading for the v2_suite_1_sub scripts. Every path comes from ../config.json."""
import json
import os
import sys

SUITE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CFG = json.load(open(os.path.join(SUITE, "config.json")))

# benchmark -> (primary per-item metric, lo, hi): used for 0-1 normalisation and the overall column
PRIMARY = {"phyeditbench": ("overall", 1, 10), "picabench": ("acc", 0, 1),
           "risebench": ("score", 1, 5), "imgedit_basic": ("score", 1, 5)}
BENCHES = list(PRIMARY)
# official PhyEditBench weighting of its 4 judged dimensions
PHY_W = {"consistency": 0.2, "instruction_following": 0.3, "physical_plausibility": 0.4, "image_quality": 0.1}


def path(p):
    """Config path -> absolute (relative paths are relative to the suite folder)."""
    return p if os.path.isabs(p) else os.path.join(SUITE, p)


MANIFEST = path(CFG["manifest"])
OUT = path(os.environ.get("SUITE_OUT", CFG["outputs_root"]))
RES = path(CFG["results_root"])
JUDGE_RAW = os.path.join(RES, "judge_raw", CFG["judge"]["judge_name"])


def manifest():
    return [json.loads(l) for l in open(MANIFEST)]


def systems():
    """Models (config.models) + references (config.references), in report order."""
    return list(CFG["models"]) + list(CFG["references"])


if __name__ == "__main__":
    # tiny CLI so shell scripts can read config values:  python common.py envs.internvlu_python
    v = CFG
    for k in sys.argv[1].split("."):
        v = v[k]
    print(path(v) if k.endswith(("root", "manifest")) else v)
