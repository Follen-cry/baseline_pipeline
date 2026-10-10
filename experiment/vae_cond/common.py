"""Config loader shared by the vae_cond scripts. `python3 common.py <dotted.key>` prints one value."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "config.json")))
CONDS = ["none", "first", "all"]
SIZES = CFG["sizes"]  # output size = frame size = condition size, "WxH"


def size_tag(size):
    """Output-folder suffix per size. 672x384 (the first run's area-512 rule) keeps the original untagged folders."""
    return "" if size == "672x384" else "_" + size


def path(key):
    p = CFG[key]
    return p if os.path.isabs(p) else os.path.join(HERE, p)


def datapoints():
    """Rows of datapoints.jsonl with image paths resolved against this folder (they are stored relative, copies under data/images)."""
    rows = [json.loads(l) for l in open(path("datapoints")) if l.strip()]
    for r in rows:
        r["frames"] = [os.path.join(HERE, p) for p in r["frames"]]
        r["target_image"] = os.path.join(HERE, r["target_image"])
    return rows


if __name__ == "__main__":
    k = sys.argv[1]
    if k in ("datapoints", "outputs_root"):
        print(path(k))
    else:
        v = CFG
        for part in k.split("."):
            v = v[part]
        print(v)
