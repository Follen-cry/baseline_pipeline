"""Config loader shared by the vae_cond scripts. `python3 common.py <dotted.key>` prints one value."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "config.json")))


def path(key):
    p = CFG[key]
    return p if os.path.isabs(p) else os.path.join(HERE, p)


def datapoints():
    return [json.loads(l) for l in open(path("datapoints")) if l.strip()]


if __name__ == "__main__":
    k = sys.argv[1]
    if k in ("datapoints", "outputs_root"):
        print(path(k))
    else:
        v = CFG
        for part in k.split("."):
            v = v[part]
        print(v)
