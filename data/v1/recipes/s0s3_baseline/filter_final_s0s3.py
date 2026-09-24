"""Drop any row in data/v1/datasets/final_s0s3/*.jsonl that references a missing image
file (found: ~2.7% of panda70m_epic_ssl rows, from cv2 seek failures near
clip boundaries during frame extraction -- see the "[frames] WARNING: failed
to read ..." lines from build_panda70m_epic_ssl.py --stage anchors). Rewrites
each file in place (after a .bak backup) and updates its meta.json length.
"""
import json
import os
import shutil
from pathlib import Path

OUT_DIR = Path("/scratch/network/ssd2/junlin/ssl_mllm/data/datasets/final_s0s3")
META_DIR = Path("/scratch/network/ssd2/junlin/ssl_mllm/data/meta")

FILES = {
    "S0_train.jsonl": "final_S0",
    "S1_train.jsonl": "final_S1",
    "S2_train.jsonl": "final_S2",
    "S3_train.jsonl": "final_S3",
    "S3_eval.jsonl": "final_S3_eval",
}


def row_paths(d):
    paths = list(d.get("image", []))
    for k in ("target_image", "cond_image"):
        if k in d:
            paths.append(d[k])
    return paths


def main():
    for fname, meta_key in FILES.items():
        path = OUT_DIR / fname
        bak = path.with_suffix(path.suffix + ".bak")
        # Always snapshot the CURRENT file before filtering -- this used to
        # only copy "if not bak.exists()", which meant re-running this script
        # after re-running merge_final_s0s3.py (e.g. after a source pipeline
        # fix) silently filtered the OLD stale .bak instead of the fresh
        # merge, discarding the fix with no error. Filtering must always be
        # idempotent against whatever's in `path` right now.
        shutil.copy(path, bak)

        kept, dropped, dropped_by_source = [], 0, {}
        with open(bak) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if all(os.path.exists(p) for p in row_paths(d)):
                    kept.append(d)
                else:
                    dropped += 1
                    src = d.get("source", "?")
                    dropped_by_source[src] = dropped_by_source.get(src, 0) + 1

        with open(path, "w") as f:
            for i, d in enumerate(kept):
                d["id"] = f"{meta_key}_{i:06d}"
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

        meta_path = META_DIR / f"{meta_key}_meta.json"
        if meta_path.exists():
            meta = json.load(open(meta_path))
            meta[meta_key]["length"] = len(kept)
            json.dump(meta, open(meta_path, "w"), indent=2)

        print(f"{fname}: kept {len(kept)}, dropped {dropped} {dropped_by_source or ''}")


if __name__ == "__main__":
    main()
