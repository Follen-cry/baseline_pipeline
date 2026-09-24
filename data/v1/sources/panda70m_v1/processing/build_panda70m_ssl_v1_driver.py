"""Driver that runs build_panda70m_ssl.py's pipeline (already at the shared
real-video-mix grid: SAMPLE_FPS=2.0, CONTEXT_SIZE=3, HORIZON=1, STRIDE=4)
against the ORIGINAL v1 clip pool (1,073 COCO/3DSRBench-filtered clips,
already downloaded to /scratch/local/ssd/junlin/data/datasets/panda70m/3dsrbench_sample5000/
in video2dataset's sharded format) instead of the v2 pool (blocked download,
/scratch/local/ssd/junlin/data/datasets/panda70m/ytdlp_merged/).

Kept as a separate driver rather than editing build_panda70m_ssl.py's own
OUT/LOCAL_CLIP_ROOT constants, so the v2 pipeline config stays untouched and
resumable once the YouTube rate-limit block clears -- this v1 pool needs no
download at all, so there's no reason to route it through the same paths.

The v1 clip pool already has a valid clip_manifest.json (built by an earlier,
pre-rework version of build_panda70m_ssl.py, which read the sharded
video2dataset layout directly) at data/datasets/panda70m_ssl/clip_manifest.json --
reused as-is; build_clip_manifest() is never called here (it now expects the
NEW flat yt-dlp sidecar format and can't read the old sharded layout).

Usage:
    python data/scripts/build_panda70m_ssl_v1_driver.py --stage splits
    python data/scripts/build_panda70m_ssl_v1_driver.py --stage anchors
    python data/scripts/build_panda70m_ssl_v1_driver.py --stage ffs
    python data/scripts/build_panda70m_ssl_v1_driver.py --stage dual
    python data/scripts/build_panda70m_ssl_v1_driver.py --stage gen --direction forward --setting-label S0 --out-stem panda70m_v1_gen_S0
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import build_panda70m_ssl as m

# v1-specific paths -- distinct from both the legacy CONTEXT_SIZE=4 build
# (data/datasets/panda70m_ssl/) and the blocked v2 pool (data/datasets/panda70m_ssl_v2/).
V1_OUT = "/scratch/network/ssd2/junlin/ssl_mllm/data/datasets/panda70m_ssl_v1"
V1_CLIP_ROOT = "/scratch/local/ssd/junlin/data/datasets/panda70m/3dsrbench_sample5000"
V1_FRAMES_ROOT = "/scratch/network/ssd/junlin/panda70m_ssl_v1_frames"
EXISTING_MANIFEST = "/scratch/network/ssd2/junlin/ssl_mllm/data/datasets/panda70m_ssl/clip_manifest.json"

m.OUT = V1_OUT
m.LOCAL_CLIP_ROOT = V1_CLIP_ROOT
m.FRAMES_ROOT = V1_FRAMES_ROOT
m.CLIP_MANIFEST = f"{V1_OUT}/clip_manifest.json"

# target: full ceiling (7016 measured at this grid), no eval carve-out needed
# for gen (S0/S1/S2 take none, per project convention) -- EVAL_HOLDOUT_CLIPS
# still reserves clips for S3's MCQ eval pool specifically.
m.EVAL_HOLDOUT_CLIPS = 130   # ~12% of 1068 valid clips, sized for a few hundred eval anchors
m.TARGET_TRAIN = 7016
m.TARGET_EVAL = 200
m.PER_CLIP_CAP_TRAIN = 10    # loose -- ceiling check showed cap barely binds at this scale
m.PER_CLIP_CAP_EVAL = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["splits", "anchors", "ffs", "gen", "dual"])
    ap.add_argument("--direction", choices=["forward", "backward", "mixed"], default="forward")
    ap.add_argument("--with-caption", action="store_true")
    ap.add_argument("--setting-label", type=str, default=None)
    ap.add_argument("--out-stem", type=str, default="panda3dsr_gen_noaction")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Path(V1_OUT).mkdir(parents=True, exist_ok=True)
    dst_manifest = Path(m.CLIP_MANIFEST)
    if not dst_manifest.exists():
        import shutil
        shutil.copy(EXISTING_MANIFEST, dst_manifest)
        print(f"copied existing manifest ({EXISTING_MANIFEST}) -> {dst_manifest}")

    if args.stage == "splits":
        m.build_clip_splits()
    elif args.stage == "anchors":
        m.build_anchors()
    elif args.stage == "ffs":
        m.build_ffs()
    elif args.stage == "gen":
        m.build_gen(direction=args.direction, has_caption=args.with_caption,
                    setting_label=args.setting_label, out_stem=args.out_stem, seed=args.seed)
    elif args.stage == "dual":
        m.build_dual()


if __name__ == "__main__":
    main()
