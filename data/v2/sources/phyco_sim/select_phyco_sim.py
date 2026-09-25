#!/usr/bin/env python3
"""Select a stratified 10K oversample from the extracted PhyCo-Sim dataset (v2 source).

Run after `download_phyco_sim.py --extract`. The raw dir holds 127,440 samples across 9 scenes;
each sample dir has rgba.mp4 (98 frames @ 24 fps, 768x432), depth.mp4, segmentation.mp4,
metadata.json, animation_data.pkl.

Why a motion filter: the renderer stops simulating once objects settle
(metadata.rendering_efficiency.settle_frame) and repeats the last frame for the rest of the
98 frames, so the useful motion can be much shorter than 4 s (median 11 frames for
ball_drop_soft_v4). A sample must keep moving for >= MIN_MOTION_FRAMES (enough for one
v2 window at Δt ≈ 0.25 s: 3Δt = 18 frames). settle_frame None (mode efficient_no_settle /
traditional) means it moves for the whole clip.

Selection (seed 42): drop *_test shards; drop samples with motion < MIN_MOTION_FRAMES;
equal quota per scene (10000 / 9); within a scene, stratify by that scene's physical
parameter (quantile bins for continuous, categories for discrete) with equal share per
bin, topping up from other bins when one runs short.

Output: <RAW_ROOT>/selection_10k.jsonl
"""
import glob, json, os, random, sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

RAW_ROOT = "/scratch/network/ssd/junlin/raw/phyco_sim"
N_TOTAL = 10_000
SEED = 42
N_FRAMES = 98
FPS = 24
MIN_MOTION_FRAMES = 18
N_QBINS = 5

# scene -> (dotted metadata key to stratify by, "cont" | "cat"); None = no varying parameter
STRATIFY = {
    "ball_drop_soft_v4": None,
    "ball_drop_v2": ("ball_restitution", "cont"),
    "ball_drop_v3": ("num_balls", "cat"),
    "ball_wall_collision": ("ball_restitution", "cont"),
    "cube_deform_soft_v2_noeff": ("jelly_texture", "cat"),
    "friction_slide_flat_force_v3": ("force_calculation.force_magnitude", "cont"),
    "friction_slide_flat_v2": ("platform_friction", "cont"),
    "jenga_force": ("velocity_profile.trajectory_description", "cat"),
    "pool_table_force": ("force_magnitude", "cont"),
}


def get(d, dotted):
    for k in dotted.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def read(meta_path, scene):
    m = json.load(open(meta_path))
    sd = os.path.dirname(meta_path)
    shard = os.path.relpath(sd, os.path.join(RAW_ROOT, scene)).split("/")[0]
    settle = get(m, "rendering_efficiency.settle_frame")
    motion = N_FRAMES if settle is None else int(settle)
    strat = STRATIFY[scene]
    return dict(
        id=f"phyco_sim__{scene}__{shard}__{os.path.basename(sd)}", source="phyco_sim", scene=scene,
        shard=shard, is_test_shard=shard.endswith("_test"), sample_dir=sd,
        video=os.path.join(sd, "rgba.mp4"), fps=FPS, n_frames=N_FRAMES,
        settle_frame=settle, motion_frames=motion, motion_s=round(motion / FPS, 3),
        strat_key=strat[0] if strat else None, strat_value=get(m, strat[0]) if strat else None,
        simulation_type=m.get("simulation_type"))


def bin_rows(rows, strat):
    """group rows into strata; also stamps each row's `strat_bin`"""
    if strat is None:
        label = lambda r: "all"
    elif strat[1] == "cat":
        label = lambda r: str(r["strat_value"])
    else:
        vals = sorted(r["strat_value"] for r in rows if r["strat_value"] is not None)
        edges = [vals[int(len(vals) * q / N_QBINS)] for q in range(1, N_QBINS)]
        label = lambda r: "none" if r["strat_value"] is None else f"q{sum(r['strat_value'] >= e for e in edges)}"
    b = defaultdict(list)
    for r in rows:
        r["strat_bin"] = label(r)
        b[r["strat_bin"]].append(r)
    return b


def take_balanced(bins, k, rng):
    for v in bins.values():
        rng.shuffle(v)
    out, keys = [], sorted(bins)
    while len(out) < k and any(bins[x] for x in keys):
        for x in keys:
            if bins[x] and len(out) < k:
                out.append(bins[x].pop())
    return out


def main():
    rng = random.Random(SEED)
    scenes = sorted(STRATIFY)
    quota = {s: N_TOTAL // len(scenes) + (i < N_TOTAL % len(scenes)) for i, s in enumerate(scenes)}
    selected, report = [], []
    for s in scenes:
        metas = sorted(m for m in glob.glob(os.path.join(RAW_ROOT, s, "**", "metadata.json"), recursive=True)
                       if os.path.exists(os.path.join(os.path.dirname(m), "rgba.mp4")))
        with ThreadPoolExecutor(16) as ex:
            rows = list(ex.map(lambda p: read(p, s), metas))
        pool = [r for r in rows if not r["is_test_shard"] and r["motion_frames"] >= MIN_MOTION_FRAMES]
        bins = bin_rows(pool, STRATIFY[s])
        bin_sizes = {k: len(v) for k, v in sorted(bins.items())}
        pick = take_balanced(bins, quota[s], rng)
        selected += pick
        report.append((s, len(rows), len(pool), len(pick), bin_sizes))
        print(f"[{s}] samples={len(rows)} eligible={len(pool)} picked={len(pick)}/{quota[s]} bins={bin_sizes}", flush=True)

    caps = {}
    for s in scenes:
        p = os.path.join(RAW_ROOT, s, "common_caption_cosmos.txt")
        caps[s] = open(p).read().strip() if os.path.exists(p) else None
    for r in selected:
        r["caption_scene"] = caps[r["scene"]]
    man = os.path.join(RAW_ROOT, "selection_10k.jsonl")
    with open(man, "w") as f:
        for r in selected:
            f.write(json.dumps(r) + "\n")
    mot = sorted(r["motion_frames"] for r in selected)
    print(f"[done] {len(selected)} selected -> {man}; motion frames p10/p50/p90 = "
          f"{mot[len(mot)//10]}/{mot[len(mot)//2]}/{mot[9*len(mot)//10]}; "
          f"scenes without caption: {[s for s, c in caps.items() if not c]}", flush=True)
    return 0 if len(selected) == N_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
