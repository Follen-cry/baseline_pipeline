#!/usr/bin/env python3
"""Select physics-process clips from Panda-70M's training_2m split by caption (v2 source).

Pool: panda70m_training_2m.csv (800K videos / 2.4M clips; a subset of the official 10M
quality tier: matching_score > 0.43, <=3 clips/video). Metadata only; no frames are
needed until download (download_panda70m.py).

Stages (run in order; each writes its own file under <OUT>/_filter/, so later stages are
re-runnable on their own):

  prefilter -- mechanical + cheap text gates over every clip:
                 * tier A: desirable_filtering == "desirable" and longest single shot
                   (TransNetV2 shot_boundary_detection) >= MIN_SHOT_S (4 s).
                 * tier B (fallback, --tier B): desirable with a 3-4 s longest shot, or
                   "2_tiny_camera_movement" (a near-static camera, which is fine for physics)
                   with a >= 3 s shot. Judged and downloaded only after tier A is used up.
                 * The download segment is the longest shot, cut to at most MAX_SEG_S, so no
                   v2 4-frame window can straddle a cut.
                 * caption hits PHYS_RE (a physics-process word) and misses HARD_REJECT_RE
                   (screens, games, animation, talking heads).
               -> stage1_candidates.jsonl (tier A) / stage1_candidates_tierB.jsonl
  judge     -- Qwen3-VL-30B-A3B-Instruct-FP8 (vLLM, OpenAI API, text-only) labels each
               candidate caption with JUDGE_SYSTEM_PROMPT. Candidates are visited in a
               seeded random order and the stage stops once --target-pass captions have
               passed, so the passed set is an unbiased sample of the full candidate pool.
               Resumable (skips clip_keys already judged).
               -> stage2_judged.jsonl / stage2_judged_tierB.jsonl (appended incrementally)
  export    -- passed = physical_process AND motion_rich AND NOT manipulation_focused
               AND confidence >= MIN_CONF.
               Both tiers are merged, each row tagged with `tier`.
               -> <OUT>/pool_passed.jsonl (the download pool, one row per clip)

Usage (run on a node that can reach the vLLM server):
    python filter_panda70m.py --stage prefilter
    python filter_panda70m.py --stage judge --limit 200 --out-name pilot_judged.jsonl  # prompt pilot
    python filter_panda70m.py --stage judge --target-pass 12000 --workers 8
    python filter_panda70m.py --stage prefilter --tier B
    python filter_panda70m.py --stage judge --tier B --target-pass 1000000
    python filter_panda70m.py --stage export
"""
import argparse, ast, csv, json, os, random, re, sys, threading, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

METADATA_CSV = ("/scratch/network/ssd2/junlin/ssl_mllm/data/datasets/panda70m/metadata/"
                "panda70m_training_2m.csv")
OUT = "/scratch/network/ssd/junlin/raw/panda70m"
FILTER_DIR = os.path.join(OUT, "_filter")
STAGE1 = {"A": os.path.join(FILTER_DIR, "stage1_candidates.jsonl"),
          "B": os.path.join(FILTER_DIR, "stage1_candidates_tierB.jsonl")}
STAGE2 = {"A": os.path.join(FILTER_DIR, "stage2_judged.jsonl"),
          "B": os.path.join(FILTER_DIR, "stage2_judged_tierB.jsonl")}
POOL = os.path.join(OUT, "pool_passed.jsonl")

SEED = 42
MIN_SHOT_S = 4.0   # tier A; v2 needs 3 gaps, so 4 s covers dt <= 1.0 s with margin
MIN_SHOT_S_B = 3.0  # tier B floor: exactly 3 gaps of 1.0 s
MAX_SEG_S = 20.0   # cap per-clip download length
MIN_CONF = 4

VLLM_BASE = os.environ.get("VLLM_BASE", "http://127.0.0.1:8010/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "qwen3-vl-30b-fp8")

# ---------------------------------------------------------------------------
# prefilter regexes: deliberately loose (recall over precision); the LLM decides.
# ---------------------------------------------------------------------------
PHYS_RE = re.compile(r"\b(" + "|".join([
    r"fall(s|ing|en)?", r"drop(s|ped|ping)?", r"tumbl\w*", r"topple\w*", r"collaps\w*",
    r"avalanche\w*", r"landslide\w*", r"roll(s|ing|ed)?", r"slid(e|es|ing)", r"skid\w*",
    r"bounc\w*", r"collid\w*", r"collision\w*", r"crash(es|ed|ing)?", r"smash\w*",
    r"impact\w*", r"hit(s|ting)? the (ground|water|wall|floor)",
    r"splash\w*", r"pour(s|ed|ing)?", r"flow(s|ing|ed)?", r"stream(s|ing)?", r"waterfall\w*",
    r"wave(s)?", r"surf\w*", r"ripple\w*", r"drip\w*", r"spray\w*", r"spill\w*", r"gush\w*",
    r"fountain\w*", r"rain(s|ing)?", r"flood\w*", r"rapids", r"river", r"current",
    r"boil\w*", r"bubbl\w*", r"foam\w*", r"melt\w*", r"freez\w*", r"ice", r"lava",
    r"smoke", r"smoking", r"steam\w*", r"fire", r"flames?", r"burn(s|ing|ed)?", r"blaze\w*",
    r"explo\w*", r"erupt\w*", r"spark\w*", r"firework\w*", r"detonat\w*",
    r"shatter\w*", r"break(s|ing)?", r"broke(n)?", r"crack\w*", r"burst\w*", r"pop(s|ping)?",
    r"crush\w*", r"deform\w*", r"bend(s|ing)?", r"stretch\w*",
    r"spin(s|ning)?", r"rotat\w*", r"swing(s|ing)?", r"pendulum", r"oscillat\w*", r"wobbl\w*",
    r"float(s|ing)?", r"sink(s|ing)?", r"launch\w*", r"projectile\w*", r"thrown", r"flung",
    r"fly(ing)? (through|across|into) the air", r"in the air", r"domino\w*",
    r"wind", r"windy", r"blow(s|ing|n)?", r"sway\w*", r"flutter\w*", r"flap\w*",
    r"dust", r"sand", r"mud", r"liquid", r"water", r"snow(ing|fall)?", r"hail",
    r"balloon\w*", r"magnet\w*", r"gravity", r"physics", r"experiment",
]) + r")\b", re.I)

HARD_REJECT_RE = re.compile(r"\b(" + "|".join([
    r"video game", r"gameplay", r"minecraft", r"fortnite", r"roblox", r"screenshot",
    r"screen recording", r"computer screen", r"animated", r"animation", r"cartoon",
    r"3d render\w*", r"logo", r"slideshow", r"news anchor", r"interview\w*",
]) + r")\b", re.I)


def _sec(t: str) -> float:
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _fmt(sec: float) -> str:
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:06.3f}"


def _in_tier(tier: str, des: str, shot_s: float) -> bool:
    if tier == "A":
        return des == "desirable" and shot_s >= MIN_SHOT_S
    return ((des == "desirable" and MIN_SHOT_S_B <= shot_s < MIN_SHOT_S)
            or (des == "2_tiny_camera_movement" and shot_s >= MIN_SHOT_S_B))


def stage_prefilter(tier: str):
    os.makedirs(FILTER_DIR, exist_ok=True)
    csv.field_size_limit(sys.maxsize)
    c = Counter()
    with open(METADATA_CSV) as f, open(STAGE1[tier], "w") as out:
        for row in csv.DictReader(f):
            ts = ast.literal_eval(row["timestamp"])
            caps = ast.literal_eval(row["caption"])
            ms = ast.literal_eval(row["matching_score"])
            des = ast.literal_eval(row["desirable_filtering"])
            sbd = ast.literal_eval(row["shot_boundary_detection"])
            for i, (s, e) in enumerate(ts):
                c["clips"] += 1
                c[des[i]] += 1
                # longest continuous shot; SBD intervals are relative to the clip start
                raw = sbd[i]
                if raw and isinstance(raw[0], str):  # a few rows store one flat [a, b] pair
                    raw = [raw]
                shots = [(_sec(p[0]), _sec(p[1])) for p in raw if len(p) == 2]
                if not shots:
                    continue
                a, b = max(shots, key=lambda x: x[1] - x[0])
                if not _in_tier(tier, des[i], b - a):
                    continue
                c["tier_ok"] += 1
                cap = caps[i]
                if HARD_REJECT_RE.search(cap):
                    c["hard_reject"] += 1
                    continue
                if not PHYS_RE.search(cap):
                    continue
                c["candidates"] += 1
                s0 = _sec(s)
                seg = (s0 + a, s0 + min(b, a + MAX_SEG_S))
                out.write(json.dumps({
                    "clip_key": f"{row['videoID']}_{i}", "tier": tier,
                    "desirable_filtering": des[i], "videoID": row["videoID"],
                    "clip_idx": i, "url": row["url"], "timestamp": [s, e],
                    "segment": [_fmt(seg[0]), _fmt(seg[1])],
                    "segment_dur_s": round(seg[1] - seg[0], 3),
                    "caption": cap, "matching_score": ms[i],
                }) + "\n")
            if c["clips"] % 500_000 < 3:
                print(f"[prefilter] {dict(c)}", flush=True)
    print(f"[prefilter] done: {dict(c)} -> {STAGE1[tier]}")


# ---------------------------------------------------------------------------
# judge
# ---------------------------------------------------------------------------
CATEGORIES = ["falling", "rolling_sliding", "collision_impact", "fluid_flow", "splash_pour",
              "fire_smoke_explosion", "deformation_breaking", "projectile",
              "rotation_oscillation", "wind_driven", "other_physics", "none"]

JUDGE_SYSTEM_PROMPT = f"""You label captions of short YouTube video clips (4-20 s). The goal is \
a training set for learning PHYSICAL DYNAMICS: clips where objects or materials visibly move or \
change under physical laws, and that motion is what the clip is about.

You see ONE auto-generated caption. Judge only what the caption says; do not imagine unstated \
content.

Fields:
- "physical_process": true if the caption's main content is objects/materials moving or \
changing physically: gravity (falling, dropping, tumbling, collapsing, avalanches), rolling or \
sliding, collisions, bouncing, crashes and impacts, fluids (water flowing, waves, waterfalls, \
rapids, splashing, dripping, rain, fountains, boiling, bubbling), smoke / steam / fire / \
explosions / sparks / eruptions, deformation (breaking, shattering, cracking, bursting, melting, \
burning, crushing), objects flying through the air, spinning / swinging / oscillating objects, \
wind-driven motion (trees swaying, flags flapping, leaves or dust blowing), floating / sinking.
- "motion_rich": true if the described process involves clearly visible, substantial motion or \
change during the clip (not a still or nearly still scene; "a calm lake" or "a candle on a table" \
is false).
- "manipulation_focused": true if the caption is mainly about a person (or hands) handling, \
using, making or operating something: cutting, cooking, stirring, mixing, assembling, crafting, \
painting, cleaning, holding up or showing an object, using a tool, operating a machine, pouring \
drinks as a serving action. Rule of thumb: if the grammatical subject is a person acting ON an \
object, it is manipulation; if the subject is the object/material and its motion, it is not. \
A person may TRIGGER the process as long as the caption is about the resulting motion \
("a ball bounces off the wall after being thrown", "water splashes as a rock is dropped into a \
pool" -> manipulation_focused=false).
- "category": one of {CATEGORIES} (use "none" when physical_process is false).
- "confidence": integer 1-5, how sure you are that this clip fits the goal \
(physical_process AND motion_rich AND NOT manipulation_focused).

ALWAYS physical_process=false for: people talking, presenting, interviews, news; human \
locomotion, dancing, exercise or sports where the focus is the athletes ("a man running", \
"players playing soccer", "a skateboarder doing tricks"); animals behaving ("a dog runs in the \
yard"); vehicles merely driving, flying or sailing ("cars on a highway", "a plane in the sky"); \
only camera motion over a static scene (drone / aerial / panning shots of landscapes or cities); \
screen recordings, video games, animation, CGI, slideshows, text or logos; static scenes, \
portraits, product displays.
The PHYSICAL PROCESS ITSELF must be the subject of the caption. If the main subject is a person, an animal, or a self-propelled vehicle and the physics is only background or a side effect, physical_process=false: fish / turtles / ducks swimming, surfers riding waves, kids on a swing, a person whose hair blows in the wind, a car driving with smoke behind it, "a person is boiling water in a pot" (a person doing something -> manipulation_focused=true).
Also false: a person using a tool or machine even if it throws sparks or spray (grinding, welding, snow plowing); passive phrasings of manipulation ("noodles are being poured into a strainer", "steak is being cooked") -> manipulation_focused=true; fantasy or CGI content (dragons, monsters, special effects in movies).
These DO count as physical processes: vehicle crashes or impacts, a rocket launch with an \
exhaust plume, a boat tossed by waves, objects carried by wind or water, a building demolition.

Examples:
Caption: "Water rushing down a rocky waterfall into a pool."
{{"physical_process": true, "motion_rich": true, "manipulation_focused": false, "category": "fluid_flow", "confidence": 5}}
Caption: "A glass falls off the table and shatters on the floor."
{{"physical_process": true, "motion_rich": true, "manipulation_focused": false, "category": "deformation_breaking", "confidence": 5}}
Caption: "A woman is pouring milk into a bowl of flour and stirring it."
{{"physical_process": false, "motion_rich": true, "manipulation_focused": true, "category": "none", "confidence": 1}}
Caption: "A man is cutting a piece of wood with a saw."
{{"physical_process": false, "motion_rich": true, "manipulation_focused": true, "category": "none", "confidence": 1}}
Caption: "A group of people playing basketball in a gym."
{{"physical_process": false, "motion_rich": true, "manipulation_focused": false, "category": "none", "confidence": 1}}
Caption: "An aerial view of a river flowing through a forest."
{{"physical_process": false, "motion_rich": false, "manipulation_focused": false, "category": "none", "confidence": 2}}
Caption: "Huge waves crashing against the rocks on the shore."
{{"physical_process": true, "motion_rich": true, "manipulation_focused": false, "category": "fluid_flow", "confidence": 5}}
Caption: "A man throws a rock into the lake and it splashes."
{{"physical_process": true, "motion_rich": true, "manipulation_focused": false, "category": "splash_pour", "confidence": 4}}
Caption: "A man on a surfboard rides a large wave."
{{"physical_process": false, "motion_rich": true, "manipulation_focused": false, "category": "none", "confidence": 1}}
Caption: "Fish swimming around a coral reef."
{{"physical_process": false, "motion_rich": true, "manipulation_focused": false, "category": "none", "confidence": 1}}
Caption: "A person is boiling pasta in a pot on the stove."
{{"physical_process": false, "motion_rich": true, "manipulation_focused": true, "category": "none", "confidence": 1}}
Caption: "A pot of water boiling vigorously on a stove."
{{"physical_process": true, "motion_rich": true, "manipulation_focused": false, "category": "fluid_flow", "confidence": 5}}
Caption: "Smoke rising from a burning house."
{{"physical_process": true, "motion_rich": true, "manipulation_focused": false, "category": "fire_smoke_explosion", "confidence": 5}}

Return ONLY the JSON object, nothing else."""

_JSON_RE = re.compile(r"\{.*\}", re.S)


def query_judge(caption: str, timeout: int = 120) -> dict:
    payload = {
        "model": VLLM_MODEL,
        "messages": [{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                     {"role": "user", "content": f'Caption: "{caption}"'}],
        "max_tokens": 100,
        "temperature": 0,
    }
    for attempt in range(4):
        try:
            r = requests.post(f"{VLLM_BASE}/chat/completions", json=payload, timeout=timeout)
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"]
            obj = json.loads(_JSON_RE.search(raw).group(0))
            cat = obj.get("category", "none")
            return {
                "physical_process": bool(obj["physical_process"]),
                "motion_rich": bool(obj["motion_rich"]),
                "manipulation_focused": bool(obj["manipulation_focused"]),
                "category": cat if cat in CATEGORIES else "other_physics",
                "confidence": int(obj.get("confidence", 0)),
            }
        except Exception as e:  # noqa: BLE001 -- retry transient server / parse errors
            err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"judge failed: {err!r}")


def passes(j: dict) -> bool:
    return (j["physical_process"] and j["motion_rich"] and not j["manipulation_focused"]
            and j["confidence"] >= MIN_CONF)


def stage_judge(workers: int, target_pass: int, limit: int, tier: str, out_path: str):
    cands = [json.loads(l) for l in open(STAGE1[tier])]
    random.Random(SEED).shuffle(cands)
    done, n_pass = set(), 0
    if os.path.exists(out_path):
        for l in open(out_path):
            r = json.loads(l)
            done.add(r["clip_key"])
            n_pass += passes(r["judge"])
    todo = [c for c in cands if c["clip_key"] not in done]
    if limit:
        todo = todo[:max(0, limit - len(done))]
    print(f"[judge] {len(cands)} candidates, {len(done)} already judged ({n_pass} passed), "
          f"{len(todo)} to go, target_pass={target_pass}", flush=True)

    lock, stop = threading.Lock(), threading.Event()
    stats = Counter(passed=n_pass)
    t0 = time.time()

    def work(c):
        if stop.is_set():
            return None
        c = dict(c, judge=query_judge(c["caption"]))
        return c

    with open(out_path, "a") as out, ThreadPoolExecutor(workers) as ex:
        # submit in chunks so we can stop soon after hitting the target
        it = iter(todo)
        pending = set()
        while True:
            while len(pending) < workers * 4 and not stop.is_set():
                c = next(it, None)
                if c is None:
                    break
                pending.add(ex.submit(work, c))
            if not pending:
                break
            fut = next(as_completed(pending))
            pending.discard(fut)
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001
                stats["errors"] += 1
                print(f"[judge] error: {e}", flush=True)
                continue
            if r is None:
                continue
            with lock:
                out.write(json.dumps(r) + "\n")
                stats["judged"] += 1
                stats["passed"] += passes(r["judge"])
                if stats["judged"] % 500 == 0:
                    out.flush()
                    rate = stats["judged"] / (time.time() - t0)
                    print(f"[judge] {dict(stats)} {rate:.1f}/s", flush=True)
                if target_pass and stats["passed"] >= target_pass:
                    stop.set()
    print(f"[judge] done: {dict(stats)} in {time.time() - t0:.0f}s -> {out_path}")


def stage_export():
    rows = []
    for tier, path in STAGE2.items():
        if os.path.exists(path):
            rows += [dict(json.loads(l), tier=tier) for l in open(path)]
    passed = [r for r in rows if passes(r["judge"])]
    with open(POOL, "w") as f:
        for r in passed:
            f.write(json.dumps(r) + "\n")
    cats = Counter(r["judge"]["category"] for r in passed)
    print(f"[export] judged {len(rows)}, passed {len(passed)} "
          f"({len(passed) / max(1, len(rows)):.1%}) -> {POOL}")
    print(f"  by tier: {dict(Counter(r['tier'] for r in passed))}")
    for k, v in cats.most_common():
        print(f"  {k:24s} {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["prefilter", "judge", "export"])
    ap.add_argument("--tier", choices=["A", "B"], default="A")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--target-pass", type=int, default=12_000)
    ap.add_argument("--limit", type=int, default=0, help="judge at most N candidates (pilot)")
    ap.add_argument("--out-name", default=None, help="judge output file name in _filter/")
    a = ap.parse_args()
    if a.stage == "prefilter":
        stage_prefilter(a.tier)
    elif a.stage == "judge":
        out = os.path.join(FILTER_DIR, a.out_name) if a.out_name else STAGE2[a.tier]
        stage_judge(a.workers, 0 if a.limit else a.target_pass, a.limit, a.tier, out)
    else:
        stage_export()


if __name__ == "__main__":
    main()
