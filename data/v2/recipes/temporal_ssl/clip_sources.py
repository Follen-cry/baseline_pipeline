#!/usr/bin/env python3
"""Per-source clip loaders for the v2 window pool.

Each loader reads that source's frozen selection manifest under RAW_ROOT and returns uniform clip rows:

    source, clip_id  : unique id within the source (also the frame dir name)
    video            : video file, or a directory of frame images (then `fps` is required)
    fps              : only for frame directories
    caption          : the source's own caption, used verbatim (outcome descriptions are fine)
    caption_source   : where the caption text comes from
    category         : the stratification label of the source (phenomenon / class / task / ...)
    source_meta      : source-specific fields kept for later analysis

Caption per source:
    physinone      caption.txt of the scene
    phyco_sim      the scene's common_caption_cosmos*.txt (friction_slide_flat_v2: the file of the
                   push direction on screen); ball_drop_v3 has none -> template from ball_drop_v2's
    physictran38k  metadata `prompt`
    vbvr           the generator `prompt`
    baai_physics   dataset.jsonl `caption` (Qwen-VL-72B, shipped with the data)
    mit_physics    the Qwen3-VL caption from our vlm_check pass (MiT itself ships only the class
                   label), minus watermark / "N sequential frames" remarks. The stock-footage title in
                   the file name is kept in source_meta only: it describes the whole stock video, not
                   the 3 s MiT cut, and often misses the physical process (checked on samples)
    panda70m       Panda-70M `caption`
    ssv2           the filled-in `label`
"""
import json, math, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../common"))
from paths import RAW_ROOT  # noqa: E402

SOURCES = ("physinone", "phyco_sim", "physictran38k", "vbvr",
           "baai_physics", "mit_physics", "panda70m", "ssv2")


def _rows(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def _raw(*p):
    return os.path.join(RAW_ROOT, *p)


def physinone():
    out = []
    for r in _rows(_raw("physinone", "selection_5k.jsonl")):
        d = r["out_dir"]
        done = json.load(open(os.path.join(d, "DONE")))
        out.append({"source": "physinone", "clip_id": r["case_id"], "video": os.path.join(d, "rgb"),
                    "fps": r.get("fps", 30), "caption": open(os.path.join(d, "caption.txt")).read().strip(),
                    "caption_source": "caption.txt", "category": "+".join(r["phenomena"]),
                    "source_meta": {"activity_type": r["activity_type"], "phenomena": r["phenomena"],
                                    "camera": done["camera"], "rendered_frames": done["frames"]}})
    return out


# PhyCo friction_slide_flat_v2: captions per on-screen push direction, 8 sectors of 45 degrees
_DIRS = ("right", "up-right", "up", "up-left", "left", "down-left", "down", "down-right")
_BALL_DROP_V3_TEMPLATE = (
    "{n} balls fall onto the ground naturally due to gravity and bounce up after impact. The only "
    "things that are moving in the video are the balls. All other objects are stationary. The video "
    "accurately captures the physics of balls falling and bouncing off a surface.")
_PHYCO_PARAMS = ("ball_restitution", "platform_friction", "num_balls", "jelly_texture", "force_magnitude")


def _screen_direction(meta):
    """8-way on-screen direction of the applied velocity (image y points down); None if the push
    point is off-screen (44 of the 1,111 selected samples), which then get the generic caption."""
    a = meta["applied_velocities_image"][0]
    if "velocity_arrow_unit_vector" not in a:
        return None
    vx, vy = a["velocity_arrow_unit_vector"]
    ang = math.degrees(math.atan2(-vy, vx)) % 360
    return _DIRS[int((ang + 22.5) // 45) % 8]


def phyco_sim():
    caps = {}

    def cap(scene, suffix=""):
        p = _raw("phyco_sim", scene, f"common_caption_cosmos{suffix}.txt")
        if p not in caps:
            caps[p] = open(p).read().strip() if os.path.exists(p) else ""
        return caps[p], os.path.basename(p)

    out = []
    for r in _rows(_raw("phyco_sim", "selection_10k.jsonl")):
        scene = r["scene"]
        meta = json.load(open(os.path.join(r["sample_dir"], "metadata.json")))
        sm = {"scene": scene, "strat_key": r.get("strat_key"), "strat_value": r.get("strat_value"),
              "settle_frame": r.get("settle_frame"), "motion_frames": r.get("motion_frames")}
        sm.update({k: meta[k] for k in _PHYCO_PARAMS if k in meta})
        if scene == "friction_slide_flat_v2":
            sm["screen_direction"] = _screen_direction(meta)
            d = sm["screen_direction"]
            text, src = cap(scene, "_" + d if d else "")
        elif scene == "ball_drop_v3":
            n = meta.get("num_balls") or r.get("strat_value")
            text, src = _BALL_DROP_V3_TEMPLATE.format(n=str(n).capitalize() if n else "Several"), "template(ball_drop_v2)"
        else:
            text, src = cap(scene)
        assert text, (scene, src)
        out.append({"source": "phyco_sim", "clip_id": r["id"], "video": r["video"], "caption": text,
                    "caption_source": src, "category": scene, "source_meta": sm})
    return out


def physictran38k():
    return [{"source": "physictran38k", "clip_id": r["id"], "video": r["video"], "caption": r["prompt"],
             "caption_source": "metadata.prompt", "category": r["transition"],
             "source_meta": {k: r[k] for k in ("domain", "subdomain", "transition", "tier", "in_moving_list")}}
            for r in _rows(_raw("physictran38k", "selection_10k.jsonl"))]


def vbvr():
    return [{"source": "vbvr", "clip_id": r["id"], "video": r["video"], "caption": r["prompt"],
             "caption_source": "generator.prompt", "category": r["task"],
             "source_meta": {k: r[k] for k in ("task", "chunk", "task_id")}}
            for r in _rows(_raw("vbvr", "manifest_all.jsonl")) if r["split"] == "train"]


def baai_physics():
    return [{"source": "baai_physics", "clip_id": r["id"], "video": r["video"], "caption": r["caption"],
             "caption_source": "dataset.caption", "category": r["kw_group"],
             "source_meta": {"query_folder": r["query_folder"], "kw_groups": r["kw_groups"],
                             "cam_motion": r["qc"]["cam_motion"], "obj_motion": r["qc"]["obj_motion"]}}
            for r in _rows(_raw("baai_physics", "selection_5k.jsonl"))]


# MiT file names: getty-<title>-video-id<digits>_<n>, vb-<title>-<random id>_<n>, a few
# <title>-video-id<id>_<n>; yt / giphy / flickr / meta / numeric names are ids only, vine / bing are
# social-media noise. Titles describe the whole stock video (MiT cut 3 s of it) and are often cut off.
_MIT_TRAILING_STOP = {"and", "the", "of", "in", "to", "a", "an", "with", "at", "on", "from", "for", "by", "as"}
_MIT_DROP = {"hd", "4k", "uhd"}


def _mit_title(video):
    name = re.sub(r"_\d+$", "", os.path.basename(video)[:-4])
    if name.startswith("vb-"):
        toks = name[3:].split("-")
        while toks and (re.search(r"\d", toks[-1]) or not re.search(r"[aeiouy]", toks[-1])):
            toks.pop()                                   # trailing random id tokens
    else:
        m = re.match(r"(?:getty-)?(.*)-video-id[0-9A-Za-z]+$", name)
        if not m or name.startswith(("vine-", "bing-")):
            return None
        toks = m.group(1).split("-")
    toks = [t for t in toks if t and t.lower() not in _MIT_DROP]
    while toks and toks[-1].lower() in _MIT_TRAILING_STOP:
        toks.pop()
    if len(toks) < 3:
        return None
    text = " ".join(toks)
    return text[0].upper() + text[1:] + "."


# Remarks about the Getty watermark or about the 4 frames the VLM was shown (~5% of captions)
_MIT_VLM_NOISE = [
    r",?\s*(?:and\s+)?(?:with|featuring|showing)\s+(?:a|an|the)?\s*[^,.]*?watermark[^,.]*",
    r",?\s*(?:with\s+(?:the\s+|its\s+|their\s+)?\w+(?:\s+\w+)?\s+)?(?:as\s+)?(?:captured|shown|seen|depicted)\s+"
    r"(?:in|across|over|through)\s+(?:a\s+(?:sequence|series)\s+of\s+)?(?:the\s+)?"
    r"(?:four|4|several|multiple|consecutive|sequential)\s+[^,.]*?frames[^,.]*",
    # only at the end of the caption, where dropping it cannot break the sentence
    r",?\s*(?:as\s+)?(?:(?:captured|shown|seen)\s+)?(?:in|across)\s+(?:a|the)\s+(?:sequence|series)\s+of\s+"
    r"(?:four\s+|4\s+)?(?:black and white\s+)?(?:real\s+)?(?:camera\s+|satellite\s+|video\s+)?(?:footage\s+)?"
    r"(?:frames|images|footage)\s*\.?\s*$",
    r",?\s*across\s+(?:the\s+)?(?:four|4)\s+frames\s*\.?\s*$",
]


def _mit_vlm_caption(text):
    for p in _MIT_VLM_NOISE:
        text = re.sub(p, "", text, flags=re.I)
    text = re.sub(r"\s+([,.])", r"\1", text).strip()
    return text if text.endswith(".") else text + "."


def mit_physics():
    out = []
    for r in _rows(_raw("mit_physics", "selection_5k.jsonl")):
        title = _mit_title(r["video"])
        out.append({"source": "mit_physics", "clip_id": r["id"], "video": r["video"],
                    "caption": _mit_vlm_caption(r["vlm"]["caption"]),
                    "caption_source": "vlm_check.qwen3vl",
                    "category": r["class"],
                    "source_meta": {"class": r["class"], "mit_split": r["split"], "filename_title": title,
                                    "vlm_caption": r["vlm"]["caption"], "cam_motion": r["qc"]["cam_motion"],
                                    "obj_motion": r["qc"]["obj_motion"]}})
    return out


def panda70m():
    return [{"source": "panda70m", "clip_id": r["id"], "video": r["video"], "caption": r["caption"],
             "caption_source": "panda70m.caption", "category": r["category"],
             "source_meta": {k: r.get(k) for k in ("tier", "video_id", "segment", "visual")}}
            for r in _rows(_raw("panda70m", "selection_5k.jsonl"))]


def ssv2():
    return [{"source": "ssv2", "clip_id": r["id"], "video": r["video"], "caption": r["label"],
             "caption_source": "ssv2.label", "category": r["template"],
             "source_meta": {k: r[k] for k in ("split", "group", "template", "placeholders")}}
            for r in _rows(_raw("ssv2", "selection_10k.jsonl"))]


LOADERS = {s: globals()[s] for s in SOURCES}


def load(source):
    rows = LOADERS[source]()
    ids = [r["clip_id"] for r in rows]
    assert len(ids) == len(set(ids)), f"{source}: duplicate clip ids"
    return rows


if __name__ == "__main__":
    for s in SOURCES:
        rows = load(s)
        print(f"{s:14s} {len(rows):7d} clips, {len({r['caption'] for r in rows}):6d} distinct captions")
