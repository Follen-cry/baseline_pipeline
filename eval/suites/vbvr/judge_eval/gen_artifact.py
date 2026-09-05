#!/usr/bin/env python3
"""Build the pure-VLM-judge scoring system artifact: per-task rubrics
(translated from the current rule-based evaluator's active criteria/weights),
the exact prompt text sent to the judge, and real examples comparing the
judge's score against the rule-based score on identical images.

    python gen_artifact.py --examples full_judge_examples.json --out judge_scoring.html
"""
import argparse
import base64
import html as htmlmod
import json
import os
import sys
from io import BytesIO

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "evaluators"))
from llm_judge_full import TASK_CRITERIA, build_full_judge_instruction  # noqa: E402

# Copied from artifacts/gen_rules_artifact.py's IMPL_DETAILS (re-checked 2026-08-26,
# not imported directly -- that script's HTML-writing runs at module import time,
# which would silently regenerate that other artifact's file as a side effect of
# building this one). Keep in sync if that file's detector descriptions change.
RULE_BASED_IMPL = {
    "ball_bounces_given_time": "<code>final_position</code> reuses <code>_track_ball_positions</code> (Hough circle detection, dark-blob contour fallback), run on the single final frame only.",
    "stable_sort": "<code>_detect_shapes</code>: HSV range (0,30,30)&ndash;(180,255,255) (anything not white/black) + contour extraction, area&gt;500px filter, to recover each shape's type/position/color for all 4 sub-scores.",
    "multi_object_placement": "<code>_detect_colored_objects</code> / <code>_detect_star_markers</code>: per-color HSV ranges (red/blue/green/yellow) + contour area filters (objects &gt;300px; stars 100&ndash;2000px with &ge;8 approxPolyDP vertices). Match thresholds: 30px for color_matching, 20px for star_invariance.",
    "grid_shift": "<code>_detect_colored_blocks</code> feeds a gate &mdash; <code>_evaluate_completeness</code> + <code>_evaluate_block_pattern_preservation</code> must both score &gt;0.5 (\"block_preserved\") before direction_correctness/step_accuracy/position_precision are scored at all; otherwise those 3 hard-zero and only completeness counts.",
    "rotation_puzzle": "<code>_detect_blue_pipes</code> isolates the pipe-tile shapes in the 2&times;2 grid via color mask; path_connection/rotation_accuracy compare tile connectivity and orientation between frame 0 and frame -1.",
    "shape_color_then_move": "<code>_detect_shapes_with_info</code> (shape/color/position) is called on both frame 0 and frame -1; first_row_preservation diffs the two directly, second_row_completion/color_accuracy compare the new shapes against it.",
    "animal_size_sorting": "<code>_detect_animals</code>: contour-based shape detection per animal; <code>_evaluate_sorting</code> / <code>_evaluate_alignment</code> then check final left-to-right order and baseline position.",
    "maze": "<code>_detect_path_markers</code> (orange/yellow mask) vs <code>_detect_walls</code> (black mask) &mdash; path_validity checks the drawn path pixels don't overlap the wall mask. <b>REVISED 2026-08-26:</b> this and the other 3 sub-scores (path_completeness, navigation_accuracy) were found to be blind to path LENGTH &mdash; a 5-pixel scribble near the start marker that touches no wall and forms one connected blob scored 0.4&ndash;0.71 in real generations, same as a full solve, because none of the original checks required the path to actually reach anywhere. Added a coverage gate on top: generated path-marker pixel count &divide; the GT's own path pixel count for that sample, multiplying the whole task_specific score by a ramp (full credit at &ge;60% coverage, floored at 10% below 15% coverage). Confirmed on a fresh random 10-sample check: 9/10 previously-inflated (0.24&ndash;0.71) stub-path scores collapsed to 0.02&ndash;0.07, matching what the images actually show (no real path drawn); clean synthetic full-solve pairs are unaffected (gate stays 1.0).",
    "2d_geometric_transformation": "<code>_detect_main_shape</code>: HSV saturation&gt;50 mask isolates the colored shape from the grayscale background. <code>_detect_target_outline</code>: Canny edges pick out the dashed target outline. Orientation via <code>cv2.fitEllipse</code> angle.",
}

VISUAL_COHERENCE_NOTE = (
    "<b>Also applies to every task above (added 2026-08-26):</b> a generic "
    "<code>_visual_coherence_penalty</code> gate runs after task_specific is computed, "
    "regardless of which detector produced it. Real (untrained-base-model) generations "
    "are often visually incoherent in ways clean synthetic test pairs never are &mdash; "
    "e.g. a ball_bounces_given_time sample scored 0.995 where the \"ball\" was actually a "
    "thin orange ring of dots with no resemblance to the GT's solid pink circle, and a "
    "shape_color_then_move sample scored 1.0 with garbled text baked into wrong-colored "
    "shapes &mdash; both cases the per-task detectors' loose color/position thresholds were "
    "coincidentally satisfied by, and both a VLM judge immediately called wrong on sight. "
    "The gate computes what fraction of the candidate's foreground pixels (by area, and "
    "separately by connected-component &mdash; so a small wrong object can't hide inside a "
    "large shared correct one, like a static border) have no close match anywhere in that "
    "sample's own GT palette, and discounts task_specific accordingly. Verified as a true "
    "no-op on every task's clean synthetic pairs (penalty stays 1.0) before trusting it on "
    "real data; raised rule-vs-judge Spearman correlation across 900 real generations from "
    "0.14 to 0.26."
)

CATEGORY_OF = {
    "ball_bounces_given_time": "Knowledge",
    "stable_sort": "Perception",
    "multi_object_placement": "Perception",
    "grid_shift": "Transformation",
    "rotation_puzzle": "Transformation",
    "shape_color_then_move": "Abstraction",
    "animal_size_sorting": "Perception",
    "maze": "Spatiality",
    "2d_geometric_transformation": "Transformation",
}
CAT_VAR = {
    "Abstraction": "--cat-abstraction",
    "Knowledge": "--cat-knowledge",
    "Perception": "--cat-perception",
    "Spatiality": "--cat-spatiality",
    "Transformation": "--cat-transformation",
}
TASK_ORDER = list(TASK_CRITERIA.keys())


def esc(s):
    return htmlmod.escape(s or "", quote=True)


def img_data_uri(path, max_side=360, quality=82):
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return None
    w, h = im.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def weight_bar(criteria):
    total = sum(w for _, w, _ in criteria)
    segs = []
    palette = ["#9A3E85", "#C46FB0", "#DB93C7", "#EFC3E2", "#F7DEF0"]
    for i, (key, w, _) in enumerate(criteria):
        pct = round(w / total * 1000) / 10
        color = palette[i % len(palette)]
        segs.append(f'<div class="wbar-seg" style="width:{pct}%;background:{color};" title="{esc(key)} &middot; {pct}%"></div>')
    return f'<div class="wbar">{"".join(segs)}</div>'


def criteria_list(criteria):
    items = []
    for key, w, levels in criteria:
        items.append(f"""
        <li><span class="crit-head"><code>{esc(key)}</code><span class="crit-weight">{w:g}%</span></span>
          <ul class="level-defs">
            <li><span class="level-tag level-correct">correct</span>{esc(levels['correct'])}</li>
            <li><span class="level-tag level-partial">partial</span>{esc(levels['partial'])}</li>
            <li><span class="level-tag level-wrong">wrong</span>{esc(levels['wrong'])}</li>
          </ul></li>""")
    return "<ul class='crit-list'>" + "".join(items) + "</ul>"


def agreement_badge(rule_score, judge_score):
    diff = judge_score - rule_score
    if diff >= 0.5:
        return '<span class="badge badge-severe">Judge scored much higher</span>'
    if abs(diff) >= 0.5:
        return '<span class="badge badge-severe">Judge scored much lower</span>'
    if abs(diff) >= 0.25:
        return '<span class="badge badge-warn">Disagreement</span>'
    return '<span class="badge badge-ok">Close agreement</span>'


def example_card(ex):
    input_uri = img_data_uri(ex["input_image"])
    gen_uri = img_data_uri(ex["generated_image"])
    gt_uri = img_data_uri(ex["target_image"])
    prompt = ex["prompt"].split("Task: ", 1)[-1] if "Task: " in ex["prompt"] else ex["prompt"]

    crit_rows = ""
    if ex["judge_criteria"]:
        crit_rows = "".join(
            f'<div class="crow"><code>{esc(k)}</code><span class="level-tag level-{esc(v)}">{esc(v)}</span></div>'
            for k, v in ex["judge_criteria"].items()
        )

    return f"""
    <div class="example">
      <div class="example-head">
        <span class="sample-label">{esc(ex['label'])}</span>
        {agreement_badge(ex['rule_score'], ex['judge_score'])}
      </div>
      <div class="sample-imgs">
        <figure><img src="{input_uri}" alt="input" loading="lazy"><figcaption>input</figcaption></figure>
        <figure><img src="{gen_uri}" alt="InternVL-U output" loading="lazy"><figcaption>InternVL-U output</figcaption></figure>
        <figure><img src="{gt_uri}" alt="ground truth" loading="lazy"><figcaption>ground truth</figcaption></figure>
      </div>
      <p class="sample-prompt">{esc(prompt)}</p>
      <div class="score-compare">
        <div class="score-box">
          <div class="score-label">Rule-based</div>
          <div class="score-num mono">{ex['rule_score']:.3f}</div>
        </div>
        <div class="score-box accent-box">
          <div class="score-label">Judge (Qwen3-VL-30B)</div>
          <div class="score-num mono">{ex['judge_score']:.3f}</div>
          <div class="crit-breakdown">{crit_rows}</div>
        </div>
      </div>
      <p class="judge-reasoning"><b>Judge reasoning:</b> &ldquo;{esc(ex['judge_reasoning'] or '')}&rdquo;</p>
    </div>"""


def task_section(task_name, examples):
    task_desc, criteria = TASK_CRITERIA[task_name]
    category = CATEGORY_OF[task_name]
    cat_var = CAT_VAR[category]
    instruction = build_full_judge_instruction(task_name, "<the sample's actual prompt would appear here>")
    ex_html = "".join(example_card(ex) for ex in examples)

    return f"""
    <section class="task-section" id="task-{esc(task_name)}">
      <div class="task-head">
        <span class="pill" style="background:var({cat_var});">{esc(category)}</span>
        <h2><code class="mono">{esc(task_name)}</code></h2>
      </div>
      <p class="task-desc">{esc(task_desc)}</p>
      <div class="impl-note">
        <p class="section-label">How the rule-based evaluator actually works</p>
        <p class="impl-text">{RULE_BASED_IMPL.get(task_name, "Not documented.")}</p>
        <p class="impl-text impl-text-gate">{VISUAL_COHERENCE_NOTE}</p>
      </div>
      <div class="two-col">
        <div>
          <p class="section-label">Criteria (from the current rule-based evaluator)</p>
          {weight_bar(criteria)}
          {criteria_list(criteria)}
        </div>
        <div>
          <p class="section-label">Exact prompt sent to the judge</p>
          <pre class="judge-prompt mono">{esc(instruction)}</pre>
        </div>
      </div>
      <p class="section-label examples-label">Real examples (same images the rule-based system scored)</p>
      <div class="examples-grid">{ex_html}</div>
    </section>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default=os.path.join(HERE, "full_judge_examples.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "judge_scoring.html"))
    args = ap.parse_args()

    data = json.load(open(args.examples))
    results = data["results"]

    all_gaps = []
    n_judge_zero_rule_positive = 0
    hallucination = None  # (task, ex) where judge scored much higher than rule
    for t, exs in results.items():
        for ex in exs:
            gap = ex["rule_score"] - ex["judge_score"]
            all_gaps.append(abs(gap))
            if ex["judge_score"] == 0.0 and ex["rule_score"] > 0.0:
                n_judge_zero_rule_positive += 1
            if gap <= -0.5 and (hallucination is None or gap < hallucination[2]):
                hallucination = (t, ex, gap)
    n_examples = len(all_gaps)
    mean_gap = sum(all_gaps) / n_examples if n_examples else 0
    n_large = sum(1 for g in all_gaps if g >= 0.5)

    if hallucination:
        h_task, h_ex, _ = hallucination
        h_label = h_ex["label"].split(" (")[0].lower()
        h_judge_score = f"{h_ex['judge_score']:.3f}"
        h_rule_score = f"{h_ex['rule_score']:.3f}"
        h_reasoning = (h_ex["judge_reasoning"] or "")[:60]
    else:
        h_task, h_label, h_judge_score, h_rule_score, h_reasoning = "", "", "0.000", "0.000", ""

    sections = "".join(task_section(t, results[t]) for t in TASK_ORDER if t in results)
    nav = "".join(f'<a href="#task-{esc(t)}">{esc(t)}</a>' for t in TASK_ORDER if t in results)

    html = f"""<title>Rule vs. Judge Scoring</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {{
  --bg: #F7F2F6;
  --surface: #FFFFFF;
  --surface-alt: #F0E7EE;
  --ink: #1C1420;
  --ink-soft: #5B4C57;
  --ink-faint: #8C7C88;
  --border: #E0D2DC;
  --accent: #9A3E85;
  --accent-soft: #F3E3EF;
  --code-bg: #F1E6EE;
  --good: #1C6B5E;
  --warn: #A9700E;
  --severe: #A8395A;
  --cat-abstraction: #6B4FA0;
  --cat-knowledge: #2A5CA6;
  --cat-perception: #A8395A;
  --cat-spatiality: #1C6B5E;
  --cat-transformation: #B06A1A;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #1B141A;
    --surface: #241C23;
    --surface-alt: #2C2229;
    --ink: #F3E9F1;
    --ink-soft: #C9B7C4;
    --ink-faint: #93818E;
    --border: #3B2E38;
    --accent: #DB93C7;
    --accent-soft: #3A2635;
    --code-bg: #2E2430;
    --good: #5FBFA3;
    --warn: #D9A548;
    --severe: #E37FA0;
    --cat-abstraction: #A98FE0;
    --cat-knowledge: #7DA8E8;
    --cat-perception: #E37FA0;
    --cat-spatiality: #5FBFA3;
    --cat-transformation: #E0A356;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #1B141A;
  --surface: #241C23;
  --surface-alt: #2C2229;
  --ink: #F3E9F1;
  --ink-soft: #C9B7C4;
  --ink-faint: #93818E;
  --border: #3B2E38;
  --accent: #DB93C7;
  --accent-soft: #3A2635;
  --code-bg: #2E2430;
  --good: #5FBFA3;
  --warn: #D9A548;
  --severe: #E37FA0;
  --cat-abstraction: #A98FE0;
  --cat-knowledge: #7DA8E8;
  --cat-perception: #E37FA0;
  --cat-spatiality: #5FBFA3;
  --cat-transformation: #E0A356;
}}
* {{ box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--ink); font-family: "IBM Plex Sans", system-ui, sans-serif; margin: 0; padding: 0 0 5rem; }}
.mono {{ font-family: "IBM Plex Mono", ui-monospace, monospace; font-variant-numeric: tabular-nums; }}
.wrap {{ max-width: 1100px; margin: 0 auto; padding: 0 2rem; }}
header.top {{ border-bottom: 1px solid var(--border); background: var(--surface); padding: 3rem 0 2.25rem; }}
.eyebrow {{ font-family: "IBM Plex Mono", monospace; font-size: 0.78rem; letter-spacing: 0.09em; text-transform: uppercase; color: var(--accent); margin: 0 0 0.9rem; display: flex; align-items: center; gap: 0.6rem; }}
.eyebrow::before {{ content: ""; width: 1.6rem; height: 1px; background: var(--accent); display: inline-block; }}
h1 {{ font-size: clamp(1.8rem, 3.2vw, 2.4rem); line-height: 1.15; margin: 0 0 0.9rem; text-wrap: balance; }}
.lede {{ color: var(--ink-soft); max-width: 72ch; line-height: 1.6; margin: 0 0 1.6rem; }}
.callout {{ background: var(--accent-soft); border: 1px solid var(--border); border-left: 3px solid var(--accent); border-radius: 8px; padding: 0.9rem 1.1rem; max-width: 72ch; font-size: 0.9rem; line-height: 1.6; color: var(--ink); margin: 0 0 1.6rem; }}
.stat-row {{ display: flex; flex-wrap: wrap; gap: 0.9rem; margin-top: 1.2rem; }}
.stat {{ background: var(--surface-alt); border: 1px solid var(--border); border-radius: 10px; padding: 0.9rem 1.1rem; min-width: 9rem; }}
.stat-label {{ font-size: 0.74rem; color: var(--ink-faint); text-transform: uppercase; letter-spacing: 0.06em; }}
.stat-value {{ font-family: "IBM Plex Mono", monospace; font-size: 1.7rem; font-weight: 600; color: var(--ink); }}
.stat-value.accent {{ color: var(--accent); }}
code {{ background: var(--code-bg); border-radius: 4px; padding: 0.05em 0.4em; font-size: 0.92em; }}
.pill {{ display: inline-block; padding: 0.18rem 0.6rem; border-radius: 999px; font-size: 0.72rem; font-weight: 600; color: #fff; letter-spacing: 0.01em; white-space: nowrap; }}
nav.task-nav {{ display: flex; flex-wrap: wrap; gap: 0.4rem 0.9rem; margin-top: 1.4rem; }}
nav.task-nav a {{ font-family: "IBM Plex Mono", monospace; font-size: 0.78rem; color: var(--accent); text-decoration: none; }}
nav.task-nav a:hover {{ text-decoration: underline; }}
section.task-section {{ max-width: 1100px; margin: 0 auto; padding: 2.6rem 2rem; border-bottom: 1px solid var(--border); }}
.task-head {{ display: flex; align-items: center; gap: 0.7rem; margin-bottom: 0.6rem; }}
.task-head h2 {{ font-size: 1.25rem; margin: 0; }}
.task-desc {{ color: var(--ink-soft); max-width: 76ch; line-height: 1.55; margin: 0 0 1.4rem; }}
.impl-note {{ background: var(--surface-alt); border: 1px solid var(--border); border-radius: 8px; padding: 0.8rem 1rem; margin: 0 0 1.6rem; max-width: 90ch; }}
.impl-text {{ margin: 0; font-size: 0.82rem; color: var(--ink-soft); line-height: 1.55; }}
.impl-text-gate {{ margin-top: 0.7rem; padding-top: 0.7rem; border-top: 1px dashed var(--border); }}
.two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-bottom: 1.6rem; }}
.section-label {{ font-size: 0.76rem; color: var(--ink-faint); text-transform: uppercase; letter-spacing: 0.05em; margin: 0 0 0.6rem; }}
.wbar {{ display: flex; height: 0.9rem; border-radius: 5px; overflow: hidden; border: 1px solid var(--border); margin-bottom: 0.8rem; }}
.wbar-seg {{ height: 100%; }}
.crit-list {{ list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.9rem; }}
.crit-head {{ display: flex; justify-content: space-between; align-items: baseline; font-size: 0.86rem; }}
.crit-weight {{ font-family: "IBM Plex Mono", monospace; color: var(--accent); font-weight: 600; }}
.level-defs {{ list-style: none; margin: 0.35rem 0 0; padding: 0; display: flex; flex-direction: column; gap: 0.3rem; }}
.level-defs li {{ font-size: 0.78rem; color: var(--ink-soft); line-height: 1.45; }}
.level-tag {{ display: inline-block; font-family: "IBM Plex Mono", monospace; font-size: 0.66rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em; padding: 0.1rem 0.4rem; border-radius: 4px; margin-right: 0.4rem; color: #fff; vertical-align: middle; }}
.level-correct {{ background: var(--good); }}
.level-partial {{ background: var(--warn); }}
.level-wrong {{ background: var(--severe); }}
.judge-prompt {{ background: var(--code-bg); border: 1px solid var(--border); border-radius: 8px; padding: 0.9rem 1rem; font-size: 0.72rem; line-height: 1.55; white-space: pre-wrap; word-break: break-word; max-height: 22rem; overflow-y: auto; color: var(--ink-soft); }}
.examples-label {{ margin-top: 0.4rem; }}
.examples-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 1.2rem; }}
.example {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1rem; }}
.example-head {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.7rem; }}
.sample-label {{ font-size: 0.76rem; color: var(--ink-faint); text-transform: uppercase; letter-spacing: 0.04em; }}
.badge {{ font-size: 0.68rem; font-weight: 600; padding: 0.15rem 0.55rem; border-radius: 999px; color: #fff; }}
.badge-ok {{ background: var(--good); }}
.badge-warn {{ background: var(--warn); }}
.badge-severe {{ background: var(--severe); }}
.sample-imgs {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; margin-bottom: 0.7rem; }}
.sample-imgs figure {{ margin: 0; }}
.sample-imgs img {{ width: 100%; aspect-ratio: 1; object-fit: contain; background: var(--surface-alt); border-radius: 6px; border: 1px solid var(--border); }}
.sample-imgs figcaption {{ text-align: center; font-size: 0.66rem; color: var(--ink-faint); margin-top: 0.3rem; text-transform: uppercase; letter-spacing: 0.03em; }}
.sample-prompt {{ font-size: 0.8rem; color: var(--ink-soft); line-height: 1.5; margin: 0 0 0.8rem; max-height: 5em; overflow-y: auto; }}
.score-compare {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.7rem; margin-bottom: 0.7rem; }}
.score-box {{ background: var(--surface-alt); border: 1px solid var(--border); border-radius: 8px; padding: 0.6rem 0.8rem; }}
.score-box.accent-box {{ border-color: var(--accent); }}
.score-label {{ font-size: 0.68rem; color: var(--ink-faint); text-transform: uppercase; letter-spacing: 0.03em; margin-bottom: 0.15rem; }}
.score-num {{ font-size: 1.3rem; font-weight: 600; }}
.crit-breakdown {{ margin-top: 0.5rem; display: flex; flex-direction: column; gap: 0.15rem; }}
.crow {{ display: flex; justify-content: space-between; font-size: 0.72rem; color: var(--ink-soft); }}
.judge-reasoning {{ font-size: 0.78rem; color: var(--ink-soft); line-height: 1.5; font-style: italic; margin: 0; }}
footer.meta {{ padding: 2.4rem 2rem 0; max-width: 1100px; margin: 0 auto; }}
footer.meta p {{ color: var(--ink-faint); font-size: 0.85rem; line-height: 1.7; max-width: 74ch; }}
@media (max-width: 720px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
</style>

<header class="top">
  <div class="wrap">
    <p class="eyebrow">InternVL-U &middot; VBVR-CustomEval &middot; Qwen3-VL-30B-fp8</p>
    <h1>How Each Task Is Scored: Rule vs. Judge</h1>
    <p class="lede">For each of the 9 active tasks with real eval data, this page explains BOTH ways InternVL-U's generations get scored: the existing rule-based detector (a short plain-language summary of what its code actually checks) and a pure Qwen3-VL-30B-fp8 judge built as a like-for-like alternative &mdash; same sub-criteria and weights the rule-based evaluator uses (its active, non-dropped <code>task_specific</code> dimensions &mdash; see <code>evaluators/llm_judge_full.py</code>), classified into wrong / partial / correct per criterion rather than a bare number, then combined with the same weights. It's a straight swap of detector-code for judge-vision, not a different rubric. Validated on the same "Representative" / "Worst-scoring" examples already shown in the target-frame-prediction results artifact, so rule vs. judge scores are directly comparable on identical images.</p>
    <div class="callout"><b>2026-08-26/27 revision, round 2:</b> auditing fresh random samples by eye (not just aggregate stats) surfaced two more issues, on top of the visual-coherence gate described in each task's card below. (1) <code>maze</code>'s rule-based score had a real bug &mdash; short path stubs that never reached the end scored 0.4&ndash;0.71, now fixed with a path-coverage gate (see that task's card); rule and judge means for this task now agree closely (0.03 vs 0.03). (2) The judge had a real, harder-to-fix weakness on tasks needing precise grid-cell or tile-position verification (<code>grid_shift</code>, <code>rotation_puzzle</code>): it would confidently mark position-precision criteria "correct" with a detailed-sounding but fabricated justification (e.g. "moved exactly 2 cells left, landing in the correct grid cells") on candidates that visibly don't match the ground truth. A prompt instruction requiring element-by-element comparison helped partially; sending higher-resolution images (900px vs 512px) made no difference. For <code>grid_shift</code>'s and <code>rotation_puzzle</code>'s position-precision-style criteria specifically, <b>trust the rule-based score over the judge's</b> &mdash; the rule side is deterministic pixel comparison and held up under repeated manual spot-checks.</div>
    <div class="callout"><b>A stronger fix was tried and reverted:</b> forcing the judge to write a concrete "observation" for each criterion before its verdict (a standard LLM-grounding technique) fixed several individually-flagged false positives when spot-tested against known problem cases. But re-run at FULL SCALE (all 900 real generations, not just the cases it was tuned against) it was a clear regression: judge means jumped up across nearly every task (e.g. <code>maze</code> 0.06&rarr;0.63) and rule-vs-judge correlation <i>dropped</i> (0.26&rarr;0.17). Checked directly: on a maze image with nothing drawn at all (just walls and the end flag, no path, no start marker), the judge scored it a perfect 1.0 with a detailed, entirely fabricated description ("a complete, valid path of yellow dots connecting start to end"). 180/900 samples (20%) showed this pattern &mdash; forcing a plausible-sounding justification made the model MORE prone to confabulating support for a verdict it was already inclined toward, not more grounded. Reverted; kept the lighter element-by-element instruction from the previous paragraph, which tested fine at full scale. <b>Lesson:</b> a prompt change validated only against previously-flagged failure cases is confirmation-biased and does not predict full-corpus behavior &mdash; check a fresh random sample before committing to a full re-score.</div>
    <div class="callout"><b>Headline finding:</b> the judge is much harsher than the rule-based system on partial-success cases &mdash; on {n_judge_zero_rule_positive} of {n_examples} examples here it scored 0 where the rule-based system gave partial credit for a close-but-imperfect match, behaving almost binary (correct / not) rather than continuous, even with the 3-level wrong/partial/correct scale (this isn't a granularity artifact of a 0-100 scale). It also isn't infallible in the other direction: on <code>{esc(h_task)}</code>'s {esc(h_label)} example, the judge scored a clearly wrong layout as a perfect {h_judge_score} ("{esc(h_reasoning)}&hellip;") when the rule-based system correctly caught it at {h_rule_score} &mdash; see that task's card below for the actual images.</div>
    <div class="stat-row">
      <div class="stat"><div class="stat-label">Tasks covered</div><div class="stat-value mono">{len(results)}</div></div>
      <div class="stat"><div class="stat-label">Examples scored</div><div class="stat-value mono">{n_examples}</div></div>
      <div class="stat"><div class="stat-label">Mean |rule &minus; judge|</div><div class="stat-value accent mono">{mean_gap:.3f}</div></div>
      <div class="stat"><div class="stat-label">Gap &ge; 0.5</div><div class="stat-value mono">{n_large}/{n_examples}</div></div>
    </div>
    <nav class="task-nav">{nav}</nav>
  </div>
</header>

{sections}

<footer class="meta">
  <p><b>Judge:</b> <code>qwen3-vl-30b-fp8</code> via a local OpenAI-compatible vLLM server (<code>localhost:8000</code>), temperature 0, max_tokens 400. <b>Images:</b> first frame, InternVL-U's candidate output, and ground truth, each downscaled to 512px on the long edge before encoding (tested 900px on the judge's most stubborn false-positive cases in round 2: no change in verdict or reasoning, so 512px is not the bottleneck). <b>Aggregation:</b> weighted sum of the judge's per-criterion 0-100 sub-scores using the same weights <code>image_evaluator.py</code>'s rule-based evaluator uses for that task (see <code>evaluators/llm_judge_full.py</code>'s <code>TASK_CRITERIA</code>). This is a standalone alternative scoring path, independent of <code>llm_judge.py</code>'s narrower supplement-only judge (which covers only <code>key_door_matching</code>, a parked task). <b>Final validated state (2026-08-27):</b> rule-vs-judge Spearman correlation across all 900 real generations is 0.314 (up from a 0.14 baseline before any of this round's fixes).</p>
  <p><code>glass_refraction</code> is not covered here: it has no real InternVL-U eval generations yet (its DataFactory generator repo is access-restricted, same limitation noted in the results artifact and README).</p>
</footer>
"""

    with open(args.out, "w") as f:
        f.write(html)
    print(f"wrote {args.out} {len(html)} chars")


if __name__ == "__main__":
    main()
