#!/usr/bin/env python3
"""Generate the VBVR image-only evaluation strategy artifact HTML."""
import json
import html as htmlmod

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
EXAMPLES_PATH = _os.path.join(_HERE, "vbvr_task_examples.json")
VALIDATION_PATH = _os.path.join(_HERE, "..", "validation", "validation_results.json")
OUT_PATH = _os.path.join(_HERE, "vbvr_image_eval_strategy.html")

with open(EXAMPLES_PATH) as f:
    EXAMPLES = json.load(f)

with open(VALIDATION_PATH) as f:
    VALIDATION = json.load(f)["results"]

import sys as _sys
_sys.path.insert(0, _os.path.join(_HERE, "..", "evaluators"))
from image_evaluator import LOCKED_TASKS  # noqa: E402

# tier: 1 = fully compatible, 2 = partial degradation, 3 = unreliable
TASKS = [
    dict(id="O-12", full="O-12_shape_color_then_scale_data-generator", name="shape_color_then_scale",
         split="ID", category="Abstraction", tier=1,
         weights=[("two_step_rule", 30, False), ("first_step", 25, False), ("second_step", 25, False), ("sequence", 20, False)],
         rationale="task_specific reads only video_frames[0] / video_frames[-1] &mdash; validated: 5/5 GT-matched samples scored identically (1.000) between the video and image evaluators, and 5/5 deliberately-mismatched samples scored identically too (0.075&ndash;0.475). Pilot task for the mixin."),
    dict(id="O-13", full="O-13_shape_outline_then_move_data-generator", name="shape_outline_then_move",
         split="ID", category="Abstraction", tier=1,
         weights=[("two_step_rule", 30, False), ("first_step", 25, False), ("second_step", 25, False), ("sequence", 20, False)],
         rationale="Same composite-transform pattern as O-12: centroid position is read from frame 0 and frame -1 only, no intermediate sampling."),
    dict(id="O-19", full="O-19_mirror_reflection_data-generator", name="mirror_reflection",
         split="ID", category="Knowledge", tier=1,
         weights=[("reflection_angle", 40, False), ("symmetry", 30, False), ("ray_extension", 20, False), ("starting_point", 10, False)],
         rationale="Reflection-law geometry (incident angle = reflected angle) is measured by detecting line segments in the final frame alone via Hough transform &mdash; no trajectory needed."),
    dict(id="O-44", full="O-44_rotation_puzzle_data-generator", name="rotation_puzzle",
         split="ID", category="Transformation", tier=1,
         weights=[("path_connection", 40, False), ("rotation_accuracy", 30, False), ("position_preservation", 20, False), ("alignment_precision", 10, False)],
         rationale="Pipe-connectivity is a property of the final tile arrangement; evaluator diffs frame 0 vs frame -1 only."),
    dict(id="G-3", full="G-3_stable_sort_data-generator", name="stable_sort",
         split="ID", category="Perception", tier=1,
         weights=[("classification", 30, False), ("order", 30, False), ("fidelity", 30, False), ("layout", 10, False)],
         rationale="Sort correctness is read entirely off the final arrangement; fidelity compares shape count/area between frame 0 and frame -1."),
    dict(id="G-18", full="G-18_grid_shortest_path_data-generator", name="grid_shortest_path",
         split="ID", category="Spatiality", tier=2,
         weights=[("path_optimal", 50, True), ("completion", 25, False), ("movement", 15, False), ("fidelity", 10, False)],
         rationale="Reclassified after the 20-pair validation run caught it: manual review missed that path_optimal (50%!) sums position deltas via a bare <code>for frame in video_frames:</code> loop &mdash; a third disguise beyond indexed access and passed-as-argument calls. With 2 frames the generated path collapses to a straight line while the GT path (still the real, un-truncated video) stays the true winding length, structurally biasing the ratio low. Only 20% exact-match with the video evaluator on correctly-paired samples exposed this; dropped and renormalized across completion/movement/fidelity."),
    dict(id="O-11", full="O-11_shape_color_then_move_data-generator", name="shape_color_then_move",
         split="OOD", category="Abstraction", tier=1,
         weights=[("first_row_preservation", 40, False), ("second_row_completion", 35, False), ("color_accuracy", 20, False), ("shape_count", 5, False)],
         rationale="Row-completion analogy task: shape/color detection runs on frame 0 and frame -1 only. Our OOD-L1 operator-recombination pick."),
    dict(id="O-39", full="O-39_maze_data-generator", name="maze",
         split="OOD", category="Spatiality", tier=1,
         weights=[("path_validity", 45, False), ("path_completeness", 30, False), ("navigation_accuracy", 20, False), ("element_preservation", 5, False)],
         rationale="Like grid_shortest_path, every sub-score is computed by comparing the maze image in frame 0 against frame -1 (wall/marker detection), not by replaying the solve animation."),
    dict(id="O-65", full="O-65_animal_size_sorting_data-generator", name="animal_size_sorting",
         split="OOD", category="Perception", tier=1,
         weights=[("sorting", 40, False), ("alignment", 30, False), ("fidelity", 20, False), ("completeness", 10, False)],
         rationale="Final-arrangement check, same pattern as stable_sort. Fidelity/completeness diff frame 0 vs frame -1."),

    dict(id="O-36", full="O-36_grid_shift_data-generator", name="grid_shift",
         split="ID", category="Transformation", tier=2,
         weights=[("direction_correctness", 30, False), ("step_accuracy", 30, False), ("synchronization", 20, True), ("position_precision", 15, False), ("completeness", 5, False)],
         rationale="synchronization (20%) samples up to 10 frames to check all blocks move in lockstep; with only 2 frames it falls back to a neutral constant 0.5 (defined behavior in the source for &lt;3 frames) rather than measuring anything. The other 80% of the sub-score is frame 0 / frame -1 only."),
    dict(id="G-5", full="G-5_multi_object_placement_data-generator", name="multi_object_placement",
         split="ID", category="Perception", tier=2,
         weights=[("color_matching", 30, False), ("alignment", 25, False), ("path", 20, True), ("fidelity", 15, False), ("star_invariance", 10, False)],
         rationale="path (20%) measures motion-smoothness variance across up to 10 sampled frames; with 2 frames it hard-codes 0.2, a mild fixed penalty on every submission regardless of quality. The remaining 80% (color/alignment/fidelity/star) is fully first&ndash;final safe."),
    dict(id="O-6", full="O-6_2d_geometric_transformation_data-generator", name="2d_geometric_transformation",
         split="OOD", category="Transformation", tier=2,
         weights=[("rotation_center", 30, True), ("rotation_angle", 35, False), ("position_alignment", 25, False), ("shape_fidelity", 10, False)],
         rationale="rotation_center (30%) needs &ge;3 sampled centers to fit a circular arc; with 2 frames it returns a hard 0.0 (not a neutral default) &mdash; a systematic &minus;7.5%-of-total penalty baked into every image-only score on this task. Our OOD-L1 structural-transform pick, worth a manual override before trusting absolute numbers."),

    dict(id="O-15", full="O-15_ball_bounces_given_time_data-generator", name="ball_bounces_given_time",
         split="ID", category="Knowledge", tier=3,
         weights=[("final_position", 100, False)],
         rationale="REVISED 2026-08-25: briefly carried a path_shape criterion (IoU between generated/GT orange bounce-path masks), validated against VBVR-Bench's frozen GT samples where final_frame.png draws the entire path as a static polyline. But freshly generating new samples via this task's own DataFactory generator the same day showed the visual style has since changed upstream &mdash; final_frame.png now shows only the ball resting at its final position, no path drawn. path_shape was removed rather than kept as a metric that no longer matches what this task's data looks like; final_position (ball-position distance, unchanged from the original detector) is now the sole task_specific score. bounce_count/physics/trajectory/smoothness remain unrecoverable from a single frame under the current visual style and aren't replaced. No VLM judge is used for this task &mdash; final_position is a plain deterministic check that doesn't need supplementation."),
    dict(id="G-45", full="G-45_key_door_matching_data-generator", name="key_door_matching",
         split="ID", category="Spatiality", tier=3,
         weights=[("agent_movement", 30, False), ("key_collected", 35, True), ("door_reached", 25, True), ("sequence", 10, True)],
         rationale="key_collected + door_reached + sequence (70% of the sub-score) require the agent to be spatially co-located with the key/door in a *sampled* frame; with only start/end frames available, an agent that legitimately collected the key mid-path will usually score as if it never did. Only agent_movement (30%) is safe."),
    dict(id="O-18", full="O-18_glass_refraction_data-generator", name="glass_refraction",
         split="OOD", category="Knowledge", tier=1,
         weights=[("snells_law", 50, False), ("ray_direction", 30, False), ("ray_completeness", 15, False), ("scene_fidelity", 5, False)],
         rationale="Replaces O-62_gravity_physics (2026-08-25): that task's generator doesn't clamp the ball at ground level, so 4 of 5 downloaded GT samples simulate the ball to a negative height by the stated duration and render an empty final_frame &mdash; confirmed by computing h(3)=h0+v0&middot;3-0.5g&middot;9 for each (all negative). Every scoring method inherits that upstream defect. glass_refraction has no such failure mode (refraction geometry always stays on-canvas) and only reads video_frames[0]/[-1] &mdash; Tier 1, no degradation needed. VBVR-Bench itself files this under its own In-Domain_50 folder, but that's just where its GT ships &mdash; since we pick our own train/test split rather than inheriting VBVR's, we simply don't train on glass_refraction's data and treat it as our OOD task."),
]

# Locked 10-task set (2026-08-25): user-selected per category, not a uniform
# ratio -- see image_evaluator.py's LOCKED_TASKS comment for the full reasoning.
TASKS = [t for t in TASKS if t["full"] in LOCKED_TASKS]

TIER_META = {
    1: dict(label="Tier 1 &middot; Direct reuse", short="Fully compatible",
            desc="task_specific reads only the first and final frame in the original video evaluator, unchanged. ImageOnlyEvalMixin wraps the original class with no logic changes &mdash; validated to reproduce the video evaluator's score exactly (see per-card exact-match %)."),
    2: dict(label="Tier 2 &middot; Redefined, minor gap", short="One sub-metric replaced",
            desc="One sub-metric needed &ge;3 sampled frames and has been dropped, with the remaining weights renormalized &mdash; implemented and validated below (rank correlation vs. the video evaluator, not exact match, is the right comparison here)."),
    3: dict(label="Tier 3 &middot; Redefined, larger gap", short="Majority rewritten",
            desc="The majority of the original task_specific weight depended on tracking an object across many sampled frames and has been replaced with an honest 2-frame-only formulation (sometimes a single dominant sub-metric) &mdash; implemented and validated below, but treat scores as a coarser signal than Tier 1/2."),
}

CATEGORY_ORDER = ["Abstraction", "Knowledge", "Perception", "Spatiality", "Transformation"]

def esc(s):
    return htmlmod.escape(s or "", quote=True)

def weight_bar(weights, tier):
    total = sum(w for _, w, _ in weights)
    segs = []
    for i, (label, w, degraded) in enumerate(weights):
        pct = round(w / total * 1000) / 10
        cls = "seg-degraded" if degraded else f"seg-{i % 4}"
        segs.append(f'<div class="wbar-seg {cls}" style="width:{pct}%" title="{esc(label)} &middot; {w}%"></div>')
    return "".join(segs)

def weight_legend(weights):
    items = []
    for label, w, degraded in weights:
        cls = "legend-degraded" if degraded else ""
        mark = " &#9888;" if degraded else ""
        items.append(f'<span class="wlegend-item {cls}">{esc(label)} <b>{w}%</b>{mark}</span>')
    return "".join(items)

def fmt(x, pct=False):
    if x is None:
        return "&ndash;"
    return f"{x*100:.0f}%" if pct else f"{x:.3f}"

def validation_strip(t):
    v = VALIDATION.get(t["full"], {}).get("summary")
    if not v:
        return ""
    gap = v.get("image_discriminative_gap")
    gap_cls = "vgap-ok" if (gap is not None and gap > 0.15) else ("vgap-weak" if (gap is not None and gap >= 0) else "vgap-bad")
    exact = v.get("image_vs_video_exact_match_rate")
    spear = v.get("image_vs_video_spearman")
    if t["tier"] == 1:
        right = f'<span class="vstat-item">video match <b>{fmt(exact, pct=True)}</b></span>'
    else:
        right = f'<span class="vstat-item">rank corr. vs video <b>{fmt(spear)}</b></span>'
    return f'''
      <div class="card-validation">
        <span class="vstat-item">20-pair test: correct <b>{fmt(v.get('image_mean_correct'))}</b> / wrong <b>{fmt(v.get('image_mean_wrong'))}</b></span>
        <span class="vstat-item {gap_cls}">gap {fmt(gap)}</span>
        {right}
      </div>'''

def task_card(t):
    ex = EXAMPLES.get(t["full"], {})
    prompt = ex.get("prompt", "")
    if len(prompt) > 220:
        prompt = prompt[:217].rsplit(" ", 1)[0] + "…"
    first_uri = ex.get("first_uri") or ""
    final_uri = ex.get("final_uri") or ""
    split_cls = "pill-id" if t["split"] == "ID" else "pill-ood"
    return f'''
    <article class="card" data-tier="{t['tier']}" data-category="{esc(t['category'])}">
      <header class="card-head">
        <div class="card-id">{t['id']}</div>
        <div class="card-titles">
          <h3>{esc(t['name'])}</h3>
          <div class="card-tags">
            <span class="pill {split_cls}">{t['split']}</span>
            <span class="pill pill-cat">{esc(t['category'])}</span>
          </div>
        </div>
      </header>
      <div class="card-frames">
        <figure><img src="{first_uri}" alt="first frame" loading="lazy"><figcaption>first_frame</figcaption></figure>
        <div class="frame-arrow" aria-hidden="true">&#8594;</div>
        <figure><img src="{final_uri}" alt="final frame" loading="lazy"><figcaption>final_frame</figcaption></figure>
      </div>
      <p class="card-prompt">&ldquo;{esc(prompt)}&rdquo;</p>
      <div class="card-weights">
        <div class="wbar">{weight_bar(t['weights'], t['tier'])}</div>
        <div class="wlegend">{weight_legend(t['weights'])}</div>
      </div>
      <p class="card-rationale">{t['rationale']}</p>
      {validation_strip(t)}
    </article>'''

tier_sections = []
for tier in (1, 2, 3):
    meta = TIER_META[tier]
    cards = "".join(task_card(t) for t in TASKS if t["tier"] == tier)
    count = sum(1 for t in TASKS if t["tier"] == tier)
    tier_sections.append(f'''
    <section class="tier-section" id="tier-{tier}">
      <div class="tier-head tier-head-{tier}">
        <div class="tier-head-left">
          <span class="tier-num">{tier}</span>
          <div>
            <h2>{meta['label']}</h2>
            <p>{meta['desc']}</p>
          </div>
        </div>
        <div class="tier-count">{count} task{'s' if count != 1 else ''}</div>
      </div>
      <div class="card-grid">{cards}</div>
    </section>''')

n_tier = {t: sum(1 for x in TASKS if x["tier"] == t) for t in (1, 2, 3)}
n_id = sum(1 for x in TASKS if x["split"] == "ID")
n_ood = sum(1 for x in TASKS if x["split"] == "OOD")

tier1_exact = [VALIDATION[t["full"]]["summary"]["image_vs_video_exact_match_rate"] for t in TASKS if t["tier"] == 1]
tier1_exact_avg = sum(tier1_exact) / len(tier1_exact) if tier1_exact else 0
spear_vals = [VALIDATION[t["full"]]["summary"]["image_vs_video_spearman"] for t in TASKS
              if VALIDATION[t["full"]]["summary"]["image_vs_video_spearman"] is not None]
spear_avg = sum(spear_vals) / len(spear_vals) if spear_vals else 0
total_pairs = sum(VALIDATION[t["full"]]["summary"]["n_pairs"] for t in TASKS)
n_locked = len(TASKS)

HTML = f'''<title>VBVR Image-Only Evaluation</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {{
  --bg: #F1F4F2;
  --surface: #FFFFFF;
  --surface-alt: #E7EBE8;
  --ink: #12181A;
  --ink-soft: #4C5A5A;
  --ink-faint: #7C8B8A;
  --border: #D2DAD6;
  --accent: #2A3AA6;
  --accent-ink: #FFFFFF;
  --accent-soft: #E7E9F8;
  --safe: #1C8468;
  --safe-soft: #DEF0E9;
  --partial: #A9700E;
  --partial-soft: #F6E9D2;
  --risky: #B23A26;
  --risky-soft: #F7E1DC;
  --code-bg: #E9ECEA;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #14181A;
    --surface: #1B2124;
    --surface-alt: #21282B;
    --ink: #E9EEEC;
    --ink-soft: #ABBAB8;
    --ink-faint: #7E8D8B;
    --border: #303A3D;
    --accent: #8791E8;
    --accent-ink: #12141F;
    --accent-soft: #262B4B;
    --safe: #52BC9C;
    --safe-soft: #1B322C;
    --partial: #D9A548;
    --partial-soft: #362D1B;
    --risky: #E17E63;
    --risky-soft: #3A241F;
    --code-bg: #23292C;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #14181A;
  --surface: #1B2124;
  --surface-alt: #21282B;
  --ink: #E9EEEC;
  --ink-soft: #ABBAB8;
  --ink-faint: #7E8D8B;
  --border: #303A3D;
  --accent: #8791E8;
  --accent-ink: #12141F;
  --accent-soft: #262B4B;
  --safe: #52BC9C;
  --safe-soft: #1B322C;
  --partial: #D9A548;
  --partial-soft: #362D1B;
  --risky: #E17E63;
  --risky-soft: #3A241F;
  --code-bg: #23292C;
}}

* {{ box-sizing: border-box; }}
body {{
  background: var(--bg);
  color: var(--ink);
  font-family: "IBM Plex Sans", system-ui, sans-serif;
  margin: 0;
  padding: 0 0 5rem;
}}
::selection {{ background: var(--accent-soft); }}
.mono {{ font-family: "IBM Plex Mono", ui-monospace, monospace; }}

.wrap {{
  max-width: 1180px;
  margin: 0 auto;
  padding: 0 2rem;
}}

/* ---------- header ---------- */
header.top {{
  border-bottom: 1px solid var(--border);
  background: var(--surface);
  padding: 3rem 0 2.25rem;
}}
.eyebrow {{
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.78rem;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  color: var(--accent);
  margin: 0 0 0.9rem;
  display: flex;
  align-items: center;
  gap: 0.6rem;
}}
.eyebrow::before {{
  content: "";
  width: 1.6rem;
  height: 1px;
  background: var(--accent);
  display: inline-block;
}}
h1 {{
  font-size: clamp(1.8rem, 3.2vw, 2.5rem);
  line-height: 1.15;
  margin: 0 0 0.85rem;
  text-wrap: balance;
  font-weight: 700;
  letter-spacing: -0.01em;
}}
.lede {{
  max-width: 62ch;
  color: var(--ink-soft);
  font-size: 1.05rem;
  line-height: 1.6;
  margin: 0 0 1.75rem;
}}
.stat-row {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
}}
.stat {{
  border: 1px solid var(--border);
  border-radius: 3px;
  padding: 0.55rem 0.9rem;
  background: var(--surface-alt);
  font-size: 0.85rem;
  color: var(--ink-soft);
  display: flex;
  align-items: baseline;
  gap: 0.4rem;
}}
.stat b {{
  font-family: "IBM Plex Mono", monospace;
  font-size: 1rem;
  color: var(--ink);
}}
.stat.stat-safe b {{ color: var(--safe); }}
.stat.stat-partial b {{ color: var(--partial); }}
.stat.stat-risky b {{ color: var(--risky); }}

/* ---------- method strip ---------- */
.method {{
  padding: 2.5rem 0 0.5rem;
}}
.method h2 {{
  font-size: 1.05rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--ink-faint);
  font-weight: 600;
  margin: 0 0 1.1rem;
}}
.method-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1px;
  background: var(--border);
  border: 1px solid var(--border);
  border-radius: 4px;
  overflow: hidden;
}}
.method-step {{
  background: var(--surface);
  padding: 1.1rem 1.2rem;
  position: relative;
}}
.method-step .step-no {{
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.75rem;
  color: var(--accent);
  display: block;
  margin-bottom: 0.4rem;
}}
.method-step p {{
  margin: 0;
  font-size: 0.88rem;
  color: var(--ink-soft);
  line-height: 1.5;
}}
.method-step code {{
  font-family: "IBM Plex Mono", monospace;
  background: var(--code-bg);
  border-radius: 2px;
  padding: 0.05rem 0.3rem;
  font-size: 0.83em;
  color: var(--ink);
}}

/* ---------- tier legend nav ---------- */
.tier-nav {{
  display: flex;
  gap: 0.6rem;
  flex-wrap: wrap;
  margin: 1.75rem 0 0;
}}
.tier-nav a {{
  text-decoration: none;
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 0.4rem 0.9rem 0.4rem 0.7rem;
  font-size: 0.82rem;
  color: var(--ink-soft);
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  background: var(--surface);
}}
.tier-nav a .dot {{ width: 0.55rem; height: 0.55rem; border-radius: 50%; display: inline-block; }}
.tier-nav a[href="#tier-1"] .dot {{ background: var(--safe); }}
.tier-nav a[href="#tier-2"] .dot {{ background: var(--partial); }}
.tier-nav a[href="#tier-3"] .dot {{ background: var(--risky); }}

/* ---------- tier sections ---------- */
.tier-section {{ padding-top: 3rem; scroll-margin-top: 1.5rem; }}
.tier-head {{
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 1.5rem;
  border-bottom: 2px solid var(--border);
  padding-bottom: 1.1rem;
  margin-bottom: 1.6rem;
}}
.tier-head-left {{ display: flex; gap: 1rem; align-items: flex-start; }}
.tier-num {{
  font-family: "IBM Plex Mono", monospace;
  font-weight: 600;
  font-size: 1.1rem;
  width: 2.1rem;
  height: 2.1rem;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  flex: none;
  margin-top: 0.1rem;
}}
.tier-head-1 .tier-num {{ background: var(--safe-soft); color: var(--safe); }}
.tier-head-2 .tier-num {{ background: var(--partial-soft); color: var(--partial); }}
.tier-head-3 .tier-num {{ background: var(--risky-soft); color: var(--risky); }}
.tier-head h2 {{ margin: 0 0 0.3rem; font-size: 1.3rem; }}
.tier-head p {{ margin: 0; max-width: 60ch; color: var(--ink-soft); font-size: 0.92rem; line-height: 1.5; }}
.tier-count {{
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.82rem;
  color: var(--ink-faint);
  white-space: nowrap;
  padding-bottom: 0.2rem;
}}

/* ---------- cards ---------- */
.card-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 1.1rem;
}}
.card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 1.1rem;
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
}}
.card-head {{ display: flex; gap: 0.7rem; align-items: flex-start; }}
.card-id {{
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--accent);
  background: var(--accent-soft);
  border-radius: 3px;
  padding: 0.25rem 0.4rem;
  flex: none;
  margin-top: 0.1rem;
}}
.card-titles h3 {{
  margin: 0 0 0.35rem;
  font-size: 1rem;
  font-family: "IBM Plex Mono", monospace;
  font-weight: 600;
  word-break: break-word;
}}
.card-tags {{ display: flex; gap: 0.4rem; flex-wrap: wrap; }}
.pill {{
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  border-radius: 999px;
  padding: 0.18rem 0.55rem;
  font-weight: 600;
  border: 1px solid transparent;
}}
.pill-id {{ background: var(--safe-soft); color: var(--safe); }}
.pill-ood {{ background: var(--risky-soft); color: var(--risky); }}
.pill-cat {{ background: var(--surface-alt); color: var(--ink-soft); border-color: var(--border); }}

.card-frames {{
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: center;
  gap: 0.5rem;
  background: var(--surface-alt);
  border-radius: 4px;
  padding: 0.6rem;
}}
.card-frames figure {{ margin: 0; text-align: center; }}
.card-frames img {{
  width: 100%;
  aspect-ratio: 1;
  object-fit: contain;
  border-radius: 3px;
  background: var(--surface);
  border: 1px solid var(--border);
  display: block;
}}
.card-frames figcaption {{
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.62rem;
  color: var(--ink-faint);
  margin-top: 0.3rem;
}}
.frame-arrow {{ color: var(--ink-faint); font-size: 1.1rem; }}

.card-prompt {{
  font-size: 0.83rem;
  color: var(--ink-soft);
  font-style: italic;
  line-height: 1.45;
  margin: 0;
  border-left: 2px solid var(--border);
  padding-left: 0.65rem;
}}

.card-weights {{ display: flex; flex-direction: column; gap: 0.45rem; }}
.wbar {{
  display: flex;
  width: 100%;
  height: 0.5rem;
  border-radius: 3px;
  overflow: hidden;
  border: 1px solid var(--border);
}}
.wbar-seg {{ height: 100%; }}
.wbar-seg.seg-0 {{ background: var(--accent); opacity: 0.85; }}
.wbar-seg.seg-1 {{ background: var(--accent); opacity: 0.65; }}
.wbar-seg.seg-2 {{ background: var(--accent); opacity: 0.45; }}
.wbar-seg.seg-3 {{ background: var(--accent); opacity: 0.28; }}
.wbar-seg.seg-degraded {{
  background: repeating-linear-gradient(135deg, var(--risky), var(--risky) 3px, transparent 3px, transparent 6px);
}}
.wlegend {{ display: flex; flex-wrap: wrap; gap: 0.3rem 0.6rem; }}
.wlegend-item {{
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.68rem;
  color: var(--ink-faint);
}}
.wlegend-item b {{ color: var(--ink-soft); font-weight: 600; }}
.wlegend-item.legend-degraded {{ color: var(--risky); }}
.wlegend-item.legend-degraded b {{ color: var(--risky); }}

.card-rationale {{
  font-size: 0.82rem;
  color: var(--ink-soft);
  line-height: 1.55;
  margin: 0;
  padding-top: 0.65rem;
  border-top: 1px dashed var(--border);
}}
.card-rationale code {{
  font-family: "IBM Plex Mono", monospace;
  background: var(--code-bg);
  border-radius: 2px;
  padding: 0.02rem 0.28rem;
  font-size: 0.92em;
}}

.card-validation {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem 0.7rem;
  background: var(--surface-alt);
  border-radius: 4px;
  padding: 0.55rem 0.7rem;
  margin-top: -0.15rem;
}}
.vstat-item {{
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.7rem;
  color: var(--ink-faint);
  white-space: nowrap;
}}
.vstat-item b {{ color: var(--ink-soft); font-weight: 600; }}
.vstat-item.vgap-ok b {{ color: var(--safe); }}
.vstat-item.vgap-weak b {{ color: var(--partial); }}
.vstat-item.vgap-bad b {{ color: var(--risky); }}

/* ---------- footer ---------- */
footer.page-foot {{
  margin-top: 4rem;
  padding-top: 2rem;
  border-top: 1px solid var(--border);
}}
footer.page-foot p {{
  color: var(--ink-faint);
  font-size: 0.82rem;
  max-width: 70ch;
  line-height: 1.6;
}}
footer.page-foot code {{
  font-family: "IBM Plex Mono", monospace;
  background: var(--code-bg);
  border-radius: 2px;
  padding: 0.05rem 0.3rem;
}}

@media (max-width: 620px) {{
  .wrap {{ padding: 0 1.1rem; }}
  header.top {{ padding: 2.2rem 0 1.75rem; }}
  .tier-head {{ flex-direction: column; align-items: flex-start; gap: 0.5rem; }}
}}
</style>

<header class="top">
  <div class="wrap">
    <p class="eyebrow">VBVR-EvalKit &middot; InternVL-U adaptation</p>
    <h1>Image-only evaluation strategy for the 15 locked VBVR tasks</h1>
    <p class="lede">InternVL-U produces a single <code class="mono">final_frame</code>, not a video trajectory. These {n_locked} tasks are the locked set actually used for InternVL-U training/eval (out of 15 implemented in <code class="mono">image_evaluator.py</code> &mdash; the other 5 were dropped per-category, either as redundant with a similar ID task or as too eval-problematic to trust). Each has a dedicated image-only evaluator, tested on 20 pairs (5 correctly matched + 15 deliberately mismatched, {total_pairs} evaluations total, 0 errors) against the original video evaluator run under the same condition. ID/OOD below is <b>our own</b> train/test split, not VBVR-Bench's official one &mdash; since we pick which tasks InternVL-U actually trains on, a task can live in VBVR-Bench's own In-Domain_50 folder and still be one we exclude from training and hold out as OOD.</p>
    <div class="stat-row">
      <div class="stat stat-safe">{n_tier[1]} <b>{n_tier[1]}</b> Tier 1 &mdash; direct reuse</div>
      <div class="stat stat-partial">{n_tier[2]} <b>{n_tier[2]}</b> Tier 2 &mdash; redefined</div>
      <div class="stat stat-risky">{n_tier[3]} <b>{n_tier[3]}</b> Tier 3 &mdash; redefined</div>
      <div class="stat">Split <b>{n_id} ID / {n_ood} OOD</b></div>
      <div class="stat">Tier&nbsp;1 exact match <b>{tier1_exact_avg*100:.0f}%</b></div>
      <div class="stat">Redefined avg rank corr. <b>{spear_avg:.2f}</b></div>
    </div>
  </div>
</header>

<div class="wrap">
  <section class="method">
    <h2>How each task was built and checked</h2>
    <div class="method-grid">
      <div class="method-step"><span class="step-no">01</span><p><b>Tier 1</b> (9 tasks): feed a synthetic 2-frame video <code>[first_frame, final_frame]</code> to the <b>unmodified</b> video evaluator class via <code>ImageOnlyEvalMixin</code> &mdash; no logic rewrite, since these only ever read the first/last frame.</p></div>
      <div class="method-step"><span class="step-no">02</span><p><b>Tier 2/3</b> (6 tasks): override <code>_evaluate_task_specific</code> per class, reusing the original evaluator's own detector helpers (e.g. <code>_detect_keys</code>, <code>_track_ball_positions</code>) on just 2 frames, dropping sub-metrics that are mathematically undefined from an endpoint pair and renormalizing the rest &mdash; documented per class in <code>image_evaluator.py</code>.</p></div>
      <div class="method-step"><span class="step-no">03</span><p><b>Validate</b>: build 20 test pairs per task from the 5 downloaded GT samples (5 correct + 15 index-shifted mismatches), run both the video evaluator and the image evaluator on each pair, and compare &mdash; exact match expected for Tier&nbsp;1, rank correlation for the redefined tiers.</p></div>
      <div class="method-step"><span class="step-no">04</span><p><b>Caught a real bug</b>: <code>G-18_grid_shortest_path</code> was manually classified Tier&nbsp;1, but validation showed only 20% exact match even on correct pairs. A bare <code>for frame in video_frames:</code> loop &mdash; a third disguise beyond indexed access and passed-as-argument calls &mdash; had been missed by code review. Reclassified to Tier&nbsp;2 and fixed; see its card below.</p></div>
    </div>
    <nav class="tier-nav">
      <a href="#tier-1"><span class="dot"></span>Tier 1 &middot; direct reuse</a>
      <a href="#tier-2"><span class="dot"></span>Tier 2 &middot; redefined, minor gap</a>
      <a href="#tier-3"><span class="dot"></span>Tier 3 &middot; redefined, larger gap</a>
    </nav>
  </section>

  {"".join(tier_sections)}

  <footer class="page-foot">
    <p>Implementation: <code>Evaluation/VBVR-CustomEval/evaluators/image_evaluator.py</code> (<code>ImageOnlyEvalMixin</code> + 15 task classes, registered in <code>TASK_IMAGE_EVALUATOR_MAP</code>) &mdash; imports the vendored <code>vbvr_bench</code> package from the sibling <code>Evaluation/VBVR-EvalKit/</code> clone rather than editing it in place. Validation harness: <code>Evaluation/VBVR-CustomEval/validation/test_harness.py</code>, full per-pair results in <code>validation_results.json</code>. Weight bars show each sub-metric's share of the original <code>task_specific</code> dimension; hatched red segments are the ones dropped or redefined in image-only mode. The 20-pair test's "wrong" pairing is a weak adversary for a few tasks whose 5 GT samples happen to look similar to each other (gap &asymp; 0 despite 100% exact match with the video evaluator) &mdash; treat the exact-match / rank-correlation columns as the primary signal, the discriminative gap as secondary. Example frames and prompts are sample <code>00000</code> from each task's downloaded VBVR-Bench ground truth.</p>
  </footer>
</div>
'''

with open(OUT_PATH, "w") as f:
    f.write(HTML)

print("wrote", OUT_PATH, len(HTML), "chars")
