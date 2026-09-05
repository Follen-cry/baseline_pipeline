#!/usr/bin/env python3
"""Generate the VBVR scoring-rules glossary artifact HTML."""
import json
import html as htmlmod

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
EXAMPLES_PATH = _os.path.join(_HERE, "vbvr_task_examples.json")
OUT_PATH = _os.path.join(_HERE, "vbvr_scoring_rules.html")

with open(EXAMPLES_PATH) as f:
    EXAMPLES = json.load(f)

import sys as _sys
_sys.path.insert(0, _os.path.join(_HERE, "..", "evaluators"))
from image_evaluator import LOCKED_TASKS  # noqa: E402
from llm_judge import RUBRICS, build_judge_instruction  # noqa: E402

# Sample prompt.txt used to render a concrete example of the judge instruction
# per task (rubric text itself is task-level, not sample-specific, but a real
# prompt makes the rendered example legible rather than a placeholder).
_SAMPLE_PROMPTS = {
    name: EXAMPLES.get(full, {}).get("prompt", "")
    for name, full in [
        ("key_door_matching", "G-45_key_door_matching_data-generator"),
    ]
}


def esc(s):
    return htmlmod.escape(s or "", quote=True)


# Each task: id, full generator name, name, split, short task description (what
# the model must produce), criteria = [(label, weight%, rule text, degraded_bool)],
# and an optional "image_note" explaining how image-only mode adapts the rule.
CATEGORIES = [
    dict(name="Abstraction", blurb="Find the rule from an example transformation, then apply it to a new case.",
         tasks=[
        dict(id="O-12", full="O-12_shape_color_then_scale_data-generator", name="shape_color_then_scale", split="ID",
             task_desc="Two-step analogy (A&rarr;B&rarr;C :: D&rarr;?&rarr;?): first the shape's color changes, then its size scales.",
             criteria=[
                ("two_step_rule", 30, "Final shape's color AND area both land within tolerance of GT (area ratio &gt;0.7, hue difference &lt;20&deg;) &mdash; both right scores full credit, only one right scores partial, neither scores zero.", False),
                ("first_step", 25, "Mean pixel difference between the generated and GT final frame &mdash; effectively “how close is the overall picture,” despite the name.", False),
                ("second_step", 25, "Area ratio between generated and GT final shape &mdash; did the scale step land in the right ballpark.", False),
                ("sequence", 20, "The weaker of first_step and second_step &mdash; both steps have to hold up, not just one.", False),
             ]),
        dict(id="O-13", full="O-13_shape_outline_then_move_data-generator", name="shape_outline_then_move", split="ID",
             task_desc="Two-step analogy: first the shape's outline/fill style changes, then it moves vertically.",
             criteria=[
                ("two_step_rule", 30, "Recognizes that style changes before position &mdash; final frame checked against the expected style+position combination.", False),
                ("first_step", 25, "Style (outline vs. filled) changed correctly while position stays centered.", False),
                ("second_step", 25, "Vertical position changed correctly while the new style is preserved.", False),
                ("sequence", 20, "Consistency of doing style-then-move in the right order.", False),
             ]),
        dict(id="O-11", full="O-11_shape_color_then_move_data-generator", name="shape_color_then_move", split="OOD",
             task_desc="Row-completion analogy: top row (A, B, C) is fixed; bottom row keeps D and adds E, F colored like B.",
             criteria=[
                ("first_row_preservation", 40, "The three top-row reference shapes must stay pixel-identical &mdash; touching them is an instant penalty.", False),
                ("second_row_completion", 35, "D stays in place, and two new shapes E and F appear alongside it.", False),
                ("color_accuracy", 20, "E and F are colored to match B, not any other reference shape.", False),
                ("shape_count", 5, "Final frame has exactly 6 shapes total (3 top + 3 bottom).", False),
             ]),
    ]),
    dict(name="Knowledge", blurb="Apply a physical or geometric law (optics, gravity) that has to be simulated correctly, not just recognized.",
         tasks=[
        dict(id="O-19", full="O-19_mirror_reflection_data-generator", name="mirror_reflection", split="ID",
             task_desc="Given an incident light ray and a mirror, draw the reflected ray per the law of reflection (angle in = angle out).",
             criteria=[
                ("reflection_angle", 40, "Detects line segments in the final frame (Hough transform) and checks the reflected ray's angle equals the incident angle.", False),
                ("symmetry", 30, "The reflected ray is mirror-symmetric to the incident ray about the mirror's normal.", False),
                ("ray_extension", 20, "The ray is drawn all the way to the image boundary, not left short.", False),
                ("starting_point", 10, "The reflected ray originates from the same point where the incident ray met the mirror.", False),
             ]),
        dict(id="O-15", full="O-15_ball_bounces_given_time_data-generator", name="ball_bounces_given_time", split="ID",
             task_desc="Given a ball's start position and velocity, show where it ends up after bouncing off walls N times.",
             criteria=[
                ("final_position", 100, "Ball's final position is close to GT's final position (reuses the original ball detector, run on the single final frame).", False),
             ],
             image_note="<b>Revised 2026-08-25</b> &mdash; this task briefly carried a second criterion, <code>path_shape</code> (IoU between generated/GT drawn bounce-path masks), added because VBVR-Bench's frozen GT samples render the entire bounce path as a static orange polyline in final_frame.png, jointly covering bounce_count/physics/trajectory in one check (validated at correct=1.000 / wrong=0.004-0.009). But generating fresh samples from this task's own DataFactory generator the same day showed its visual style has since changed upstream: final_frame.png now shows only the ball resting at its final position, with no path drawn at all. path_shape was removed rather than kept as a metric no longer supported by what this task's data actually looks like. final_position (unchanged) is now the sole task_specific score, at 100% weight. bounce_count/physics/trajectory/smoothness (all 4 of the original video-mode sub-metrics) remain unrecoverable from a single frame under the current visual style and are not replaced &mdash; and no VLM judge is used for this task, since final_position is already a plain deterministic check."),
        dict(id="O-18", full="O-18_glass_refraction_data-generator", name="glass_refraction", split="OOD",
             task_desc="Given an incident light ray and glass refractive index, draw the refracted ray per Snell's law.",
             criteria=[
                ("snells_law", 50, "Detects the incident and refracted rays (line detection on the final frame) and checks the refraction angle matches Snell's law for the stated refractive index.", False),
                ("ray_direction", 30, "The refracted ray bends toward/away from the normal in the physically correct direction.", False),
                ("ray_completeness", 15, "The refracted ray is drawn all the way to the image edge, not left short.", False),
                ("scene_fidelity", 5, "The glass boundary and incident ray are unchanged from the first frame.", False),
             ],
             image_note="Replaces gravity_physics (2026-08-25): that task's own generator doesn't clamp the falling object at ground level, so most GT samples simulate it to a negative height and render an empty final_frame (confirmed: 4 of 5 downloaded samples, by computing h(3) for each &mdash; all negative). Every scoring method inherits that defect; it isn't fixable by changing the evaluator. glass_refraction has no such failure mode and needs no image-only adaptation at all (Tier 1). It ships in VBVR-Bench's own In-Domain_50 folder, but since we choose our own train/test split rather than VBVR's, we simply exclude its data from training and treat it as our OOD task &mdash; Knowledge stays a clean 2 ID / 1 OOD like every other category."),
    ]),
    dict(name="Perception", blurb="Read off a structured fact about the scene directly &mdash; sort, place, or arrange, no inference needed.",
         tasks=[
        dict(id="G-3", full="G-3_stable_sort_data-generator", name="stable_sort", split="ID",
             task_desc="Sort shapes by type, then by size within each type, arranged left to right.",
             criteria=[
                ("classification", 30, "Shapes of the same type end up grouped adjacent to each other.", False),
                ("order", 30, "Within each group, shapes run small-to-large left to right.", False),
                ("fidelity", 30, "Every shape's type, size and color still matches what was in the first frame &mdash; nothing added, removed, or recolored.", False),
                ("layout", 10, "All shapes sit on the same horizontal line.", False),
             ]),
        dict(id="G-5", full="G-5_multi_object_placement_data-generator", name="multi_object_placement", split="ID",
             task_desc="Move each colored object onto the star marker of the matching color.",
             criteria=[
                ("color_matching", 30, "Each object ends up within 30px of the star marker sharing its color.", False),
                ("alignment", 25, "Average distance from each object's center to its matching star &mdash; precision beyond just “close enough.”", False),
                ("path", 20, "Movement across the video is smooth (low frame-to-frame variance) rather than teleporting.", True),
                ("fidelity", 15, "Object count and total area preserved between first and final frame.", False),
                ("star_invariance", 10, "The star markers themselves never move.", False),
             ],
             image_note="path can't be measured from an endpoint pair &mdash; with only 2 points, “motion smoothness” is mathematically trivial (a straight line has zero variance by definition). <b>Confirmed unrecoverable</b> (2026-08-25): checked final_frame.png for a drawn motion trail (the kind ball_bounces_given_time's GT briefly had before its generator's visual style changed) &mdash; there is none here, objects just sit at their resting position, no pixel evidence of how they got there. Deliberately dropped rather than handed to a judge (a judge shown one frame has the same zero signal). Other 4 criteria renormalized to 100%."),
        dict(id="O-65", full="O-65_animal_size_sorting_data-generator", name="animal_size_sorting", split="OOD",
             task_desc="Sort animals from smallest to largest, left to right.",
             criteria=[
                ("sorting", 40, "Final left-to-right order matches ascending size.", False),
                ("alignment", 30, "All animals sit on the same baseline.", False),
                ("fidelity", 20, "Each animal's size and appearance is preserved from the first frame.", False),
                ("completeness", 10, "No animal is missing from the final arrangement.", False),
             ]),
    ]),
    dict(name="Spatiality", blurb="Navigate or reason about positions in a 2D layout &mdash; grids, mazes, keys and doors.",
         tasks=[
        dict(id="G-18", full="G-18_grid_shortest_path_data-generator", name="grid_shortest_path", split="ID",
             task_desc="Move an agent through a grid to a goal cell via the shortest legal path, avoiding obstacles.",
             criteria=[
                ("path_optimal", 50, "Total path length (summed across every sampled frame) compared against GT's path length.", True),
                ("completion", 25, "Agent's final position lands within 50px of GT's final position (or the marked endpoint).", False),
                ("movement", 15, "No diagonal jumps &mdash; only up/down/left/right steps.", False),
                ("fidelity", 10, "The agent sprite itself is still detectable/unchanged.", False),
             ],
             image_note="Caught by validation, not by code review: path_optimal sums position deltas via a bare <code>for frame in video_frames:</code> loop, invisible to a grep for indexed or passed-as-argument access. With 2 frames the generated path collapses to a straight line while GT's path (from the real video) stays the true winding length, biasing the ratio low &mdash; only 20% exact-match with the video evaluator on correct pairs exposed it. Dropped and renormalized across completion/movement/fidelity."),
        dict(id="O-39", full="O-39_maze_data-generator", name="maze", split="OOD",
             task_desc="Navigate an agent through a maze from the start to the end marker.",
             criteria=[
                ("path_validity", 45, "The drawn path never crosses a wall and stays continuous.", False),
                ("path_completeness", 30, "The path actually connects start to end &mdash; no gaps.", False),
                ("navigation_accuracy", 20, "Only adjacent-cell moves, no jumps.", False),
                ("element_preservation", 5, "Maze walls/markers are unchanged from the first frame.", False),
             ]),
        dict(id="G-45", full="G-45_key_door_matching_data-generator", name="key_door_matching", split="ID",
             task_desc="Move an agent to pick up a key, then to the door matching that key's color.",
             criteria=[
                ("agent_movement", 30, "Agent must physically move &gt;30px from its start &mdash; staying put is an automatic 0 for the whole task.", False),
                ("key_collected", 35, "Agent's tracked path came within 50px of a key AND that key visually disappeared afterward.", True),
                ("door_reached", 25, "Agent's tracked path (in the second half of the video) came close to a door whose color matches the collected key.", True),
                ("sequence", 10, "The key visit happened before the door visit.", True),
             ],
             image_note="Important: none of these four checks ever compare against the GT's specific key/door layout &mdash; it's a self-consistency check on the generated video alone (“did a plausible key-then-door sequence happen”), regardless of whether it's *this puzzle's* correct key/door pairing. That's true in full video mode too, not something image-only mode introduced. Our image-mode version keeps that same self-consistency spirit: key_collected becomes “did a key visible in frame 1 disappear by the final frame,” door_reached becomes “is the agent's final position near a matching-color door,” sequence is dropped (unrecoverable from 2 static frames) and the rest renormalized."),
    ]),
    dict(name="Transformation", blurb="Simulate a spatial edit &mdash; shift, rotate, or reconfigure &mdash; and preserve everything that shouldn't change.",
         tasks=[
        dict(id="O-44", full="O-44_rotation_puzzle_data-generator", name="rotation_puzzle", split="ID",
             task_desc="Rotate L-shaped pipe tiles in a 2&times;2 grid so every pipe opening connects into one continuous path.",
             criteria=[
                ("path_connection", 40, "Every pipe opening lines up with a neighboring tile's opening &mdash; the full loop connects.", False),
                ("rotation_accuracy", 30, "Tiles were rotated in clean 90&deg; increments, not partial angles.", False),
                ("position_preservation", 20, "Tiles stayed in their original grid cell &mdash; only rotation changed, not placement.", False),
                ("alignment_precision", 10, "Pipe openings line up precisely at tile edges, not offset.", False),
             ]),
        dict(id="O-36", full="O-36_grid_shift_data-generator", name="grid_shift", split="ID",
             task_desc="Move every colored block in an N&times;N grid the same direction, by the same number of steps, simultaneously.",
             criteria=[
                ("direction_correctness", 30, "All blocks moved in the instructed direction.", False),
                ("step_accuracy", 30, "All blocks moved the exact number of steps specified.", False),
                ("synchronization", 20, "Sampled across many frames: all blocks display the same displacement at every point in time (moving together, not staggered).", True),
                ("position_precision", 15, "Final block positions match GT positions precisely.", False),
                ("completeness", 5, "Same number of blocks, same pattern, before and after.", False),
             ],
             image_note="synchronization needs &ge;3 sampled frames to compare displacement over time; with only 2 it's undefined rather than measurable. <b>Confirmed unrecoverable</b> (2026-08-25): final_frame.png shows only each block's resting cell, no trail or timing cue. Deliberately dropped rather than handed to a judge. Remaining 4 criteria (already first/final-frame checks) renormalized to 100%."),
        dict(id="O-6", full="O-6_2d_geometric_transformation_data-generator", name="2d_geometric_transformation", split="OOD",
             task_desc="Rotate a shape around a marked pivot point until it aligns with a dashed target outline.",
             criteria=[
                ("rotation_center", 30, "Shape's center is tracked across many sampled frames and fit to a circular arc &mdash; confirms it rotated about a fixed point rather than sliding.", True),
                ("rotation_angle", 35, "Final shape's orientation (fitted ellipse angle) matches the target outline's orientation.", False),
                ("position_alignment", 25, "Final shape's center is close to the target outline's center.", False),
                ("shape_fidelity", 10, "Shape's area is preserved (not accidentally resized during rotation).", False),
             ],
             image_note="rotation_center needs &ge;3 sampled centers to fit an arc; with 2 frames the original code hard-returns 0.0 &mdash; a fixed penalty baked into every score, not a neutral skip. <b>Confirmed unrecoverable</b> (2026-08-25): final_frame.png shows only the end orientation plus a static pivot dot, no arc or sweep is drawn. Deliberately dropped rather than handed to a judge. Renormalized across rotation_angle/position_alignment/shape_fidelity."),
    ]),
]

# Locked 10-task set (2026-08-25): all 15 stay listed below (every one has a
# working evaluator), but only 10 are actually used for InternVL-U
# training/eval right now -- not a uniform per-category ratio. The other 5
# are parked, each for a specific documented reason, not just "not gotten to
# yet". See image_evaluator.py's LOCKED_TASKS comment for the source of truth.
PARKED_REASONS = {
    "shape_color_then_scale": "Parked 2026-08-25: this and shape_outline_then_move are both Abstraction ID candidates with the same two-step-analogy structure -- redundant with each other, so neither is in the current set (Abstraction runs OOD-only for now).",
    "shape_outline_then_move": "Parked 2026-08-25: redundant with shape_color_then_scale (same two-step-analogy structure) -- neither Abstraction ID task is in the current set (Abstraction runs OOD-only for now).",
    "mirror_reflection": "Parked 2026-08-25: Knowledge's one ID slot went to ball_bounces_given_time instead -- a deliberate choice to keep tracking that task's Known Issue, not an oversight (mirror_reflection itself has no known problems).",
    "grid_shortest_path": "Parked 2026-08-25: one of the two most eval-problematic ID tasks in Spatiality (Tier 2, only 0.22 rank-correlation with the video evaluator after the path_optimal fix) -- Spatiality runs OOD-only for now.",
    "key_door_matching": "Parked 2026-08-25, and not just deprioritized -- dropped because its own GT data was found ambiguous by two independent methods (rule-based key-disappearance detector and the VLM judge), not only its evaluator. See README Known Issues.",
}
# Implementation detail per task: which detector function(s) actually run,
# verified 2026-08-25 by re-reading the current source (both the vendored
# vbvr_bench evaluator and, where overridden, image_evaluator.py) rather than
# trusting this file's own prior text -- exactly the kind of drift that
# caused the ball_bounces_given_time inconsistency found earlier. Only tasks
# with directly-verified detail are listed; no entry means not yet re-checked
# this pass.
IMPL_DETAILS = {
    "ball_bounces_given_time": "<code>final_position</code> reuses <code>_track_ball_positions</code> (Hough circle detection, dark-blob contour fallback), run on the single final frame only.",
    "stable_sort": "<code>_detect_shapes</code>: HSV range (0,30,30)&ndash;(180,255,255) (anything not white/black) + contour extraction, area&gt;500px filter, to recover each shape's type/position/color for all 4 sub-scores.",
    "multi_object_placement": "<code>_detect_colored_objects</code> / <code>_detect_star_markers</code>: per-color HSV ranges (red/blue/green/yellow) + contour area filters (objects &gt;300px; stars 100&ndash;2000px with &ge;8 approxPolyDP vertices). Match thresholds: 30px for color_matching, 20px for star_invariance.",
    "grid_shift": "<code>_detect_colored_blocks</code> feeds a gate &mdash; <code>_evaluate_completeness</code> + <code>_evaluate_block_pattern_preservation</code> must both score &gt;0.5 (\"block_preserved\") before direction_correctness/step_accuracy/position_precision are scored at all; otherwise those 3 hard-zero and only completeness counts.",
    "2d_geometric_transformation": "<code>_detect_main_shape</code>: HSV saturation&gt;50 mask isolates the colored shape from the grayscale background. <code>_detect_target_outline</code>: Canny edges pick out the dashed target outline. Orientation via <code>cv2.fitEllipse</code> angle.",
    "glass_refraction": "<code>_detect_lines</code>: Canny(50,150) + HoughLinesP(threshold=50) for general line detection. <code>_detect_red_line</code>: HoughLinesP(threshold=30) on a red color mask specifically for the refracted ray.",
    "maze": "<code>_detect_path_markers</code> (orange/yellow mask) vs <code>_detect_walls</code> (black mask) &mdash; path_validity checks the drawn path pixels don't overlap the wall mask.",
    "shape_color_then_move": "<code>_detect_shapes_with_info</code> (shape/color/position) is called on both frame 0 and frame -1; first_row_preservation diffs the two directly, second_row_completion/color_accuracy compare the new shapes against it.",
    "rotation_puzzle": "<code>_detect_blue_pipes</code> isolates the pipe-tile shapes in the 2&times;2 grid via color mask; path_connection/rotation_accuracy compare tile connectivity and orientation between frame 0 and frame -1.",
    "animal_size_sorting": "<code>_detect_animals</code>: contour-based shape detection per animal; <code>_evaluate_sorting</code> / <code>_evaluate_alignment</code> then check final left-to-right order and baseline position.",
}

for _cat in CATEGORIES:
    for _t in _cat["tasks"]:
        _t["active"] = _t["full"] in LOCKED_TASKS
        _t["parked_reason"] = PARKED_REASONS.get(_t["name"])
        _t["impl_detail"] = IMPL_DETAILS.get(_t["name"])


def weight_bar(criteria):
    """
    Renders ONLY the criteria actually used for scoring right now (dropped
    ones are excluded from the bar entirely, not shown as a discounted/hatched
    slice of it) -- their original weights renormalized so the bar still sums
    to 100%. This matches image_evaluator.py's real IMAGE_TASK_WEIGHTS, not
    the original video-mode weights.
    """
    active = [(label, w) for label, w, _, degraded in criteria if not degraded]
    total_active = sum(w for _, w in active) or 1
    segs = []
    for i, (label, w) in enumerate(active):
        pct = round(w / total_active * 1000) / 10
        segs.append(f'<div class="wbar-seg seg-{i % 5}" style="width:{pct}%" title="{esc(label)} &middot; {pct}% (of current scoring)"></div>')
    return "".join(segs)


def criteria_list(criteria):
    active = [(label, w) for label, w, _, degraded in criteria if not degraded]
    total_active = sum(w for _, w in active) or 1
    items = []
    for label, w, rule, degraded in criteria:
        if degraded:
            weight_html = f'<span class="crit-weight crit-weight-dropped">{w}%</span>'
            flag = ' <span class="crit-flag crit-flag-dropped" title="Confirmed unrecoverable from a single frame; excluded from scoring, not just discounted">&#10007; not scored</span>'
        else:
            renorm = round(w / total_active * 1000) / 10
            weight_html = f'<span class="crit-weight">{renorm}%</span>' if abs(renorm - w) > 0.05 else f'<span class="crit-weight">{w}%</span>'
            flag = ''
        items.append(f'''
        <li class="crit-item{' crit-degraded' if degraded else ''}">
          <div class="crit-head"><code>{esc(label)}</code>{weight_html}{flag}</div>
          <p>{rule}</p>
        </li>''')
    return "".join(items)


def scoring_assembly_note(t):
    return f'''
        <div class="assembly-note">
          <b>How this fits the full score:</b> the criteria above are the internal breakdown of the
          <code>task_specific</code> dimension, which is <b>25%</b> of VBVR's standard 5-dimension score
          (<code>first_frame_consistency</code> 15% / <code>final_frame_accuracy</code> 35% /
          <code>temporal_smoothness</code> 15% / <code>visual_quality</code> 10% / <code>task_specific</code> 25%).
          Image-only mode drops <code>temporal_smoothness</code> (undefined for 2 frames) and renormalizes the
          rest. Our validation harness calls <code>evaluate(..., task_specific_only=True)</code>, which reports
          <b>only</b> <code>task_specific</code> as the score &mdash; so in every number shown in the companion
          <i>Image-Only Evaluation</i> artifact, the criteria above <b>are</b> the whole score, not 25% of it.
        </div>'''


def judge_prompt_block(t):
    if t["name"] not in RUBRICS:
        return ""
    sample_prompt = _SAMPLE_PROMPTS.get(t["name"]) or "<task prompt from prompt.txt goes here>"
    instruction = build_judge_instruction(t["name"], sample_prompt)
    return f'''
        <div class="judge-block">
          <div class="judge-head">&#129504; Also scored by VLM judge (<code>qwen3-vl-30b-fp8</code>, local vLLM) &mdash; supplements, doesn't replace, the rule-based score above</div>
          <p class="judge-sub">Exact instruction text sent alongside the 3 images (start frame / candidate / GT), reproduced live from <code>llm_judge.build_judge_instruction()</code> using this task's real sample-00000 prompt:</p>
          <pre class="judge-prompt">{esc(instruction)}</pre>
        </div>'''


def task_entry(t):
    ex = EXAMPLES.get(t["full"], {})
    first_uri = ex.get("first_uri") or ""
    final_uri = ex.get("final_uri") or ""
    split_cls = "pill-id" if t["split"] == "ID" else "pill-ood"
    note_html = f'<p class="task-note"><b>In image-only mode:</b> {t["image_note"]}</p>' if t.get("image_note") else ""
    if t["active"]:
        status_html = '<span class="pill pill-active">&#9679; Active</span>'
    else:
        status_html = '<span class="pill pill-parked">&#9675; Parked</span>'
    parked_html = f'<p class="task-note task-note-parked">{t["parked_reason"]}</p>' if t.get("parked_reason") else ""
    impl_html = f'<p class="task-note task-note-impl"><b>Implementation:</b> {t["impl_detail"]}</p>' if t.get("impl_detail") else ""
    return f'''
    <article class="task{'' if t["active"] else ' task-parked'}">
      <div class="task-frames">
        <figure><img src="{first_uri}" alt="first frame" loading="lazy"></figure>
        <div class="frame-arrow" aria-hidden="true">&#8594;</div>
        <figure><img src="{final_uri}" alt="final frame" loading="lazy"></figure>
      </div>
      <div class="task-body">
        <header class="task-head">
          <span class="task-id">{t['id']}</span>
          <h3>{esc(t['name'])}</h3>
          <span class="pill {split_cls}">{t['split']}</span>
          {status_html}
        </header>
        <p class="task-desc">{t['task_desc']}</p>
        <div class="wbar">{weight_bar(t['criteria'])}</div>
        <ul class="crit-list">{criteria_list(t['criteria'])}</ul>
        {scoring_assembly_note(t)}
        {impl_html}
        {note_html}
        {parked_html}
        {judge_prompt_block(t)}
      </div>
    </article>'''


category_sections = []
for cat in CATEGORIES:
    entries = "".join(task_entry(t) for t in cat["tasks"])
    category_sections.append(f'''
    <section class="cat-section" id="cat-{cat['name'].lower()}">
      <div class="cat-head">
        <h2>{cat['name']}</h2>
        <p>{cat['blurb']}</p>
      </div>
      <div class="task-list">{entries}</div>
    </section>''')

n_degraded = sum(1 for cat in CATEGORIES for t in cat["tasks"] for c in t["criteria"] if c[3])
n_criteria = sum(len(t["criteria"]) for cat in CATEGORIES for t in cat["tasks"])
n_tasks = sum(len(cat["tasks"]) for cat in CATEGORIES)
n_active = sum(1 for cat in CATEGORIES for t in cat["tasks"] if t["active"])
n_parked = n_tasks - n_active
n_judged = sum(1 for cat in CATEGORIES for t in cat["tasks"] if t["name"] in RUBRICS)

HTML = f'''<title>VBVR Scoring Rules</title>
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
  --accent: #2E6E5E;
  --accent-soft: #E1EDE9;
  --warn: #A9700E;
  --warn-soft: #F6E9D2;
  --code-bg: #E9ECEA;
  --cat-abstraction: #6B4FA0;
  --cat-knowledge: #2A5CA6;
  --cat-perception: #A8395A;
  --cat-spatiality: #1C6B5E;
  --cat-transformation: #B06A1A;
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
    --accent: #5FBFA3;
    --accent-soft: #1B322C;
    --warn: #D9A548;
    --warn-soft: #362D1B;
    --code-bg: #23292C;
    --cat-abstraction: #A98FE0;
    --cat-knowledge: #7DA8E8;
    --cat-perception: #E37FA0;
    --cat-spatiality: #5FBFA3;
    --cat-transformation: #E0A356;
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
  --accent: #5FBFA3;
  --accent-soft: #1B322C;
  --warn: #D9A548;
  --warn-soft: #362D1B;
  --code-bg: #23292C;
  --cat-abstraction: #A98FE0;
  --cat-knowledge: #7DA8E8;
  --cat-perception: #E37FA0;
  --cat-spatiality: #5FBFA3;
  --cat-transformation: #E0A356;
}}

* {{ box-sizing: border-box; }}
body {{
  background: var(--bg);
  color: var(--ink);
  font-family: "IBM Plex Sans", system-ui, sans-serif;
  margin: 0;
  padding: 0 0 5rem;
}}
.mono {{ font-family: "IBM Plex Mono", ui-monospace, monospace; }}
.wrap {{ max-width: 1080px; margin: 0 auto; padding: 0 2rem; }}

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
  display: flex; align-items: center; gap: 0.6rem;
}}
.eyebrow::before {{ content: ""; width: 1.6rem; height: 1px; background: var(--accent); display: inline-block; }}
h1 {{
  font-size: clamp(1.8rem, 3.2vw, 2.4rem);
  line-height: 1.15;
  margin: 0 0 0.85rem;
  text-wrap: balance;
  font-weight: 700;
  letter-spacing: -0.01em;
}}
.lede {{ max-width: 66ch; color: var(--ink-soft); font-size: 1.02rem; line-height: 1.6; margin: 0 0 1.5rem; }}
.stat-row {{ display: flex; flex-wrap: wrap; gap: 0.6rem; }}
.stat {{
  border: 1px solid var(--border); border-radius: 3px; padding: 0.5rem 0.85rem;
  background: var(--surface-alt); font-size: 0.83rem; color: var(--ink-soft);
  display: flex; align-items: baseline; gap: 0.4rem;
}}
.stat b {{ font-family: "IBM Plex Mono", monospace; font-size: 0.98rem; color: var(--ink); }}

.cat-nav {{ display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 1.5rem; }}
.cat-nav a {{
  text-decoration: none; border: 1px solid var(--border); border-radius: 999px;
  padding: 0.35rem 0.85rem; font-size: 0.8rem; color: var(--ink-soft); background: var(--surface);
}}

.cat-section {{ padding-top: 3rem; scroll-margin-top: 1.5rem; }}
.cat-head {{ border-bottom: 2px solid var(--border); padding-bottom: 1rem; margin-bottom: 1.6rem; }}
.cat-head h2 {{ margin: 0 0 0.3rem; font-size: 1.35rem; }}
.cat-head p {{ margin: 0; color: var(--ink-soft); font-size: 0.92rem; max-width: 60ch; }}

.task-list {{ display: flex; flex-direction: column; gap: 1.4rem; }}
.task {{
  display: grid;
  grid-template-columns: 128px 1fr;
  gap: 1.2rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 1.1rem;
}}
.task-frames {{ display: flex; flex-direction: column; gap: 0.4rem; align-items: center; justify-content: flex-start; }}
.task-frames figure {{ margin: 0; width: 100%; }}
.task-frames img {{
  width: 100%; aspect-ratio: 1; object-fit: contain; border-radius: 3px;
  background: var(--surface-alt); border: 1px solid var(--border); display: block;
}}
.frame-arrow {{ color: var(--ink-faint); font-size: 0.9rem; text-align: center; }}

.task-head {{ display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap; margin-bottom: 0.5rem; }}
.task-id {{
  font-family: "IBM Plex Mono", monospace; font-size: 0.72rem; font-weight: 600;
  color: var(--accent); background: var(--accent-soft); border-radius: 3px; padding: 0.2rem 0.4rem;
}}
.task-head h3 {{ margin: 0; font-size: 1.05rem; font-family: "IBM Plex Mono", monospace; font-weight: 600; }}
.pill {{
  font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.04em; border-radius: 999px;
  padding: 0.15rem 0.5rem; font-weight: 600;
}}
.pill-id {{ background: var(--accent-soft); color: var(--accent); }}
.pill-ood {{ background: var(--warn-soft); color: var(--warn); }}

.task-desc {{ font-size: 0.92rem; color: var(--ink-soft); line-height: 1.5; margin: 0 0 0.8rem; }}

.wbar {{ display: flex; width: 100%; height: 0.4rem; border-radius: 3px; overflow: hidden; border: 1px solid var(--border); margin-bottom: 0.9rem; }}
.wbar-seg {{ height: 100%; }}
.wbar-seg.seg-0 {{ background: var(--accent); opacity: 0.9; }}
.wbar-seg.seg-1 {{ background: var(--accent); opacity: 0.7; }}
.wbar-seg.seg-2 {{ background: var(--accent); opacity: 0.52; }}
.wbar-seg.seg-3 {{ background: var(--accent); opacity: 0.36; }}
.wbar-seg.seg-4 {{ background: var(--accent); opacity: 0.22; }}
.wbar-seg.seg-degraded {{ background: repeating-linear-gradient(135deg, var(--warn), var(--warn) 3px, transparent 3px, transparent 6px); }}

.crit-list {{ list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.6rem; }}
.crit-item {{ border-left: 2px solid var(--border); padding-left: 0.7rem; }}
.crit-item.crit-degraded {{ border-left-color: var(--warn); }}
.crit-head {{ display: flex; align-items: baseline; gap: 0.5rem; margin-bottom: 0.15rem; }}
.crit-head code {{
  font-family: "IBM Plex Mono", monospace; font-size: 0.85rem; font-weight: 600; color: var(--ink);
  background: var(--code-bg); border-radius: 2px; padding: 0.05rem 0.35rem;
}}
.crit-weight {{ font-family: "IBM Plex Mono", monospace; font-size: 0.78rem; color: var(--ink-faint); }}
.crit-weight-dropped {{ text-decoration: line-through; color: var(--warn); opacity: 0.7; }}
.crit-flag {{ font-size: 0.68rem; color: var(--warn); font-family: "IBM Plex Mono", monospace; }}
.crit-flag-dropped {{ font-weight: 600; }}
.crit-item p {{ margin: 0; font-size: 0.85rem; color: var(--ink-soft); line-height: 1.5; }}

.task-note {{
  margin: 0.9rem 0 0; padding: 0.65rem 0.8rem; background: var(--warn-soft); border-radius: 4px;
  font-size: 0.82rem; color: var(--ink-soft); line-height: 1.55;
}}
.task-note b {{ color: var(--warn); }}
.task-note code {{ font-family: "IBM Plex Mono", monospace; background: var(--code-bg); border-radius: 2px; padding: 0.02rem 0.3rem; font-size: 0.92em; }}

.pill-active {{ background: var(--accent-soft); color: var(--accent); }}
.pill-parked {{ background: var(--surface-alt); color: var(--ink-faint); border: 1px solid var(--border); }}
.task-parked {{ opacity: 0.82; }}
.task-parked .task-frames {{ filter: grayscale(0.5); }}

.task-note-parked {{ background: var(--surface-alt); border: 1px dashed var(--border); }}
.task-note-parked {{ color: var(--ink-faint); }}

.task-note-impl {{ background: var(--surface-alt); border-left: 2px solid var(--accent); }}
.task-note-impl b {{ color: var(--accent); }}
.task-note-impl code {{ font-family: "IBM Plex Mono", monospace; background: var(--code-bg); border-radius: 2px; padding: 0.02rem 0.3rem; font-size: 0.92em; }}

.assembly-note {{
  margin: 0.9rem 0 0; padding: 0.6rem 0.75rem; background: var(--code-bg); border-radius: 4px;
  font-size: 0.78rem; color: var(--ink-faint); line-height: 1.55;
}}
.assembly-note b {{ color: var(--ink-soft); }}
.assembly-note code {{ font-family: "IBM Plex Mono", monospace; background: var(--surface); border-radius: 2px; padding: 0.02rem 0.3rem; font-size: 0.95em; border: 1px solid var(--border); }}

.judge-block {{
  margin-top: 0.9rem; padding: 0.75rem 0.85rem; border: 1px solid var(--cat-knowledge);
  border-radius: 4px; background: var(--surface-alt);
}}
.judge-head {{ font-size: 0.82rem; font-weight: 600; color: var(--cat-knowledge); margin-bottom: 0.4rem; }}
.judge-head code {{ font-family: "IBM Plex Mono", monospace; background: var(--code-bg); border-radius: 2px; padding: 0.02rem 0.3rem; font-size: 0.92em; color: var(--ink); }}
.judge-sub {{ font-size: 0.78rem; color: var(--ink-faint); margin: 0 0 0.55rem; line-height: 1.5; }}
.judge-sub code {{ font-family: "IBM Plex Mono", monospace; }}
.judge-prompt {{
  font-family: "IBM Plex Mono", monospace; font-size: 0.72rem; line-height: 1.6; color: var(--ink-soft);
  background: var(--code-bg); border-radius: 4px; padding: 0.7rem 0.8rem; margin: 0;
  white-space: pre-wrap; word-break: break-word; max-height: 320px; overflow-y: auto;
}}

footer.page-foot {{ margin-top: 4rem; padding-top: 2rem; border-top: 1px solid var(--border); }}
footer.page-foot p {{ color: var(--ink-faint); font-size: 0.82rem; max-width: 72ch; line-height: 1.6; }}
footer.page-foot code {{ font-family: "IBM Plex Mono", monospace; background: var(--code-bg); border-radius: 2px; padding: 0.05rem 0.3rem; }}

@media (max-width: 620px) {{
  .wrap {{ padding: 0 1.1rem; }}
  .task {{ grid-template-columns: 1fr; }}
  .task-frames {{ flex-direction: row; }}
  .task-frames figure {{ width: 45%; }}
}}
</style>

<header class="top">
  <div class="wrap">
    <p class="eyebrow">VBVR-EvalKit &middot; companion to Image-Only Evaluation</p>
    <h1>What each of the 15 VBVR evaluators actually checks</h1>
    <p class="lede">Every task is scored by a deterministic, code-based rule &mdash; not an LLM judge, except {n_judged} task that also has a supplementary VLM-judge rubric defined (prompt reproduced on its card below; <code class="mono">key_door_matching</code>'s rubric is kept for reference even though that task itself is parked). No currently-Active task uses a judge right now &mdash; <code class="mono">ball_bounces_given_time</code> briefly did, but was simplified to a plain deterministic <code class="mono">final_position</code> check once its judge-supplemented criterion (<code class="mono">path_shape</code>) turned out to depend on generator behavior that's since changed upstream (see its card below). This is a plain-language reading of each evaluator's <code class="mono">_evaluate_task_specific</code> source: what sub-criteria it checks, how much each is worth, and which ones needed rewriting for InternVL-U's single-image output. All 15 implemented evaluators are shown below; the <span class="pill pill-active" style="display:inline">&#9679; Active</span> / <span class="pill pill-parked" style="display:inline">&#9675; Parked</span> badge on each says whether it's in the current 10-task training/eval set (not a uniform per-category ratio &mdash; see <code class="mono">image_evaluator.py</code>'s <code class="mono">LOCKED_TASKS</code> for the full per-category reasoning, and each parked card's own note for why that specific one is sitting out).</p>
    <div class="stat-row">
      <div class="stat">{n_tasks} evaluators implemented <b>&middot;</b> {n_criteria} scoring criteria</div>
      <div class="stat">{n_active} <b>Active</b> / {n_parked} <b>Parked</b></div>
      <div class="stat">{n_degraded} criteria <b>needed multi-frame</b> &rarr; redefined for image-only mode</div>
      <div class="stat">{n_judged} tasks <b>have a judge rubric</b></div>
    </div>
    <nav class="cat-nav">
      {"".join(f'<a href="#cat-{c["name"].lower()}">{c["name"]}</a>' for c in CATEGORIES)}
    </nav>
  </div>
</header>

<div class="wrap">
  {"".join(category_sections)}

  <footer class="page-foot">
    <p>Source: <code>Evaluation/VBVR-EvalKit/vbvr_bench/evaluators/</code> (original video-mode rules, vendored, unmodified) and <code>Evaluation/VBVR-CustomEval/evaluators/image_evaluator.py</code> (image-only redefinitions). Weight bars show each criterion's share of the score <b>as actually computed right now</b> (renormalized after excluding any dropped criterion) &mdash; not the original video-mode weights. A criterion listed with a struck-through weight and &#10007; <b>not scored</b> is confirmed excluded from the current calculation entirely, per the note below each task. See the companion <i>Image-Only Evaluation</i> artifact for per-task validation numbers (exact-match rate vs. the video evaluator, discriminative gap on 20 test pairs). Example frames and prompts are sample <code>00000</code> from each task's downloaded VBVR-Bench ground truth.</p>
  </footer>
</div>
'''

with open(OUT_PATH, "w") as f:
    f.write(HTML)

print("wrote", OUT_PATH, len(HTML), "chars")
