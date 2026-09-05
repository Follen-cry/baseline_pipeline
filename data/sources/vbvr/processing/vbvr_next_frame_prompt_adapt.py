"""Per-task prompt adaptation for the VBVR 3-frame-in -> 1-frame-out format.

Production copy of data/scripts/vbvr_prompt_adapt.py, verbatim -- no logic
changes, relocated so the production next-frame pipeline doesn't depend on
the smoke-test script directory. If this file and the smoke-test original
ever diverge, this one is the copy actually used to build training data.

Problem this fixes
-------------------
Each VBVR-DataFactory generator's own prompt.txt describes the FULL video's
end-to-end goal/outcome (e.g. "...the ball stops after the 6th bounce, with
its final position at the wall where the last collision occurs"). Our
sampling pipeline (vbvr_next_frame_sample.py) reuses that SAME whole-task
prompt verbatim for every window sampled from a video -- including windows
drawn from the middle of a long clip (gravity_physics, maze, etc.), which
only show a small local slice of motion, not the final state. The model is
then asked to predict one local next-frame step while being told about the
end-of-video outcome, a real semantic mismatch.

Fix: for each of the 15 tasks, `adapt_prompt(task_name, parameters, direction)`
builds a NEW instruction from the generator's own structured `metadata.json`
"parameters" (not the free-text prompt.txt), tailored to what that specific
task's scene/dynamics actually are, followed by a shared closing that
explicitly reframes the task as local continuation. ``direction="forward"``
(default) asks for the very next frame ("do not jump ahead to the
final/goal state"); ``direction="backward"`` asks for the frame immediately
preceding the 3 shown frames instead (added for the S1 variable-target
training setting, which mixes forward- and backward-target windows drawn
from the SAME already-extracted 4-frame windows -- see
vbvr_next_frame_sample.py's ``--direction`` flag).
"""
from __future__ import annotations

NEXT_FRAME_CLOSING_FORWARD = (
    "You are shown 3 consecutive frames sampled evenly in time from this "
    "sequence. Based on the direction and rate of motion or transformation "
    "established across these 3 frames, predict the very next frame: "
    "advance everything that is moving or changing by one more equal "
    "increment along the same established direction. Do not jump ahead to "
    "the final, completed, or goal state -- show only one more incremental "
    "step of continued motion."
)

NEXT_FRAME_CLOSING_BACKWARD = (
    "You are shown 3 consecutive frames sampled evenly in time from this "
    "sequence, occurring immediately AFTER the frame you must predict. "
    "Based on the direction and rate of motion or transformation "
    "established across these 3 frames, predict the frame that occurred "
    "immediately BEFORE the first frame shown: reverse one equal increment "
    "of the same established motion. Do not jump back to the very start or "
    "initial state -- show only one step backward in time from the first "
    "frame shown."
)

# Back-compat alias -- some callers may still import the old forward-only name.
NEXT_FRAME_CLOSING = NEXT_FRAME_CLOSING_FORWARD

_CLOSINGS = {
    "forward": NEXT_FRAME_CLOSING_FORWARD,
    "backward": NEXT_FRAME_CLOSING_BACKWARD,
}

_NAMED_COLORS = {
    "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255),
    "yellow": (255, 255, 0), "magenta": (255, 0, 255), "cyan": (0, 255, 255),
    "orange": (255, 165, 0), "purple": (128, 0, 128), "pink": (255, 192, 203),
    "white": (255, 255, 255), "black": (0, 0, 0), "gray": (128, 128, 128),
    "brown": (139, 69, 19),
}


def _color_name(rgb) -> str:
    if rgb is None:
        return "colored"
    if isinstance(rgb, str):
        return rgb
    r, g, b = rgb[0], rgb[1], rgb[2]
    best, best_d = "colored", float("inf")
    for name, (nr, ng, nb) in _NAMED_COLORS.items():
        d = (r - nr) ** 2 + (g - ng) ** 2 + (b - nb) ** 2
        if d < best_d:
            best, best_d = name, d
    return best


def _adapt_ball_bounces_given_time(p: dict) -> str:
    n = p.get("num_bounces", "several")
    return (
        f"This is a top-down 2D physics simulation of a ball bouncing elastically "
        f"inside a rectangular boundary (angle of incidence equals angle of "
        f"reflection). The ball travels in a straight line at constant speed until "
        f"it hits a wall, then bounces off at the mirrored angle, repeating for a "
        f"total of {n} bounces across the full sequence."
    )


def _adapt_mirror_reflection(p: dict) -> str:
    refl = p.get("reflectivity")
    theta = p.get("theta_incident_degrees")
    refl_s = f"{refl:.2f}" if isinstance(refl, (int, float)) else "a fixed value"
    theta_s = f"{theta:.1f}°" if isinstance(theta, (int, float)) else "a fixed angle"
    return (
        f"This is a 2D ray-optics diagram: a light ray travels toward a mirror "
        f"surface and reflects off it (angle of incidence equals angle of "
        f"reflection), with mirror reflectivity {refl_s} and incident angle "
        f"{theta_s}. The reflected ray is shown extending outward from the mirror "
        f"over time, growing toward the image boundary."
    )


def _adapt_gravity_physics(p: dict) -> str:
    g = p.get("gravity")
    g_s = f"{g:.2f} m/s²" if isinstance(g, (int, float)) else "gravity"
    return (
        f"This is a 2D physics simulation of a ball in free-fall and bouncing "
        f"under {g_s}, shown with a velocity arrow indicating the ball's current "
        f"speed and direction. The ball falls, bounces off the ground with some "
        f"energy loss, and its arrow updates to reflect its changing velocity "
        f"throughout."
    )


def _adapt_shape_color_then_scale(p: dict) -> str:
    cf, ct = p.get("color_from", "its starting color"), p.get("color_to", "a new color")
    sf, st = p.get("scale_from", "its starting size"), p.get("scale_to", "a new size")
    return (
        f"This is a visual-analogy video: a shape undergoes a two-step "
        f"transformation applied to the bottom-row shape, mirroring the top-row "
        f"example -- first the color changes from {cf} to {ct}, then (only after "
        f"the color change is complete) the size changes from {sf} to {st}. "
        f"Shape identity and position stay fixed throughout; only color then size "
        f"change, in that order."
    )


def _adapt_shape_outline_then_move(p: dict) -> str:
    mv = p.get("move_to", "a small amount")
    mv_s = str(mv).replace("_", " ")
    return (
        f"This is a visual-analogy video: a shape undergoes a two-step "
        f"transformation applied to the bottom-row shape, mirroring the top-row "
        f"example -- first it switches from a filled to an outline-only style, "
        f"then (only after the style change is complete) it moves {mv_s} while "
        f"keeping its shape and size unchanged. Style changes first, then "
        f"position, in that order."
    )


def _adapt_shape_color_then_move(p: dict) -> str:
    cf, ct = p.get("color_from", "its starting color"), p.get("color_to", "a new color")
    mv = p.get("move_to", "a small amount")
    mv_s = str(mv).replace("_", " ")
    return (
        f"This is a visual-analogy video: a shape undergoes a two-step "
        f"transformation applied to the bottom-row shape, mirroring the top-row "
        f"example -- first the color changes from {cf} to {ct}, then (only after "
        f"the color change is complete) it moves {mv_s}. Color changes first, "
        f"then position, in that order."
    )


def _adapt_grid_shortest_path(p: dict) -> str:
    # start_color/end_color/agent_color in metadata are randomized per video
    # (verified: 4 of 5 raw smoke-test samples used a different color
    # assignment than "red agent / green start / purple end"), so this
    # deliberately names no colors at all rather than risk describing the
    # wrong one -- "dot" + "marked cell" stays correct regardless of the
    # actual palette drawn for a given sample.
    return (
        "This is a top-down grid-navigation video: a dot-shaped agent starts "
        "in one marked cell and must reach another marked target cell, moving "
        "exactly one grid cell at a time (up, down, left, or right) along the "
        "shortest path between them."
    )


def _adapt_key_door_matching(p: dict) -> str:
    target = _color_name(p.get("target_color"))
    return (
        f"This is a top-down grid-maze video: a green circular agent must first "
        f"walk to the {target} diamond-shaped key, pick it up, then walk to the "
        f"{target} rectangular door -- the only door matching that key's color -- "
        f"moving one grid cell at a time."
    )


def _adapt_maze(p: dict) -> str:
    size = p.get("grid_size", "N")
    return (
        f"This is a top-down {size}×{size} grid-maze video: an orange agent "
        f"follows a single fixed shortest path from the green start cell to the "
        f"red flag goal cell, moving one cell at a time through white pathways and "
        f"never through the dark walls."
    )


def _adapt_grid_shift(p: dict) -> str:
    direction = p.get("direction", "a fixed direction")
    steps = p.get("steps", "several")
    color = p.get("color_name") or _color_name(p.get("color"))
    return (
        f"This is a top-down grid video: {color} square blocks all translate "
        f"uniformly {direction}ward by a total of {steps} grid cell(s), moving "
        f"together in lockstep at the same constant rate."
    )


def _adapt_rotation_puzzle(p: dict) -> str:
    n = p.get("num_squares", "several")
    return (
        f"This is a top-down rotation-puzzle video: {n} squares, each containing "
        f"a pipe segment, rotate in place in 90° turns (clockwise or "
        f"counterclockwise, each square at its own fixed rate) until every pipe "
        f"connects into one continuous path; square positions never change, only "
        f"orientation."
    )


def _adapt_2d_geometric_transformation(p: dict) -> str:
    shape = p.get("shape_type", "polygon").replace("_", " ")
    direction = p.get("rotation_direction", "a fixed direction")
    return (
        f"This is a 2D video of a {shape} rotating {direction} in place around a "
        f"fixed marked center, moving at a constant angular rate from its initial "
        f"orientation toward a dashed target outline; size and center position "
        f"never change, only orientation."
    )


def _adapt_stable_sort(p: dict) -> str:
    return (
        "This is a top-down sorting video: two groups of shapes are each "
        "rearranged by type, then sorted smallest-to-largest within their group, "
        "and moved into a single horizontal line -- shape identity, size, and "
        "color never change, only position, with each shape moving in a straight "
        "line toward its final sorted spot."
    )


def _adapt_multi_object_placement(p: dict) -> str:
    return (
        "This is a top-down placement video: colored objects each travel in a "
        "straight line at constant speed to the star marker of the matching "
        "color, while the star markers themselves never move."
    )


def _adapt_animal_size_sorting(p: dict) -> str:
    return (
        "This is a top-down sorting video: animal faces of different sizes, "
        "scattered randomly, each move in a straight line to their sorted "
        "position -- smallest to largest, left to right -- aligned along a "
        "horizontal bottom baseline."
    )


_ADAPTERS = {
    "ball_bounces_given_time": _adapt_ball_bounces_given_time,
    "mirror_reflection": _adapt_mirror_reflection,
    "gravity_physics": _adapt_gravity_physics,
    "shape_color_then_scale": _adapt_shape_color_then_scale,
    "shape_outline_then_move": _adapt_shape_outline_then_move,
    "shape_color_then_move": _adapt_shape_color_then_move,
    "grid_shortest_path": _adapt_grid_shortest_path,
    "key_door_matching": _adapt_key_door_matching,
    "maze": _adapt_maze,
    "grid_shift": _adapt_grid_shift,
    "rotation_puzzle": _adapt_rotation_puzzle,
    "2d_geometric_transformation": _adapt_2d_geometric_transformation,
    "stable_sort": _adapt_stable_sort,
    "multi_object_placement": _adapt_multi_object_placement,
    "animal_size_sorting": _adapt_animal_size_sorting,
}


def scene_description(task_name: str, parameters: dict) -> str:
    """Just the per-task scene text (the "C" half of adapt_prompt's output),
    with no closing instruction -- for callers building their own closing/
    question, e.g. vbvr_next_frame_make_dual.py's S3 MCQ prompt, which needs
    the scene description but not the "predict the next frame" framing."""
    fn = _ADAPTERS.get(task_name)
    if fn is None:
        raise KeyError(f"no prompt adapter registered for task {task_name!r}")
    return fn(parameters or {})


def adapt_prompt(task_name: str, parameters: dict, direction: str = "forward",
                  include_scene: bool = True) -> str:
    """include_scene=True (default): scene description (task-specific, i.e. the
    "C" in the S0-S3 training-setting design -- see VBVR-DataGeneration's task
    presets doc) + the generic direction closing. This is what every prior
    version of this function produced, and is what makes the *default* output
    of this pipeline setting S2 (VC2I-F), not S0.

    include_scene=False: closing ONLY, no task-specific description -- this is
    the "no C" prompt used by settings S0/S1 (V2I-F / V2I-V). The per-task
    _ADAPTERS functions are never even called in this mode."""
    closing = _CLOSINGS.get(direction)
    if closing is None:
        raise ValueError(f"unknown direction {direction!r} (expected 'forward' or 'backward')")
    if not include_scene:
        return closing
    fn = _ADAPTERS.get(task_name)
    if fn is None:
        raise KeyError(f"no prompt adapter registered for task {task_name!r}")
    scene = fn(parameters or {})
    return scene + "\n\n" + closing
