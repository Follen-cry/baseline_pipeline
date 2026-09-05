"""
A standalone "pure judge" scoring system for VBVR image-only eval: Qwen3-VL-30B-fp8
scores EVERY active task, not just the rule-based-weak ones llm_judge.py's RUBRICS
supplements (that module is a supplement layered on top of image_evaluator.py's
rule-based scores; this one is a full alternative -- same images, no detector code
at all, a VLM does 100% of the scoring).

The per-task rubrics below are not invented from scratch -- each is a direct
plain-language translation of that task's CURRENT image-only rule-based evaluator
(image_evaluator.py), specifically its ACTIVE (non-dropped) task_specific
sub-criteria and their weights, renormalized to sum to 100 exactly the way
ImageOnlyEvalMixin's weighted_average() does. Source of truth for those weights:
artifacts/gen_rules_artifact.py's CATEGORIES list, itself re-verified against the
live evaluator source on 2026-08-25/26. Where the rule-based system drops a
sub-criterion as unrecoverable from a static image (e.g. grid_shift's
synchronization, 2d_geometric_transformation's rotation_center), the judge rubric
drops the identical criterion for the identical reason -- a VLM shown one frame
has the same missing signal a pixel detector does.

REVISED 2026-08-26: per-criterion scoring changed from a 0-100 integer (too
fine-grained for a VLM to judge reliably/consistently) to a 3-level categorical
scale -- wrong / partial / correct -- with an explicit, concrete definition of
each level written per criterion (not left to the judge's own interpretation of
what "50" vs "70" means). Levels map to wrong=0.0, partial=0.5, correct=1.0,
and this module computes the weighted total itself using the same weights the
rule-based evaluator uses -- i.e. literally swapping the rule-based evaluator's
detector functions for a VLM judge, keeping the weight/aggregation logic
identical, so the two systems are dimension-by-dimension comparable on the same
images.

REVISED 2026-08-27, then REVERTED same day: tried strengthening the prompt
further by forcing a written "observation" per criterion before its verdict
(a standard LLM grounding technique -- describe first, then judge). It fixed
several individually-flagged false positives when spot-tested (e.g. grid_shift
cases where the judge had confidently marked wrong block positions "correct").
But re-run at full scale (all 900 real generations) it was a clear regression,
not an improvement: judge means jumped up across nearly every task (e.g. maze
0.06->0.63, grid_shift 0.39->0.66) and rule-vs-judge Spearman correlation
*dropped* (0.26->0.17). Manually checked: on a maze sample with literally no
path drawn (just walls and the end flag), the judge scored it a perfect 1.0
with a detailed, confident, entirely fabricated description ("a complete,
valid path of yellow dots connecting start to end"). 180/900 samples (20%)
showed this rule~0/judge~1.0 pattern, across every task, not just position-
heavy ones. Root cause understood as: forcing the model to write a plausible-
sounding "observation" made it MORE prone to confabulating supporting detail
for a verdict it was already inclined toward (from the task instruction text
alone), rather than actually grounding the verdict in the image -- the
opposite of the intended effect. Lesson: spot-testing a prompt change only
against previously-flagged failure cases is confirmation-biased and does not
predict full-corpus behavior; a fresh random sample must be checked before
committing to a full re-score. Reverted to the pre-2026-08-27 prompt (still
includes 2026-08-26's "verify element-by-element, don't assume correctness"
addition, which tested fine and was kept).

Reuses llm_judge.py's image encoding + OpenAI-compatible client setup rather than
duplicating it.

    from llm_judge_full import FullVLMJudge, TASK_CRITERIA, build_full_judge_instruction

    judge = FullVLMJudge()
    result = judge.score(
        task_name="stable_sort",
        prompt_text=open(".../prompt.txt").read(),
        first_frame_path=".../first_frame.png",
        gen_final_path=".../model_output.png",
        gt_final_path=".../final_frame.png",
    )
    print(result["score"], result["criteria"], result["reasoning"])
"""

import json
import re
from typing import Any, Dict, List, Tuple

from llm_judge import DEFAULT_BASE_URL, DEFAULT_MODEL, _to_data_uri
from openai import OpenAI

LEVEL_VALUE = {"wrong": 0.0, "partial": 0.5, "correct": 1.0}

# task_name -> (task_desc, [(criterion_key, weight_pct, {"correct": ..., "partial": ..., "wrong": ...}), ...])
# Weights are the CURRENT active-only, renormalized-to-100 weights the image-only
# rule-based evaluator actually uses for that task (dropped criteria excluded).
TASK_CRITERIA: Dict[str, Tuple[str, List[Tuple[str, float, Dict[str, str]]]]] = {
    "ball_bounces_given_time": (
        "A ball starts at a marked position with a direction arrow, and is supposed "
        "to bounce off the boundary walls a stated number of times following elastic "
        "collision physics (angle of incidence = angle of reflection), ending at a "
        "specific final position. Note: this scene shows only the ball's resting "
        "position, not a drawn bounce path -- you cannot verify bounce count or "
        "reflection angles from a single static image, only final position.",
        [
            ("final_position", 100.0, {
                "correct": "The ball's final position in the candidate is essentially the same location as in the ground truth (small pixel offset, clearly the same resting spot).",
                "partial": "The ball is in roughly the right region/quadrant of the frame but noticeably off from the exact ground-truth position.",
                "wrong": "The ball is in a clearly different location, missing entirely, or the scene doesn't show a plausible resting ball at all.",
            }),
        ],
    ),
    "stable_sort": (
        "Shapes of two types, each in three sizes, start scattered. They must be "
        "rearranged into a single horizontal line: grouped by type, and within each "
        "group ordered smallest-to-largest left to right.",
        [
            ("classification", 30.0, {
                "correct": "All shapes of the same type are grouped adjacent to each other with no type intermixed.",
                "partial": "Most shapes are grouped by type but one or two are out of place, or a group is split.",
                "wrong": "Shapes of different types are scattered/interleaved with no type grouping.",
            }),
            ("order", 30.0, {
                "correct": "Within every type-group, shapes run smallest-to-largest left to right.",
                "partial": "Some groups are correctly ordered by size, others are not, or order is only partially correct within a group.",
                "wrong": "Shapes within groups are not ordered by size at all (random or reversed).",
            }),
            ("fidelity", 30.0, {
                "correct": "Every shape's type, size, and color exactly matches the first frame -- nothing added, removed, or recolored.",
                "partial": "Most shapes match the first frame but one shape's color/size/type looks altered or minorly distorted.",
                "wrong": "Shapes were added, removed, or multiple shapes changed type/color/size from the first frame.",
            }),
            ("layout", 10.0, {
                "correct": "All shapes sit on the same horizontal line (baseline).",
                "partial": "Most shapes are roughly aligned but a few are noticeably off the line.",
                "wrong": "Shapes are scattered at clearly different heights, not in a line.",
            }),
        ],
    ),
    "multi_object_placement": (
        "Multiple colored objects and colored star markers start scattered. Each "
        "object must move onto the star marker of the matching color; the markers "
        "themselves never move.",
        [
            ("color_matching", 37.5, {
                "correct": "Every colored object ends up overlapping or touching the star marker of the same color.",
                "partial": "Some objects reach their matching star and others don't, or objects are near but not on their matching star.",
                "wrong": "Objects are not on their matching-color stars at all (wrong star, or didn't move).",
            }),
            ("alignment", 31.25, {
                "correct": "Each object's center is closely centered on its matching star's center.",
                "partial": "Objects are on the correct star but noticeably off-center.",
                "wrong": "Objects are far from their matching star's center or not matched at all.",
            }),
            ("fidelity", 18.75, {
                "correct": "Object count and each object's size are preserved exactly from the first frame.",
                "partial": "Object sizes/count are mostly preserved but one object looks resized or slightly different.",
                "wrong": "Objects are missing, duplicated, or clearly resized from the first frame.",
            }),
            ("star_invariance", 12.5, {
                "correct": "All star markers are in exactly the same positions as the first frame.",
                "partial": "Most stars are unchanged but one appears to have moved slightly.",
                "wrong": "Star markers have moved or are missing compared to the first frame.",
            }),
        ],
    ),
    "grid_shift": (
        "Colored blocks sit in an NxN grid and must all move the same direction, by "
        "the same number of steps, simultaneously.",
        [
            ("direction_correctness", 37.5, {
                "correct": "All blocks moved in the exact direction stated in the instruction.",
                "partial": "Most blocks moved in the correct direction but one or more moved a different way.",
                "wrong": "Blocks moved in the wrong direction or did not move at all.",
            }),
            ("step_accuracy", 37.5, {
                "correct": "All blocks moved the exact number of grid steps specified.",
                "partial": "Blocks moved in the right direction but by the wrong number of steps, or steps are inconsistent across blocks.",
                "wrong": "Blocks did not move the specified distance at all, or the movement bears no relation to the instruction.",
            }),
            ("position_precision", 18.75, {
                "correct": "Final block positions precisely land on the correct grid cells, matching ground truth.",
                "partial": "Final positions are close to the correct cells but slightly misaligned (partial-cell offset).",
                "wrong": "Final positions do not match the expected grid cells at all.",
            }),
            ("completeness", 6.25, {
                "correct": "Same number of blocks, same colors/pattern, as the first frame.",
                "partial": "Most blocks match but one is missing, duplicated, or recolored.",
                "wrong": "Multiple blocks are missing, duplicated, or the pattern is unrecognizable.",
            }),
        ],
    ),
    "rotation_puzzle": (
        "L-shaped pipe tiles sit in a 2x2 grid and must be rotated in place (90-degree "
        "increments only) so every pipe opening connects into one continuous loop.",
        [
            ("path_connection", 40.0, {
                "correct": "Every pipe opening connects to a neighboring tile's opening, forming one continuous connected loop.",
                "partial": "Some pipe connections line up but at least one junction is broken or misaligned.",
                "wrong": "Pipes don't connect into a coherent path at all.",
            }),
            ("rotation_accuracy", 30.0, {
                "correct": "All tiles appear rotated in clean 90-degree increments.",
                "partial": "Most tiles are at clean angles but one or more look at a skewed/partial angle.",
                "wrong": "Tiles are at arbitrary/non-90-degree angles or rotation is not evident.",
            }),
            ("position_preservation", 20.0, {
                "correct": "Every tile stayed in its original grid cell -- only rotation changed.",
                "partial": "Most tiles stayed in place but one appears to have shifted to a different cell.",
                "wrong": "Multiple tiles moved to different grid cells (not just rotated in place).",
            }),
            ("alignment_precision", 10.0, {
                "correct": "Pipe openings line up precisely at tile edges with no visible gaps.",
                "partial": "Openings are close to aligned but have small visible gaps/offsets.",
                "wrong": "Pipe openings are clearly misaligned or don't meet at tile edges.",
            }),
        ],
    ),
    "shape_outline_then_move": (
        "The scene shows an analogy A->B->C :: D->?->? with two rows of shapes and "
        "arrows. On the top row, a shape first becomes outline-only (step 1), then "
        "moves a small amount up or down (step 2). On the bottom row, a different "
        "shape D starts filled and must undergo the same two-step transformation: "
        "convert to outline-only style, then move the same direction/amount as the "
        "top row, keeping its shape and size unchanged.",
        [
            ("first_row_preservation", 35.0, {
                "correct": "All top-row reference shapes are pixel-identical to the first frame -- completely untouched.",
                "partial": "Top-row shapes are mostly unchanged but one looks slightly altered (position/size/style).",
                "wrong": "Top-row reference shapes are noticeably changed, moved, or missing.",
            }),
            ("outline_conversion", 30.0, {
                "correct": "The bottom-row shape is clearly outline-only (hollow interior, not filled) in the final image, matching the top row's style change.",
                "partial": "The bottom-row shape shows a partial or ambiguous style change (e.g. thin fill, unclear outline).",
                "wrong": "The bottom-row shape is still filled, or its style change doesn't match the top row's pattern at all.",
            }),
            ("move_accuracy", 25.0, {
                "correct": "The bottom-row shape moved in the same direction (up/down) as the top row's demonstrated movement, by a comparable small amount.",
                "partial": "The bottom-row shape moved in the correct direction but by a clearly different amount, or the direction is ambiguous.",
                "wrong": "The bottom-row shape moved in the wrong direction, or did not move at all.",
            }),
            ("fidelity", 10.0, {
                "correct": "The bottom-row shape's type and size exactly match the first frame -- only style and position changed.",
                "partial": "Shape type/size is mostly preserved but looks slightly resized or distorted.",
                "wrong": "The shape's type changed, or it is missing/duplicated, or clearly resized from the first frame.",
            }),
        ],
    ),
    "shape_color_then_move": (
        "A top row of 3 reference shapes (A, B, C) demonstrates a color-then-move "
        "transform pattern. The bottom row keeps shape D fixed and must add two new "
        "shapes E, F colored like B, following the same pattern.",
        [
            ("first_row_preservation", 40.0, {
                "correct": "All three top-row reference shapes are pixel-identical to the first frame -- completely untouched.",
                "partial": "Top-row shapes are mostly unchanged but one looks slightly altered (color/position/size).",
                "wrong": "Top-row reference shapes are noticeably changed, moved, or missing.",
            }),
            ("second_row_completion", 35.0, {
                "correct": "Bottom row keeps D in place and adds exactly two new shapes E and F alongside it.",
                "partial": "Bottom row has D plus some new shapes, but the count is off (only one added, or extra shapes), or D moved slightly.",
                "wrong": "Bottom row does not follow the pattern at all -- D missing/moved, or no new shapes added correctly.",
            }),
            ("color_accuracy", 20.0, {
                "correct": "E and F are colored to match reference shape B specifically.",
                "partial": "E and F have some correct coloring but one is the wrong color, or the match is approximate.",
                "wrong": "E and F are colored incorrectly, not matching B at all.",
            }),
            ("shape_count", 5.0, {
                "correct": "Exactly 6 shapes total in the final image (3 top + 3 bottom).",
                "partial": "Close to 6 shapes (off by one).",
                "wrong": "Shape count is clearly wrong (missing multiple or many extra shapes).",
            }),
        ],
    ),
    "animal_size_sorting": (
        "Animals of different sizes start scattered and must be sorted smallest-to-"
        "largest, left to right, all on the same baseline.",
        [
            ("sorting", 40.0, {
                "correct": "Final left-to-right order goes strictly from smallest to largest animal.",
                "partial": "Mostly ordered by size but one or two animals are out of sequence.",
                "wrong": "Animals are not ordered by size at all (random or reversed).",
            }),
            ("alignment", 30.0, {
                "correct": "All animals sit on the same baseline (bottom-aligned).",
                "partial": "Most animals are aligned but one or two are noticeably off the baseline.",
                "wrong": "Animals are at clearly different heights, not aligned.",
            }),
            ("fidelity", 20.0, {
                "correct": "Each animal's size and appearance exactly match the first frame.",
                "partial": "Most animals match but one looks distorted, resized, or recolored.",
                "wrong": "Multiple animals are distorted, recolored, or don't resemble the first frame.",
            }),
            ("completeness", 10.0, {
                "correct": "All animals from the first frame are present in the final arrangement.",
                "partial": "Most animals are present but one is missing or an extra one appears.",
                "wrong": "Multiple animals are missing from the final arrangement.",
            }),
        ],
    ),
    "maze": (
        "An agent must navigate a maze from a green start marker to a red end marker, "
        "drawing its path through the maze.",
        [
            ("path_validity", 45.0, {
                "correct": "The drawn path stays entirely within corridors and never crosses a wall.",
                "partial": "The path is mostly valid but crosses a wall or cuts a corner in one spot.",
                "wrong": "The path crosses walls repeatedly or doesn't follow the maze corridors at all.",
            }),
            ("path_completeness", 30.0, {
                "correct": "The drawn path fully connects the start to the end marker with no gaps.",
                "partial": "The path covers most of the route but has a gap or doesn't quite reach the end.",
                "wrong": "The path is fragmented, very incomplete, or doesn't connect start and end at all.",
            }),
            ("navigation_accuracy", 20.0, {
                "correct": "The path moves only between adjacent cells, no jumps or teleports.",
                "partial": "Mostly adjacent-cell moves but one jump/teleport is visible.",
                "wrong": "The path makes multiple jumps that skip cells or teleport across the maze.",
            }),
            ("element_preservation", 5.0, {
                "correct": "Maze walls and start/end markers are unchanged from the first frame.",
                "partial": "Walls/markers are mostly unchanged but something looks slightly altered.",
                "wrong": "Walls or markers are clearly changed, missing, or redrawn differently.",
            }),
        ],
    ),
    "2d_geometric_transformation": (
        "A colored shape must be rotated around a marked pivot point until it aligns "
        "with a dashed target outline shown in the first frame.",
        [
            ("rotation_angle", 50.0, {
                "correct": "Final shape's orientation matches the dashed target outline's orientation closely.",
                "partial": "Orientation is roughly in the right direction but off by a noticeable angle.",
                "wrong": "Orientation does not match the target outline at all (wrong angle, or shape not rotated).",
            }),
            ("position_alignment", 35.71, {
                "correct": "Final shape's center is at (or very near) the dashed target outline's center.",
                "partial": "Shape is in the general area of the target but noticeably offset from its center.",
                "wrong": "Shape's position bears no relation to the target outline's location.",
            }),
            ("shape_fidelity", 14.29, {
                "correct": "Shape's size/area is preserved -- not resized during rotation.",
                "partial": "Shape is slightly larger/smaller than it should be.",
                "wrong": "Shape is drastically resized, distorted, or is a different shape entirely.",
            }),
        ],
    ),
}

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def build_full_judge_instruction(task_name: str, prompt_text: str) -> str:
    """Build the exact text instruction sent to the judge (everything except the
    3 images). Pulled out as its own function, same reasoning as llm_judge.py's
    build_judge_instruction -- so the artifact can render precisely what the
    judge sees, not a hand-copied approximation."""
    entry = TASK_CRITERIA.get(task_name)
    if entry is None:
        raise ValueError(f"No criteria registered for task '{task_name}'. "
                          f"Available: {list(TASK_CRITERIA)}")
    task_desc, criteria = entry

    rubric_blocks = []
    for key, weight, levels in criteria:
        rubric_blocks.append(
            f"- {key}:\n"
            f"    correct: {levels['correct']}\n"
            f"    partial: {levels['partial']}\n"
            f"    wrong: {levels['wrong']}"
        )
    rubric_text = "\n".join(rubric_blocks)
    criteria_keys = ", ".join(f'"{key}": "correct"|"partial"|"wrong"' for key, _, _ in criteria)

    return (
        "You are a strict, deterministic grader for a visual reasoning benchmark. "
        "You will see three images in order: (1) the starting frame, (2) a candidate "
        "generated result, (3) the ground-truth correct result. Judge ONLY image (2) "
        "against the rubric below, using image (3) as the reference for what "
        "'correct' looks like and image (1) for the scene before the task was "
        "applied.\n\n"
        "Verify each criterion against the actual pixel content of image (2), not "
        "against what the task instruction merely asks for -- a model that fails the "
        "task often still uses the right colors, shapes, or object count, just in the "
        "wrong positions. Do not mark a criterion 'correct' because the right "
        "elements appear SOMEWHERE in the image; their exact positions, alignment, "
        "and completeness must specifically match image (3). For anything laid out "
        "on a grid or along a path, compare element-by-element (cell by cell, or "
        "step by step) against image (3) rather than forming a general impression "
        "of similarity -- if you cannot point to the specific matching detail, do "
        "not call it correct. A candidate that is mostly blank, or where the drawn "
        "content covers only a small fraction of what a complete result should show "
        "(e.g. a short stub instead of a full path), is wrong or at best partial, "
        "never correct, even if the fragment that is there looks locally clean.\n\n"
        f"Task: {task_desc}\n\n"
        f"Task instruction given to the generator: \"{prompt_text}\"\n\n"
        "For EACH criterion below, classify the candidate into exactly one of three "
        "levels -- wrong, partial, or correct -- using the concrete definition given "
        "for that level (do not use a finer-grained number; pick the closest of the "
        "three):\n\n"
        f"{rubric_text}\n\n"
        "Respond with ONLY a JSON object, no other text: "
        f'{{"criteria": {{{criteria_keys}}}, "reasoning": "<one or two short sentences>"}}'
    )


class FullVLMJudge:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL,
                 temperature: float = 0.0, max_tokens: int = 400):
        self.client = OpenAI(base_url=base_url, api_key="not-needed")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def score(
        self,
        task_name: str,
        prompt_text: str,
        first_frame_path: str,
        gen_final_path: str,
        gt_final_path: str,
        n_retries: int = 2,
    ) -> Dict[str, Any]:
        entry = TASK_CRITERIA.get(task_name)
        if entry is None:
            raise ValueError(f"No criteria registered for task '{task_name}'.")
        _, criteria = entry
        weights = {key: weight for key, weight, _ in criteria}

        instruction = build_full_judge_instruction(task_name, prompt_text)
        content = [
            {"type": "text", "text": instruction},
            {"type": "text", "text": "Image 1 (starting frame):"},
            {"type": "image_url", "image_url": {"url": _to_data_uri(first_frame_path)}},
            {"type": "text", "text": "Image 2 (candidate generated result -- grade this one):"},
            {"type": "image_url", "image_url": {"url": _to_data_uri(gen_final_path)}},
            {"type": "text", "text": "Image 3 (ground-truth correct result, for reference):"},
            {"type": "image_url", "image_url": {"url": _to_data_uri(gt_final_path)}},
        ]

        last_error = None
        raw_text = None
        for attempt in range(n_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                raw_text = resp.choices[0].message.content
                match = _JSON_BLOCK_RE.search(raw_text)
                if not match:
                    raise ValueError(f"No JSON object found in response: {raw_text!r}")
                parsed = json.loads(match.group(0))
                levels = {}
                observations = {}
                for k, v in parsed["criteria"].items():
                    if isinstance(v, dict):
                        obs = v.get("observation", "")
                        v = v.get("level", "")
                    else:
                        obs = ""  # tolerate a bare string, in case the model skips the observation field
                    v = str(v).strip().lower()
                    if v not in LEVEL_VALUE:
                        raise ValueError(f"Criterion '{k}' has invalid level {v!r}, expected wrong/partial/correct")
                    levels[k] = v
                    observations[k] = obs
                missing = set(weights) - set(levels)
                if missing:
                    raise ValueError(f"Judge response missing criteria: {missing}")
                total_weight = sum(weights.values())
                weighted = sum(LEVEL_VALUE[levels[k]] * weights[k] for k in weights) / total_weight
                return {
                    "score": max(0.0, min(1.0, weighted)),
                    "criteria": levels,
                    "observations": observations,
                    "reasoning": parsed.get("reasoning", ""),
                    "raw_response": raw_text,
                    "attempts": attempt + 1,
                }
            except Exception as e:
                last_error = e
                continue

        return {
            "score": 0.0,
            "criteria": None,
            "reasoning": None,
            "raw_response": raw_text,
            "error": str(last_error),
            "attempts": n_retries + 1,
        }
