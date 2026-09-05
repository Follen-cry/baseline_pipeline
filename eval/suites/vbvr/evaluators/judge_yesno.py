"""Yes/no checklist judge -- a redesign of llm_judge_full.py's rubric judge for
tasks where the 3-level (wrong/partial/correct) weighted rubric has saturated.

Why this exists (rotation_puzzle, measured 2026-09-03 over all 500 generations
in results/vbvr_target_pred_eval/{base,s0,s1,s2,s3}_4task500sft):

  * The rubric judge emitted only THREE distinct scores -- 1.0 (418/500 = 84%),
    0.6 (36), 0.5 (44), plus 2 outliers.
  * The `partial` level was used ZERO times out of 2000 criterion verdicts, so
    the 3-level scale was really a 2-level one.
  * Two of the four criteria were constants: rotation_accuracy came back
    "correct" 498/500 and position_preservation 499/500. Together they carry
    50% of the weight, i.e. half of every score was a fixed +0.5 floor. That is
    the whole reason the five variants sat at 0.904/0.933/0.918/0.927/0.936 --
    a ~0.03 spread that cannot separate models at n=100.

So the criteria were not measuring; they were mostly asserting. This module
replaces them with a flat checklist of independent yes/no questions, each one
phrased so "yes" means the candidate matches the reference in that one respect.
Score = (number of yes) / (number of questions). That makes the number of
questions the score granularity, explicitly: N questions -> N+1 possible scores,
evenly spaced, no weights to hand-tune.

Three deliberate differences from llm_judge_full.py, per the 2026-09-03 request:

  1. TWO images, not three -- the candidate and the ground truth. The starting
     frame is dropped.
  2. NO task instruction text. The 2026-08-27 post-mortem in llm_judge_full.py
     traced the judge's confabulation to it grading against what the instruction
     ASKED FOR rather than what the pixels showed. Removing the instruction
     removes that prior entirely: the judge has nothing to be led by except the
     two images, and every question is "does the candidate match the reference
     in respect X", which is answerable from those two images alone.
  3. Every question is binary and independently checkable, with the failure it
     is meant to catch fixed in advance (see QUESTION_PROVENANCE below).

The rotation_puzzle question set mirrors, one for one, the quantities the
deterministic rewritten scorer (scorers/rotation_puzzle.py) actually measures:
4 per-cell arm-sets + 4 junction closures + the 3 scene-level gates (background,
tile grid, colour) that scorer applies as ceilings. That is not a coincidence --
it is the point. The two systems become directly comparable per sample.
"""

import base64
import json
import re
from io import BytesIO
from typing import Any, Dict, List, Tuple

from llm_judge import DEFAULT_BASE_URL, DEFAULT_MODEL, _to_data_uri
from openai import OpenAI

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

# task_name -> (scene_desc, [(question_key, question_text), ...])
#
# Question text is written to be answerable ONLY by comparing the two images,
# never by reasoning about what the task "should" produce.
TASK_QUESTIONS: Dict[str, Tuple[str, List[Tuple[str, str]]]] = {
    "rotation_puzzle": (
        "Both images show the same puzzle board: four square tiles laid out in a "
        "2x2 grid on a plain background, with a coloured pipe drawn inside each "
        "tile. The two images should differ in nothing at all if the candidate is "
        "perfect.",
        [
            ("board_intact", "Ignoring the pipes themselves: does the candidate show the same board as the reference -- the same plain background colour (the canvas has NOT been repainted, e.g. black), four square tiles in a 2x2 grid at the same positions and the same sizes, and pipes drawn in the same colour?"),
            ("cells_correct", "In EVERY one of the four tiles, does the candidate's pipe have its two arms pointing in the same two directions (up / down / left / right) as the pipe in the reference's corresponding tile? Answer false if even one tile differs, is empty, or holds something that is not a single two-armed pipe."),
            ("joins_closed", "At EVERY one of the four borders between adjacent tiles, does a pipe run all the way out to that border from the tiles on BOTH sides? Do NOT require the two pipes to physically touch: in the reference they stop just short of each other, leaving a small blank gap, and that counts as joined; a candidate that instead draws straight through the gap also counts as joined. Answer false if at even one border a pipe stops short of it or is absent."),
            ("no_extra_marks", "Does the candidate contain NO extra coloured marks beyond what the reference shows -- no additional bar, line, stub or stroke sticking out of a tile, crossing a tile, or sitting anywhere the reference is empty -- even if everything the reference DOES show is present and correct?"),
        ],
    ),
}

# task_name -> (readout_instruction, [slot_name, ...])
#
# The readout is the fix for the one failure this design hit in its first
# smoke test. With the questions alone, the judge answered "yes" to all 11 on
# s1/00068 -- a near-blank candidate with a single stub of pipe -- reasoning
# "all visual elements in the candidate match the reference exactly". It did
# the same on s3/00059, where all four elbows are at the wrong rotation.
# Deterministic, reproduced at temperature 0.
#
# The cause was NOT that the judge cannot see the candidate. Asked to describe
# s1/00068 on its own, with no reference image and no questions, it answers
# correctly and precisely ("top-left: completely white, contains no pipe ...
# bottom-left: a pipe with one arm up and one arm down"). Perception is fine;
# what collapses is the comparison -- given both images and a block of yes/no
# questions, it answers the gestalt "do these match?" once and stamps it
# across all 11 slots.
#
# So the judge is made to do the thing it demonstrably does well first:
# enumerate each image ALONE, as structured data, before any question is put
# to it. This is not a repeat of the reverted 2026-08-27 "observation" change
# (see llm_judge_full.py's docstring). That one asked for free-text prose
# justifying a verdict, with the task instruction in context to confabulate
# toward; this asks for a fixed-shape enumeration of one image at a time, and
# there is no task instruction in the prompt at all.
#
# NOTE: the absolute readout is not always right -- the judge sometimes names
# each elbow by its outward corner rather than its inward arms, so the same
# reference board comes back as NW/NE/SW/SE in one call and SE/SW/NE/NW in
# another. It is nonetheless read the SAME way for both images within a single
# call, so the comparison it feeds is sound. That is why the cell answers are
# left to the judge instead of being recomputed in Python from the readout:
# the readout is reliable as a relative encoding, not as an absolute one.
TASK_READOUT: Dict[str, Tuple[str, List[str]]] = {
    "rotation_puzzle": (
        "For each of the four tiles, report the arms of the pipe it contains as a "
        "list drawn from up/down/left/right, or [] if that tile holds no pipe at "
        "all. An arm counts only if you can actually see the pipe running to that "
        "side of the tile.",
        ["top_left", "top_right", "bottom_left", "bottom_right"],
    ),
}

# Which observed failure each question exists to catch. Kept next to the
# questions so a later edit can check it is not deleting the only detector for
# a real failure mode. Sample ids are from the 4task500sft runs.
QUESTION_PROVENANCE = {
    "board_intact": "base/00058 renders the pipes on a solid black canvas with no tiles. Fires on only 2/500 -- the weakest slot, kept as the sole dedicated detector for a candidate that repaints or re-lays-out the whole scene.",
    "cells_correct": "s3/00059 (all four elbows at the wrong rotation), base/00072 (straight bars instead of elbows), base/00049 and s1/00068 (pipes missing entirely). Fires on 71/500.",
    "joins_closed": "the two calibration cases in rotation_puzzle_notes.md, where an arm stops 36-52 px short of a 110 px tile edge so the ring does not close. Fires on 45/500.",
    "joins_closed_CONFOUND": "MEASURED 2026-09-03, unresolved. Despite the question "
        "stating explicitly that a candidate drawing straight through the gutter "
        "'also counts as joined', the judge does not honour that clause: joins_closed "
        "answers no on 52-73% of candidates that draw through the gutter but only "
        "14-27% of GT-style ones. Because the variants differ in how often they use "
        "that rendering (base 37% of generations cross the gutter, s1 15%), the "
        "between-variant spread is inflated by rendering style. Restricting to "
        "GT-style candidates only, the spread falls from 0.100 to 0.053 (deterministic "
        "CV scorer on the same subset: 0.039; rubric judge: 0.032). So roughly half "
        "the apparent separation is style, not skill. Do not quote the 0.100 figure "
        "without this caveat.",
    "no_extra_marks": "base/00028 and s2/00057 draw the correct ring PLUS a spurious vertical bar out of the top-right tile. Every other question asks 'is X present and matching', so a candidate that ADDS something the reference lacks passes them all. CAVEAT, measured on 200 samples of the 12-question variant: this question fired on 24.5% of samples yet still scored base/00028 a perfect 1.00 -- it did not catch the case it was written for. Treat its rate as noisy until it is validated against hand-labelled extra-mark cases.",
}

# The join questions were rewritten once, on evidence, and the reason matters
# for anyone editing them again.
#
# They first read "...do the pipes both reach that border and MEET there, the
# same way they do in the reference?". Two measurements killed that wording:
#
#  1. Self-consistency. Scoring each ground-truth image against ITSELF -- two
#     byte-identical images, where the only correct answer is 11/11 -- gave
#     13/15 perfect. The two failures were 7/11, and in both the four join
#     questions failed as a block. Every other question was 0/15 false-negative.
#     A question that fails on identical inputs is measuring nothing.
#
#  2. Why. The reference images do NOT show a closed loop. Each elbow stops at
#     its own tile edge and the gutter between tiles is left blank: measured
#     over the gutter bands, ground-truth ink there is exactly 0.000 in every
#     sample checked. So "do they meet, the same way they do in the reference"
#     is a contradiction -- in the reference they do not meet at all. Candidates
#     split into two renderings, both of which solve the puzzle: GT-style with
#     the gap (gutter ink 0.000, e.g. base/00003) and one continuous rectangle
#     drawn straight through the gutters (gutter ink 0.038-0.128, e.g.
#     base/00026, 00031, 00042). The judge was penalising both, inconsistently.
#
# The wording now states the gap explicitly, says both renderings count, and
# asks only whether a pipe REACHES the border from each side -- which is
# exactly what scorers/rotation_puzzle.py measures with ARM_GAP_PX, so the two
# systems are asking the same question of the same pixels.


def build_yesno_instruction(task_name: str) -> str:
    """Build the exact text sent to the judge (everything except the 2 images).
    Separate function, same reasoning as llm_judge_full.build_full_judge_instruction:
    the report artifact renders precisely what the judge saw."""
    entry = TASK_QUESTIONS.get(task_name)
    if entry is None:
        raise ValueError(f"No question set registered for task '{task_name}'. "
                         f"Available: {list(TASK_QUESTIONS)}")
    scene_desc, questions = entry

    readout_instr, slots = TASK_READOUT[task_name]
    q_block = "\n".join(f"{i}. {key}: {text}" for i, (key, text) in enumerate(questions, 1))
    keys_block = ", ".join(f'"{key}": true|false' for key, _ in questions)
    slots_block = ", ".join(f'"{s}": [...]' for s in slots)

    return (
        "You are a strict, deterministic grader for a visual reasoning benchmark.\n\n"
        "You will see exactly two images:\n"
        "  Image A -- the CANDIDATE, produced by a model. This is what you grade.\n"
        "  Image B -- the REFERENCE, the correct result.\n\n"
        f"{scene_desc}\n\n"
        f"STEP 1 -- Read out image A ALONE, ignoring image B completely. "
        f"{readout_instr}\n"
        f"STEP 2 -- Read out image B ALONE, the same way.\n"
        "STEP 3 -- Only now, answer the yes/no questions below, strictly from "
        "your two readouts and from the pixels.\n\n"
        "Answer each question with a strict yes or no. Answer only from what is "
        "actually drawn in the pixels. Do not reason about what the puzzle "
        "'ought' to look like, and do not give a candidate the benefit of the "
        "doubt: if you cannot see the specific detail the question asks about, "
        "the answer is false. Each question is independent -- answer it on its "
        "own, even if an earlier answer was false.\n\n"
        f"{q_block}\n\n"
        "Respond with ONLY a JSON object, no other text:\n"
        f'{{"A_readout": {{{slots_block}}}, "B_readout": {{{slots_block}}}, '
        f'"answers": {{{keys_block}}}, "reasoning": "<one short sentence>"}}'
    )


def _truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "yes", "y", "1"):
        return True
    if s in ("false", "no", "n", "0"):
        return False
    raise ValueError(f"Not a yes/no value: {v!r}")


class YesNoJudge:
    """Same client/interface shape as FullVLMJudge, so scoring drivers can swap
    one for the other; `score()` takes only the two image paths it uses."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL,
                 temperature: float = 0.0, max_tokens: int = 800):
        self.client = OpenAI(base_url=base_url, api_key="not-needed")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def score(self, task_name: str, gen_final_path: str, gt_final_path: str,
              n_retries: int = 2) -> Dict[str, Any]:
        entry = TASK_QUESTIONS.get(task_name)
        if entry is None:
            raise ValueError(f"No question set registered for task '{task_name}'.")
        _, questions = entry
        keys = [k for k, _ in questions]

        instruction = build_yesno_instruction(task_name)
        content = [
            {"type": "text", "text": instruction},
            {"type": "text", "text": "Image A (CANDIDATE -- grade this one):"},
            {"type": "image_url", "image_url": {"url": _to_data_uri(gen_final_path)}},
            {"type": "text", "text": "Image B (REFERENCE -- the correct result):"},
            {"type": "image_url", "image_url": {"url": _to_data_uri(gt_final_path)}},
        ]

        last_error, raw_text = None, None
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
                raw_answers = parsed["answers"]
                answers = {k: _truthy(raw_answers[k]) for k in keys}  # KeyError -> retry
                n_yes = sum(answers.values())
                return {
                    "score": n_yes / len(keys),
                    "answers": answers,
                    "n_yes": n_yes,
                    "n_questions": len(keys),
                    # kept for auditing: lets a later pass check whether a
                    # disputed answer is a perception failure or a comparison one
                    "readout": {"A": parsed.get("A_readout"), "B": parsed.get("B_readout")},
                    "reasoning": parsed.get("reasoning", ""),
                    "raw_response": raw_text,
                    "attempts": attempt + 1,
                }
            except Exception as e:                  # noqa: BLE001 - recorded, retried
                last_error = e
                continue

        return {"score": None, "answers": None, "n_yes": None, "readout": None,
                "n_questions": len(keys), "reasoning": None,
                "raw_response": raw_text, "error": str(last_error),
                "attempts": n_retries + 1}


if __name__ == "__main__":
    print(build_yesno_instruction("rotation_puzzle"))
