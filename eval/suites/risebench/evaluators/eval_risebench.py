#!/usr/bin/env python
"""RISEBench GPT-judge scoring — a close port of PhoenixZ810/RISEBench's
official `gpt_eval.py` + `utils.py` (fetched from
https://github.com/PhoenixZ810/RISEBench on 2026-09-09; upstream has no
tagged release, so this pins to that date's `main`).

Same three judge dimensions as upstream, same per-category prompt
selection, same score/completion formulas — only the plumbing differs:

- Judge backend: any OpenAI-compatible chat-completions endpoint (real
  OpenAI by default) instead of upstream's raw `requests.post` against a
  hardcoded `api_base`. Configure via `--api_key`/`--api_base_url`/`--model`
  or the `OPENAI_API_KEY`/`RISEBENCH_JUDGE_BASE_URL`/`RISEBENCH_JUDGE_MODEL`
  env vars (mirrors `suites/vbvr/evaluators/llm_judge.py`'s env-var
  convention, and `Evaluation/editing_benchmarks/PICABench/PicaEval_gpt.py`'s
  `--api_key`/`--api_base_url` flag names). Default model is "gpt-5"
  (matching this repo's other GPT-judge scripts,
  e.g. PicaEval_gpt.py's `--gpt_model` default) rather than upstream's
  "gpt-4.1-2025-04-14" — override with `--model` to reproduce the paper's
  exact judge.
- Data access: reads through `dataset.py`'s `RiseBenchItem` (extended with
  the judge-only fields `consistency_free`/`reasoning_img`/`reference_img`/
  `reference_txt`/`reasoning_wo_ins` upstream's `eval_vanilla` branches on)
  instead of a raw pandas DataFrame.
- Resume cache: a flat JSON `{index: {judge1, judge2, judge3}}` instead of
  upstream's pickle, keyed and flushed the same way (skip already-judged
  indices on a re-run).
- Concurrency: `ThreadPoolExecutor` across items (like upstream's
  `track_progress_rich`), each item still making its up-to-3 judge calls
  sequentially.

## Per-category judge calls (ported from `eval_vanilla`)

Every item gets a **Reasoning** call (image[s] + rubric vs. the reference
description/image — "does the output match what should have happened").
temporal_reasoning / causal_reasoning / spatial_reasoning also get a
**VisualPlausibility** call (image quality/realism alone) and, unless the
item is `consistency_free`, an **ApprConsistency** call (candidate vs. a
reference image — "did anything change that shouldn't have"). logical_
reasoning skips VisualPlausibility entirely (its score formula has no term
for it) and always gets an ApprConsistency call, but the two images it
compares depend on which of `reference_txt`/`reference_img` the item
carries — see `build_judge_calls` below, which mirrors `eval_vanilla`
line-for-line rather than simplifying, since the branch structure (which
image plays "img1" shifts per-category and even per-item) is upstream's
actual scoring definition, not incidental plumbing.

## Score/completion formulas (ported from `calculate_score`/`calculate_completion`)

    temporal/causal/spatial, not consistency_free:
        score = 0.3*ApprConsistency + 0.5*Reasoning + 0.2*VisualPlausibility
    temporal/causal/spatial, consistency_free:
        score = 0.2*VisualPlausibility + 0.8*Reasoning
    logical:
        score = 0.3*ApprConsistency + 0.7*Reasoning
    if Reasoning == 1 (worst case): score = max(score * 0.5, 1)

`complete` (a 0/1 "did every dimension hit the max score" indicator) and
the final report's Score-Percentage (`25*(score-1)`, mapping the 1-5 scale
onto 0-100) are ported the same way.

Usage (env: needs `openai`; judge calls cost real API credits):
    OPENAI_API_KEY=... python evaluators/eval_risebench.py \\
        --data_root /scratch/local/ssd/junlin/data/RISEBench \\
        --generated /scratch/local/ssd/junlin/results/RISEBench/InternVL-U/images \\
        --save_path /scratch/local/ssd/junlin/results/RISEBench/InternVL-U \\
        --limit 10   # smoke-test a handful of items first (Validate-then-run)
"""
import argparse
import base64
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_SUITE_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _SUITE_ROOT)
from dataset import RiseBenchItem, load_items, filter_missing  # noqa: E402

DEFAULT_BASE_URL = os.environ.get("RISEBENCH_JUDGE_BASE_URL", "https://api.openai.com/v1")
DEFAULT_MODEL = os.environ.get("RISEBENCH_JUDGE_MODEL", "gpt-5")

# Sentinel for "the API call itself failed after all retries" -- distinct from
# Python None, which means "this judge call was never made because the item
# doesn't need it" (see GPTJudge.query's docstring).
_API_FAILURE = "[JUDGE_API_CALL_FAILED]"

# subtask -> main-task prefix, purely for the aggregate report's row labels
# (ported verbatim from upstream gpt_eval.py's subtask_dic).
SUBTASK_PREFIX = {
    "Temp": ["Life Progression", "Material Progression", "Environmental Cycles", "Societal Transformation"],
    "Causal": ["Structural Deformation", "State Transition", "Chemical and Biological Transformation",
               "Physics Manifestation"],
    "Spa": ["Component Assembly", "Object Arrangement", "Viewpoint Generation", "Structural Inference",
            "Layout Reasoning"],
    "Logic": ["Pattern Prediction", "Mathematical Derivation", "Puzzle Solving"],
}


# ============================================================================
# Prompts — ported verbatim from PhoenixZ810/RISEBench's utils.py.
# ============================================================================

prompt_consist = """You are a highly skilled image evaluator. You will receive two images (an original image and a modified image) along with a specific modification instruction. The second image is known to have been altered based on this instruction, starting from the first image. Your task is to evaluate whether the two images maintain consistency in aspects not related to the given instruction.

## Task
Evaluate the consistency between the images according to the following scale (1 to 5):

- **5 (Perfect Consistency)**: Apart from changes explicitly required by the instruction, all other details (e.g., personal features, clothing, background, layout, colors, positions of objects) are completely identical between the two images.

- **4 (Minor Differences)**: Apart from changes explicitly required by the instruction, the second image is mostly consistent with the original image but contains a minor discrepancy (such as a missing minor personal feature, accessory, or tattoo).

- **3 (Noticeable Differences)**: Apart from changes explicitly required by the instruction, the second image has one significant difference from the original (such as a noticeable alteration in a person's appearance like hair or skin color, or a significant change in background environment).

- **2 (Significant Differences)**: Apart from changes explicitly required by the instruction, the second image has two or more significant differences or multiple noticeable inconsistencies (such as simultaneous changes in both personal appearance and background environment).

- **1 (Severe Differences)**: Apart from changes explicitly required by the instruction, nearly all key details (e.g., gender, major appearance features, background environment, or scene layout) significantly differ from the original image, clearly deviating from the original.

Example:

Original image: A blond, white-skinned man with a tattoo on his right shoulder, furniture in the background.
Instruction: "Show him after gaining fifty pounds."

- **Score 5**: A heavier blond, white-skinned man, tattoo on right shoulder intact, identical furniture and layout.
- **Score 4**: A heavier blond, white-skinned man, missing the tattoo on his right shoulder, identical furniture and layout.
- **Score 3**: A heavier man with black hair instead of blond (change in hair color), or original blond man but with a grassy background instead of furniture.
- **Score 2**: A heavier man with black hair (hair color changed), and the background changed to grass.
- **Score 1**: A heavier black-haired woman, and background changed to grass.

Note: When assigning scores, only consider details unrelated to the instruction. Changes explicitly requested by the instruction should NOT be regarded as inconsistencies.

## Input

**Instruction:** {instruct}

## Output Format

Provide a detailed, step-by-step explanation of your scoring process. Conclude clearly with the final score, formatted as:

**Final Score:** **1-5**"""

prompt_reasoning = """You are an expert image evaluator. For each task, you will be provided with:

1. An **instruction** describing how an image should be modified.
2. A **ground-truth textual description** that represents the intended result of the modification.
3. An **output image** generated by an assistant.

Your task is to assess the output image based on the following evaluation dimension:

## Evaluation Dimension: Alignment Between Image and Reference Description
Assess how accurately the output image aligns with the visual content described in the reference description, considering the context of the instruction.

**Scoring Criteria:**
- **5**: The image completely matches the description, accurately reflecting every detail and degree.
- **4**: The image mostly matches the description, with minor discrepancies.
- **3**: The image partially matches the description but contains differences or lacks some details.
- **2**: The image contains noticeable difference. Important details are missed or clearly inaccurate.
- **1**: The image fails to follow the instruction and does not correspond to the description at all.

**Example**
Instruction: Draw what it will look like after it is broken.
Description: An egg is completely broken, with eggshell scattered around and egg white and yolk clearly spilling out.
- **5**: Completely broken egg, clearly scattered eggshells, visible egg white and yolk spilling out.
- **4**: Broken egg, eggshell present but not fully scattered, clearly visible egg white and yolk spilling out.
- **3**: Broken egg with scattered eggshell, but egg white and yolk not spilled or still within eggshell.
- **2**: Only scattered eggshell visible, without clear egg white or yolk.
- **1**: Egg is intact, not broken.

## Input
**Instruction**  {instruct}
**GroundTruth Description:** {reference}

## Output Format

Provide a detailed, step-by-step explanation of your scoring process. Conclude clearly with the final score, formatted as:

**Final Score:** **X**
"""

prompt_reasoning_w_input = """You are an expert image evaluator. For each task, you will be provided with:

1. An original image.
2. An **instruction** describing how an image should be modified.
3. A **ground-truth textual description** that represents the intended result of the modification.
4. An **output image** generated by an assistant.

Your task is to assess the output image based on the following evaluation dimension:

## Evaluation Dimension: Alignment Between Image and Reference Description
Assess how accurately the output image aligns with the visual content described in the reference description, considering the context of the instruction.

**Scoring Criteria:**
- **5**: The image completely matches the description, accurately reflecting every detail and degree.
- **4**: The image mostly matches the description, with minor discrepancies.
- **3**: The image partially matches the description but contains differences or lacks some details.
- **2**: The image contains noticeable difference. Important details are missed or clearly inaccurate.
- **1**: The image fails to follow the instruction and does not correspond to the description at all.

**Example**
Instruction: Draw what it will look like after it is broken.
Description: An egg is completely broken, with eggshell scattered around and egg white and yolk clearly spilling out.
- **5**: Completely broken egg, clearly scattered eggshells, visible egg white and yolk spilling out.
- **4**: Broken egg, eggshell present but not fully scattered, clearly visible egg white and yolk spilling out.
- **3**: Broken egg with scattered eggshell, but egg white and yolk not spilled or still within eggshell.
- **2**: Only scattered eggshell visible, without clear egg white or yolk.
- **1**: Egg is intact, not broken.

## Input
**Instruction**  {instruct}
**GroundTruth Description:** {reference}

## Output Format

Provide a detailed, step-by-step explanation of your scoring process. Conclude clearly with the final score, formatted as:

**Final Score:** **X**
"""

prompt_generation = """You are an expert image evaluator. For each task, you will be provided with an **output image** generated by an assistant.

Your task is to independently assess the image along the following dimension and assign an integer score from **1 to 5**:

### Evaluation Dimension: Realism and Generation Quality

Assess the overall visual realism and generation fidelity of the image. Consider the image's clarity, natural appearance, and compliance with physical plausibility and real-world constraints.

**Scoring Guidelines:**

- **5** The image is sharp, visually coherent, and all elements appear highly realistic and physically plausible.
- **4** The image is clear, with most elements appearing realistic; minor details may show slight unreality.
- **3** The image is mostly clear, but some significant elements appear unrealistic or physically implausible.
- **2** The image is noticeably blurry or contains major unrealistic components or visual distortions.
- **1** The image is extremely blurry, incoherent, or severely unrealistic; realism is nearly absent.

## Output Format

After the evaluation, conclude clearly with the final score, formatted as:

**Final Score:** **X**
"""

prompt_spatial_ref = """You are an expert image evaluator. For each task, you will be provided with:

1. An **instruction** describing how an image should be modified.
2. A **ground-truth textual description** that represents the intended result of the modification.
3. An **output image** generated by an assistant.

Your task is to assess the output image based on the following evaluation dimension:

## Evaluation Dimension: Alignment Between Image and Reference Description
Assess how accurately the output image aligns with the visual content described in the reference description, considering the context of the instruction.

**Scoring Criteria:**
- **5**: The image completely matches the description, accurately reflecting every detail and degree.
- **4**: The image mostly matches the description, with minor discrepancies.
- **3**: The image partially matches the description but contains differences or lacks some details.
- **2**: The image contains noticeable difference. Important details are missed or clearly inaccurate.
- **1**: The image fails to follow the instruction and is entirely unrelated to the description.

## Input
**Instruction**  {instruct}
**GroundTruth Description:** {reference}

## Output Format

Conclude clearly with the final score, formatted as:

**Final Score:** **X**"""

prompt_spatial_ref_img = """You are an expert image evaluator. For each task, you will be provided with:

1. An **instruction** describing how an image should be modified.
2. A **reference image** that represents the intended result of the modification.
3. An **output image** generated by an assistant.

Your task is to assess the output image based on the following evaluation dimension:

## Evaluation Dimension: Alignment Between Output Image and reference Image
Assess how accurately the output image aligns with the reference image, considering the context of the instruction.

**Scoring Criteria:**
- **5**: The image completely matches the reference image and fully follows the instruction.
- **4**: The image mostly matches the reference image, with minor discrepancies.
- **3**: The image partially matches the reference image but contains differences or lacks some details.
- **2**: The image contains noticeable difference. Important details are missed or clearly inaccurate.
- **1**: The image fails to follow the instruction and is entirely unrelated to the reference image.

## Input
**Instruction**  {instruct}

## Output Format

Conclude clearly with the final score, formatted as:

**Final Score:** **X**"""

prompt_spatial_ref_w_input = """You are an expert image evaluator. For each task, you will be provided with:

1. An original image.
2. An **instruction** describing how the original image should be modified.
3. A **ground-truth textual description** that represents the intended result of the modification.
4. An **output image** generated by an assistant.

Your task is to assess the output image based on the following evaluation dimension:

## Evaluation Dimension: Alignment Between Image and Reference Description
Assess how accurately the output image aligns with the reference description based on the original image, considering the context of the instruction.

**Scoring Criteria:**
- **5**: The image completely matches the description, accurately reflecting every detail and degree.
- **4**: The image mostly matches the description, with minor discrepancies.
- **3**: The image partially matches the description but contains differences or lacks some details.
- **2**: The image contains noticeable difference. Important details are missed or clearly inaccurate.
- **1**: The image fails to follow the instruction and is entirely unrelated to the description.

## Input
**Instruction**  {instruct}
**GroundTruth Description:** {reference}

## Output Format

Conclude clearly with the final score, formatted as:

**Final Score:** **X**"""

prompt_spatial_qual = """You are a highly skilled image evaluator. Given an image, your task is to assess and determine its clarity and distortion, and then provide a score (an integer between 1 and 5) based on the following criteria:

## Task Requirements:

Determine whether the image has blurriness, distortion, visual defects, or physical inaccuracies.

Assign an appropriate score to the image based on the above criteria, considering its overall quality and detail integrity.

## Scoring Criteria:

- **5 points**: The image is very clear, with complete details, and no noticeable distortion or blurriness. All elements conform to physical laws.
- **4 points**: The image is clear, with only minor blurriness, and no noticeable distortion.
- **3 points**: The image has areas with clarity issues, such as slight blurriness or distortion. Some elements are physically incorrect.
- **2 points**: The image has noticeable blurriness or distortion, with significant detail loss, or lacks physical accuracy.
- **1 point**: The image is severely blurry or distorted, making it difficult to recognize its content, with serious degradation in visual quality, almost unusable.

Do not penalize the image for being rendered in 3D.

## Output Format

Provide a clear conclusion with the final score, formatted as follows:

**Final Score:** **1-5**

where X represents the score."""

prompt_spatial_cons = """You are a precise and analytical image consistency evaluator.

You will be given:
- **Image A**: the original image.
- **Image B**: a modified version of Image A.
- **Instruction**: a directive describing the intended modification to Image A to produce Image B.

Your task is to **evaluate how consistent Image B remains with Image A in all aspects *except* those explicitly changed by the instruction.** You must **ignore the instructed changes** and **only assess unintended differences**.

## Evaluation Scale (1 to 5):

- **5** Perfect Consistency
  All elements not related to the instruction are visually identical between Image A and Image B (e.g., style, background, object positions, colors, shapes). No unintended change is present.
- **4** Minor Difference
  One small unintended change is present (e.g., a slight color variation or minor object shape shift), but overall the image remains highly consistent.
- **3** Noticeable Difference
  One major or a few minor unintended changes are present (e.g., an object's shape, color, or background differs noticeably, or style has shifted slightly).
- **2** Significant Inconsistency
  Two or more significant differences unrelated to the instruction (e.g., changes in both object details and background or style), reducing overall fidelity.
- **1** Severe Inconsistency
  Major unintended changes dominate the image (e.g., altered visual style, scene layout, or appearance), clearly breaking consistency with Image A.

> ⚠️ Note:
> - To receive a score of 5, the modified image must be visually identical to the original in every unaffected aspect—symbols, patterns, background, texture, color, category, layout, and style must all match exactly.
> - If the background in the original is vague (e.g., plain white or composed of parts), and the background in Image B is also similar vague, you may disregard background consistency.
> - If a blue diamond shape appears in the bottom-left corner of Image 2, ignore it; it is a watermark.

## Example

**Original image**: "A silver-framed clock with a white face. Three hands (hour, minute, second) are disassembled and lie beside it."
**Instruction**: "Assemble the clock to show 9:45."

**Scoring Criteria:**
- **Score 5**: Frame, face, and hand shapes exactly as original.
- **Score 4**: One hand differs slightly in shape or thickness.
- **Score 3**: All hands identical, differing from original specs, or some other things(like text, furniture in the background) is added.
- **Score 2**: Frame color or face differs, and hand shapes are wrong.
- **Score 1**: Frame, face, and hand appearance all significantly altered, background is totally different.

## Input
**Instruction:** {instruct}

## Output Format
After evaluation, conclude with:

**Final Score:** **1-5**
"""

prompt_logical_cons_ans = """**You are a highly skilled image evaluator.** Given an image with logical problem, you will receive:

1. **Image 1**: The original image.
2. **Image 2**: A generated image from an assistant model.
3. **Problem Description**
4. **Reference Answer**

## Evaluation Task

* Compare Image 2 against Image 1 in terms of visual style, environment, and composition.
* Score each comparison as **1** (consistent) or **0** (inconsistent).

## Consistency Criteria (score 1 if true):

* Color palette, line weight, font/handwriting style, arrangement, and background setting closely match.
* Only the problem's solution differs (e.g., added or removed marks), or color brightness is slightly lighter/darker.
* Image 2 is derived by overlaying a pattern onto Image 1 without altering style or layout.
* **If a blue diamond shape appears in the bottom-left corner of Image 2, ignore it; it is a watermark.**

## Inconsistency Example (score 0):

* Image 1 shows a quadrilateral with unequal edges and arbitrary angles, but Image 2 depicts a perfect square with equal edges and right angles.

## Inputs
**Problem Description**:
{instruct}
**Reference Answer**:
{reference}

## Output
You should provide a step-by-step explanation of how you arrived at the score and conclude in the format:

**Final Score**: **X**
"""

prompt_logical_cons = """**You are a highly skilled image evaluator.** Given an image with logical problem, you will receive:

1. **Image 1**: The original image.
2. **Image 2**: A generated image answer from an assistant model.
3. **Problem Description**

## Evaluation Task

* Judge the appearance consistency between Image 1 and Image 2.
* Score each comparison as **1** (consistent) or **0** (inconsistent).

## Consistency Criteria (score 1 if true):

* Color palette, line weight, font/handwriting style, arrangement, and background setting closely match.
* Only the problem's solution differs (e.g., added or removed marks), or color brightness is slightly lighter/darker.
* Image 2 is derived by overlaying a pattern onto Image 1 without altering style or layout.
* **If a blue diamond shape appears in the bottom-left corner of Image 2, ignore it; it is a watermark.**

## Inconsistency Example (score 0):

* Image 1 shows a quadrilateral with unequal edges and arbitrary angles, but Image 2 depicts a perfect square with equal edges and right angles.
* Image 2 contains severe blur and distortion.

## Inputs
**Problem Description**:
{instruct}

## Output
Conclude in the format:

**Final Score**: **X**
"""

prompt_logical_txt = """**You are a highly skilled image evaluator.** Given an image with logical problem, you will receive:

1. **Image**: A generated image answer from an assistant model.
2. **Reference Answer**

## Task
Assign a binary score (0 or 1) for **Logical Correctness**:
- **1** if the Generated Image accurately implements the Reference Answer.
- **0** otherwise.

## Requirements
- The original image is not given and do not imagine the original image, just determine whether the generated image matches the reference answer.
- **If the given image contains more content than reference answer, give score 0. For example, reference answer is \\"a square\\" but image contains more squares or rectangles, give score 0.**

## Inputs
**Reference Answer**:
{reference}

## Output
You should provide a step-by-step explanation of how you arrived at the score and conclude in the format:

**Final Score**: **X**
"""

prompt_logical_img = """**You are a highly skilled image evaluator.** Given a logical problem, you will receive:

1. **Problem Description**
2. **Image 1**: A reference ground-truth image answer that correctly solves the problem.
3. **Image 2**: A generated image answer from an assistant model.

## Logical Correctness (0/1)
Determine whether the content of *Image 2* solves the problem in the same way as *Image 1* does.

### Scoring rules:

- Score **1** if *Image 2* exactly matches *Image 1* in terms of problem-solving logic and visual outcome.
- Score **0** if there are any differences that affect the correctness of the solution.
**If generated image answer is significantly different from ground-truth, directly score 0.**

### Examples:

- For a tic-tac-toe task, if the positions of all marks in *Image 2* are identical to those in *Image 1*, score 1; otherwise, score 0.
- For a shape-drawing task, *Image 2* must match *Image 1* precisely in shape, color, pattern arrangement, and orientation to receive a score of 1; any deviation results in a score of 0.
- If *Image 1* shows only one correct answer but *Image 2* includes multiple answers, score 0.

## Problem Description
{instruct}

## Output
You should provide a step-by-step explanation of how you arrived at the score and conclude in the format:

**Final Score**: **X**
"""

prompt_logical_img_wo_q = """**You are a highly skilled image evaluator.** You will receive:

1. **Image 1**: An image represents the correct answer.
2. **Image 2**: A generated image that is claimed to solve the problem.

## Task
Determine iwhether Image 1 and Image 2 are the same. Give 1 when same and 0 when not same. Do not consider size and background.
DO NOT GIVE 1 WHEN IMAGE 2 CONTAINS MORE CONTENT THAN IMAGE 1.
WHEN IMAGE 2 IS A PUZZLE AND IMAGE 1 IS NOT AN ANSWER, GIVE 0.

## Output
You should provide your explaination and give score in the format:

**Final Score**: **X**
"""


# ============================================================================
# OpenAI-compatible judge client
# ============================================================================

def _to_data_uri(path: str, max_side: int = 768) -> str:
    """Downscale + JPEG-encode an image as a base64 data URI (768px matches
    upstream gpt_generate's default `image_size`)."""
    from PIL import Image
    img = Image.open(path)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _build_content(parts: List[Tuple[str, str]]) -> list:
    """parts: [("text", str), ("image", path), ...] -> OpenAI chat content list."""
    content = []
    for kind, value in parts:
        if kind == "text":
            content.append({"type": "text", "text": value})
        elif kind == "image":
            content.append({"type": "image_url", "image_url": {"url": _to_data_uri(value)}})
        else:
            raise ValueError(f"unknown content part kind: {kind}")
    return content


# Reasoning models (gpt-5.x, o1/o3/o4) reject `max_tokens` (need
# `max_completion_tokens`) and reject any non-default `temperature` --
# same constraint noted in Evaluation/maze/maze_route_judge.py.
_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _is_reasoning_model(model: str) -> bool:
    return model.lower().startswith(_REASONING_MODEL_PREFIXES)


class GPTJudge:
    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL,
                 temperature: float = 0.0, max_tokens: int = 1024):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.is_reasoning = _is_reasoning_model(model)

    def query(self, parts: List[Tuple[str, str]], n_retries: int = 5) -> str:
        """Ported from upstream gpt_generate's retry loop and count. Returns
        `_API_FAILURE` (never Python None) on exhaustion -- `None` is
        reserved for "this judge call wasn't needed at all" (a
        `consistency_free` item's skipped consistency check), so a real API
        failure can't be silently mistaken for that legitimate skip by
        `scores_from_judge`."""
        content = _build_content(parts)
        kwargs = dict(model=self.model, messages=[{"role": "user", "content": content}])
        if self.is_reasoning:
            kwargs["max_completion_tokens"] = self.max_tokens
        else:
            kwargs["max_tokens"] = self.max_tokens
            kwargs["temperature"] = self.temperature
        last_err = None
        for attempt in range(1, n_retries + 1):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content.strip()
            except Exception as e:  # noqa: BLE001 — mirror upstream's blanket retry
                last_err = e
                if attempt < n_retries:
                    time.sleep(3)
        print(f"[judge] giving up after {n_retries} attempts: {last_err}", flush=True)
        return _API_FAILURE


# ============================================================================
# Per-category judge-call construction — ported from eval_vanilla.
# ============================================================================

def build_judge_calls(item: RiseBenchItem, gen_path: str) -> Dict[str, Optional[List[Tuple[str, str]]]]:
    """Returns {"consist": parts_or_None, "reasoning": parts, "quality": parts_or_None}
    -- the exact message content each judge call needs for this item, mirroring
    `eval_vanilla`'s per-category branching (including which image plays "img1"
    for the *consistency* check, which upstream reassigns per-branch)."""
    instruct = item.instruction
    img2 = gen_path
    img1 = item.input_path
    judge_rea_require_img = False
    prompt_qua = None

    if item.category in ("temporal_reasoning", "causal_reasoning"):
        reference = item.reference
        if item.reasoning_img:
            judge_rea_require_img = True
            prompt_rea = prompt_reasoning_w_input.format(instruct=instruct, reference=reference)
        else:
            prompt_rea = prompt_reasoning.format(instruct=instruct, reference=reference)
        prompt_cons = prompt_consist.format(instruct=instruct)
        prompt_qua = prompt_generation

    elif item.category == "spatial_reasoning":
        if item.reference_img_path:
            judge_rea_require_img = True
            img1 = item.reference_img_path
            prompt_rea = prompt_spatial_ref_img.format(instruct=instruct)
        elif item.reasoning_img:
            judge_rea_require_img = True
            prompt_rea = prompt_spatial_ref_w_input.format(instruct=instruct, reference=item.reference)
        else:
            prompt_rea = prompt_spatial_ref.format(instruct=instruct, reference=item.reference)
        prompt_cons = prompt_spatial_cons.format(instruct=instruct)
        prompt_qua = prompt_spatial_qual

    elif item.category == "logical_reasoning":
        if item.reference_txt:
            prompt_cons = prompt_logical_cons_ans.format(instruct=instruct, reference=item.reference_txt)
            prompt_rea = prompt_logical_txt.format(instruct=instruct, reference=item.reference_txt)
        elif item.reference_img_path:
            judge_rea_require_img = True
            img1 = item.reference_img_path
            prompt_cons = prompt_logical_cons.format(instruct=instruct)
            prompt_rea = prompt_logical_img_wo_q if item.reasoning_wo_ins else prompt_logical_img.format(instruct=instruct)
        else:
            raise ValueError(f"logical_reasoning item {item.index} has neither reference_txt nor reference_img")

    else:
        raise ValueError(f"unknown category: {item.category}")

    consist_parts = None
    if not item.consistency_free:
        consist_parts = [("text", prompt_cons), ("image", img1), ("image", img2)]

    if judge_rea_require_img:
        reasoning_parts = [("text", prompt_rea), ("image", img1), ("image", img2)]
    else:
        reasoning_parts = [("text", prompt_rea), ("image", img2)]

    quality_parts = None
    if prompt_qua is not None:
        quality_parts = [("text", prompt_qua), ("image", img2)]

    return {"consist": consist_parts, "reasoning": reasoning_parts, "quality": quality_parts}


def judge_item(judge: GPTJudge, item: RiseBenchItem, gen_path: str) -> Dict[str, Optional[str]]:
    calls = build_judge_calls(item, gen_path)
    judge1 = judge.query(calls["consist"]) if calls["consist"] is not None else None
    judge2 = judge.query(calls["reasoning"])
    judge3 = judge.query(calls["quality"]) if calls["quality"] is not None else None
    result = {"judge1": judge1, "judge2": judge2}
    if calls["quality"] is not None:
        result["judge3"] = judge3
    return result


# ============================================================================
# Score extraction/aggregation — ported verbatim from gpt_eval.py's main().
# ============================================================================

def extract(answer: Optional[str]) -> Optional[List[int]]:
    if answer is None:
        return None
    matches = re.findall(r'\*?\*?Final Score\*?\*?:?\s*([\d*\s,\n]*)', answer, re.IGNORECASE)
    numbers = []
    if matches:
        for match in matches:
            extracted_numbers = re.findall(r'\d+', match.replace('\n', ' '))
            if extracted_numbers:
                numbers.extend(map(int, extracted_numbers))
                break
        if numbers:
            return numbers

    matches = re.findall(r'\*?\*?Final Scores\*?\*?:?\s*([\d*\s,\n]*)', answer, re.IGNORECASE)
    numbers = []
    if matches:
        for match in matches:
            extracted_numbers = re.findall(r'\d+', match.replace('\n', ' '))
            if extracted_numbers:
                numbers.extend(map(int, extracted_numbers))
                break
        return numbers
    return None


def calculate_score(row: Dict[str, Any]) -> Optional[float]:
    if row["Reasoning"] is None:
        return None
    if row["category"] in ("temporal_reasoning", "causal_reasoning", "spatial_reasoning"):
        if row.get("consistency_free"):
            score = 0.2 * row["VisualPlausibility"] + 0.8 * row["Reasoning"]
        else:
            score = 0.3 * row["ApprConsistency"] + 0.5 * row["Reasoning"] + 0.2 * row["VisualPlausibility"]
    elif row["category"] == "logical_reasoning":
        score = 0.3 * row["ApprConsistency"] + 0.7 * row["Reasoning"]
    else:
        raise ValueError(row["category"])
    if row["Reasoning"] == 1:
        score = score * 0.5
        score = 1 if score < 1 else score
    return score


def calculate_completion(row: Dict[str, Any]) -> Optional[int]:
    if row["Reasoning"] is None:
        return None
    is_consistency_free = bool(row.get("consistency_free"))
    if row["category"] in ("temporal_reasoning", "causal_reasoning", "spatial_reasoning"):
        if is_consistency_free:
            return 1 if row["Reasoning"] == 5 and row["VisualPlausibility"] == 5 else 0
        return 1 if row["ApprConsistency"] == 5 and row["Reasoning"] == 5 and row["VisualPlausibility"] == 5 else 0
    elif row["category"] == "logical_reasoning":
        return 1 if row["ApprConsistency"] == 5 and row["Reasoning"] == 5 else 0
    raise ValueError(row["category"])


def scores_from_judge(judge: Dict[str, Optional[str]]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """(ApprConsistency, Reasoning, VisualPlausibility) from a raw {judge1,judge2,judge3?}
    dict, following main()'s exact branch-by-shape logic (which of the 1-5 vs.
    binary 0/1 scales applies depends on which judge calls were made)."""
    has_quality = "judge3" in judge
    if judge["judge1"] is None:
        # consistency_free: only reasoning (+ quality, always present when consistency_free
        # applies, since that flag only appears for temporal/causal/spatial).
        score2 = extract(judge["judge2"])
        score3 = extract(judge["judge3"]) if has_quality else None
        if not score2 or (has_quality and not score3):
            return None, None, None
        return None, score2[0], (score3[0] if has_quality else None)

    if not has_quality:
        # logical_reasoning: judge1/judge2 are binary 0/1 -> rescale onto the 1-5 axis.
        score1 = extract(judge["judge1"])
        score2 = extract(judge["judge2"])
        if not score1 or not score2:
            return None, None, None
        cons = 4 * min(score1[0], 1) + 1
        reas = 4 * min(score2[0], 1) + 1
        return cons, reas, None

    score1 = extract(judge["judge1"])
    score2 = extract(judge["judge2"])
    score3 = extract(judge["judge3"])
    if not score1 or not score2 or not score3:
        return None, None, None
    return score1[0], score2[0], score3[0]


def trans_to_percent(s: Optional[float]) -> Optional[float]:
    if s is None:
        return None
    return 25 * (s - 1)


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="RISEBench data_root (has datav2_total_w_subtask.json, data/)")
    ap.add_argument("--generated", required=True, help="<out>/images dir from gen_risebench_internvlu.py")
    ap.add_argument("--save_path", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY"))
    ap.add_argument("--api_base_url", default=DEFAULT_BASE_URL)
    ap.add_argument("--nproc", type=int, default=4, help="concurrent judge workers (items in flight, not API calls)")
    ap.add_argument("--limit", type=int, default=0, help="smoke-test the first N items (0 = all)")
    ap.add_argument("--max_tokens", type=int, default=None,
                     help="response token budget; default 4096 for reasoning models (gpt-5.x/o1/o3/o4 -- "
                          "their internal reasoning tokens count against this budget too, and 1024 was "
                          "observed to sometimes leave zero tokens for the actual answer), else 1024")
    args = ap.parse_args()

    if not args.api_key:
        raise RuntimeError("An API key is required: pass --api_key or set OPENAI_API_KEY.")

    items = filter_missing(load_items(args.data_root))
    if args.limit:
        items = items[: args.limit]

    os.makedirs(args.save_path, exist_ok=True)
    cache_path = os.path.join(args.save_path, f"{args.model}_judge_cache.json")
    cache: Dict[str, Dict[str, Optional[str]]] = {}
    if os.path.exists(cache_path):
        cache = json.load(open(cache_path))

    max_tokens = args.max_tokens or (4096 if _is_reasoning_model(args.model) else 1024)
    judge = GPTJudge(api_key=args.api_key, base_url=args.api_base_url, model=args.model, max_tokens=max_tokens)

    todo = []
    missing_gen = 0
    for it in items:
        cached = cache.get(it.index)
        if cached is not None and _API_FAILURE not in cached.values():
            continue  # a real success is cached; a failed-call entry is retried below
        gen_path = os.path.join(args.generated, it.gen_filename)
        if not os.path.exists(gen_path):
            missing_gen += 1
            continue
        todo.append((it, gen_path))
    if missing_gen:
        print(f"[eval] WARNING: {missing_gen} items have no generated image (skipped)", flush=True)
    print(f"[eval] {len(items)} items total, {len(todo)} need judging "
          f"({len(items) - missing_gen - len(todo)} already cached)", flush=True)

    if todo:
        from tqdm import tqdm
        with ThreadPoolExecutor(max_workers=args.nproc) as pool:
            futures = {pool.submit(judge_item, judge, it, gen_path): it for it, gen_path in todo}
            for fut in tqdm(as_completed(futures), total=len(futures)):
                it = futures[fut]
                cache[it.index] = fut.result()
                json.dump(cache, open(cache_path, "w"), indent=2)

    rows = []
    for it in items:
        judge_result = cache.get(it.index)
        if judge_result is None:
            rows.append({"index": it.index, "category": it.category, "subtask": it.subtask,
                         "consistency_free": it.consistency_free,
                         "ApprConsistency": None, "Reasoning": None, "VisualPlausibility": None,
                         "match_log": "no_output"})
            continue
        cons, reas, qual = scores_from_judge(judge_result)
        rows.append({"index": it.index, "category": it.category, "subtask": it.subtask,
                     "consistency_free": it.consistency_free,
                     "ApprConsistency": cons, "Reasoning": reas, "VisualPlausibility": qual,
                     "match_log": "succeed" if reas is not None else "failed"})

    for row in rows:
        row["score"] = calculate_score(row)
        row["complete"] = calculate_completion(row)

    per_item_path = os.path.join(args.save_path, f"{args.model}_risebench_judged.json")
    json.dump(rows, open(per_item_path, "w"), indent=2)
    print(f"[eval] wrote {per_item_path}", flush=True)

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    by_cat = {c: [r for r in rows if r["category"] == c]
              for c in ("temporal_reasoning", "causal_reasoning", "spatial_reasoning", "logical_reasoning")}

    final_score = {
        "Overall": [_mean([r["score"] for r in rows]), None, _mean([r["complete"] for r in rows])],
        "Overall_Reasoning": [_mean([r["Reasoning"] for r in rows]), None, None],
        "Overall_ApprConsistency": [_mean([r["ApprConsistency"] for r in rows]), None, None],
        "Overall_VisualPlausibility": [_mean([r["VisualPlausibility"] for r in rows]), None, None],
    }
    for label, cat in (("Temporal", "temporal_reasoning"), ("Causal", "causal_reasoning"),
                       ("Spatial", "spatial_reasoning"), ("Logical", "logical_reasoning")):
        cat_rows = by_cat[cat]
        final_score[label] = [_mean([r["score"] for r in cat_rows]), None, _mean([r["complete"] for r in cat_rows])]
        final_score[f"{label}_Reasoning"] = [_mean([r["Reasoning"] for r in cat_rows]), None, None]
        final_score[f"{label}_Consistency"] = [_mean([r["ApprConsistency"] for r in cat_rows]), None, None]
        if cat != "logical_reasoning":
            final_score[f"{label}_Quality"] = [_mean([r["VisualPlausibility"] for r in cat_rows]), None, None]

    subtasks = sorted({r["subtask"] for r in rows if r["subtask"]})
    for subtask in subtasks:
        sub_rows = [r for r in rows if r["subtask"] == subtask]
        prefix = next((p for p, names in SUBTASK_PREFIX.items() if subtask in names), None)
        label = f"{prefix}-{subtask}" if prefix else subtask
        final_score[label] = [_mean([r["score"] for r in sub_rows]), None, _mean([r["complete"] for r in sub_rows])]

    for key, (score, _, acc) in final_score.items():
        final_score[key][1] = trans_to_percent(score)

    metrics_path = os.path.join(args.save_path, "evaluation_metrics.json")
    json.dump(final_score, open(metrics_path, "w"), indent=2)
    print(f"[eval] wrote {metrics_path}", flush=True)
    print(json.dumps({k: v for k, v in final_score.items()
                       if k in ("Overall", "Temporal", "Causal", "Spatial", "Logical")}, indent=2))


if __name__ == "__main__":
    main()
