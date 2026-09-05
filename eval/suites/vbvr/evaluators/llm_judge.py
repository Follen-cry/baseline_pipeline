"""
VLM-judge scoring for VBVR tasks whose rule-based image-only score is known
weak (see Evaluation/VBVR-CustomEval/README.md "Known issues" and the
Tier 3 entries in the Scoring Rules artifact). Used to *supplement*
image_evaluator.py's rule-based scores, not replace them -- report both
side by side.

Talks to a local OpenAI-compatible vision server (vLLM/SGLang). Default
config points at the qwen3-vl-30b-fp8 server already running on this host
(localhost:8000), but both are overridable via env vars so this isn't
hardcoded to one deployment:

    VBVR_JUDGE_BASE_URL   (default: http://localhost:8000/v1)
    VBVR_JUDGE_MODEL      (default: qwen3-vl-30b-fp8)

Usage:
    from llm_judge import VLMJudge, RUBRICS

    judge = VLMJudge()
    result = judge.score(
        task_name="key_door_matching",
        prompt_text=open(".../prompt.txt").read(),
        first_frame_path=".../first_frame.png",
        gen_final_path=".../model_output.png",
        gt_final_path=".../final_frame.png",
    )
    print(result["score"], result["reasoning"])
"""

import base64
import json
import os
import re
from typing import Any, Dict, Optional

from openai import OpenAI

# =============================================================================
# Rubrics -- plain-language criteria for the tasks this judge targets.
# Kept as a flat string per task rather than importing gen_rules_artifact.py's
# TASKS list, since that module has import-time side effects (reads the
# EXAMPLES/VALIDATION json caches) we don't want triggered here.
#
# NOTE: gravity_physics was dropped from this list (and from the locked
# 15-task set entirely, replaced by glass_refraction) on 2026-08-25 -- its
# official generator doesn't clamp the ball at ground level, so ~80% of GT
# samples have an empty final_frame (ball simulated below ground, off
# canvas). The judge correctly scored those as wrong, which is what exposed
# the bug: this was a real upstream data defect, not a judge or rubric issue.
# See image_evaluator.py's GlassRefractionImageEvaluator docstring.
#
# NOTE: ball_bounces_given_time was also dropped from this list on
# 2026-08-25. It briefly had a rubric here to accompany a path_shape rule-
# based metric, but that metric was removed (the generator no longer draws
# the bounce path -- see BallBounceImageEvaluator's docstring), and the
# task's remaining rule-based signal (final_position) is a plain
# deterministic distance check that doesn't need judge supplementation.
# =============================================================================

RUBRICS = {
    "key_door_matching": """Task: an agent must navigate to a key, collect it, and then reach the
door matching that key's color.
Score how well the generated result satisfies:
- Agent movement: did the agent visibly move away from its starting position (not stay put)?
- Key collection: does a key that was visible in the first frame appear collected (agent adjacent to
  it, or key removed/changed) by the final frame?
- Correct door: is the agent positioned at a door whose color matches the key that was collected?
- Plausible order: does the overall scene suggest key-then-door, not door-then-key or door without
  a key?""",
}

DEFAULT_BASE_URL = os.environ.get("VBVR_JUDGE_BASE_URL", "http://localhost:8000/v1")
DEFAULT_MODEL = os.environ.get("VBVR_JUDGE_MODEL", "qwen3-vl-30b-fp8")

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _to_data_uri(path: str, max_side: int = 512) -> str:
    """Downscale + encode an image as a base64 data URI, to keep context usage
    reasonable (the judge server has max_model_len=8192)."""
    from PIL import Image
    from io import BytesIO

    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def build_judge_instruction(task_name: str, prompt_text: str) -> str:
    """
    Build the exact text instruction sent to the judge (everything except the
    3 images themselves). Pulled out as its own function -- rather than
    inlined in VLMJudge.score() -- so the scoring-rules artifact can render
    precisely what the judge sees, not a hand-copied approximation of it.
    """
    rubric = RUBRICS.get(task_name)
    if rubric is None:
        raise ValueError(f"No rubric registered for task '{task_name}'. "
                          f"Available: {list(RUBRICS)}")

    return (
        "You are a strict, deterministic grader for a visual reasoning benchmark. "
        "You will see three images in order: (1) the starting frame, (2) a candidate "
        "generated result, (3) the ground-truth correct result. Judge ONLY image (2) "
        "against the rubric below, using image (3) as the reference for what 'correct' "
        "looks like.\n\n"
        f"Task instruction given to the generator: \"{prompt_text}\"\n\n"
        f"Scoring rubric:\n{rubric}\n\n"
        "Respond with ONLY a JSON object, no other text: "
        '{"score": <integer 0-100>, "reasoning": "<one short sentence>"}'
    )


class VLMJudge:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL,
                 temperature: float = 0.0, max_tokens: int = 300):
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
        instruction = build_judge_instruction(task_name, prompt_text)

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
                raw_score = float(parsed["score"])
                return {
                    "score": max(0.0, min(1.0, raw_score / 100.0)),
                    "raw_score": raw_score,
                    "reasoning": parsed.get("reasoning", ""),
                    "raw_response": raw_text,
                    "attempts": attempt + 1,
                }
            except Exception as e:
                last_error = e
                continue

        return {
            "score": 0.0,
            "raw_score": None,
            "reasoning": None,
            "raw_response": raw_text,
            "error": str(last_error),
            "attempts": n_retries + 1,
        }
