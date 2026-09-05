"""
A drop-in replacement for llm_judge_full.FullVLMJudge that runs the judge
model locally via `transformers` instead of calling an OpenAI-compatible
vLLM server -- built 2026-08-31 to use Qwen3.8-27B as judge after vLLM
itself proved unable to serve it on this cluster (see the serve scripts
under Evaluation/GenEditEvalKit/: the community AWQ-INT4 quant crashes
vLLM's Marlin repack kernel, and the official bf16 checkpoint's
tensor-parallel path hits CUDA-13-only compiled extensions -- flash-attn,
custom-all-reduce -- that this cluster's driver, 530.30.02, cannot run).
Plain `transformers.generate()` with attn_implementation="sdpa" sidesteps
all of that (sdpa ships inside torch itself, not a separately-compiled
extension), at the cost of raw generate() -- no PagedAttention, no
continuous batching -- so it is meaningfully slower per call.

Reuses TASK_CRITERIA and build_full_judge_instruction from llm_judge_full
unchanged -- identical rubric, weights, and JSON-parsing/scoring logic, so
scores from this judge are dimension-by-dimension comparable to the
existing qwen3-vl-30b-fp8 judge's. The only thing that differs is which
model is asked.

    from llm_judge_transformers import TransformersVLMJudge
    judge = TransformersVLMJudge()  # loads the model once, ~10s
    result = judge.score(task_name=..., prompt_text=..., first_frame_path=...,
                          gen_final_path=..., gt_final_path=...)
"""
import json
import re
from typing import Any, Dict

import torch
torch.backends.cudnn.enabled = False  # Conv3d vision patch_embed in bf16 hits
# "GET was unable to find an engine to execute this computation" with cuDNN
# on this GPU/cuDNN version; non-cuDNN conv fallback is fine (one conv per
# image, not a hot loop).
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from llm_judge_full import TASK_CRITERIA, build_full_judge_instruction, LEVEL_VALUE

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_first_json_object(text: str):
    """Find the first BALANCED {...} object in text, not the greedy
    first-{-to-last-} span _JSON_BLOCK_RE gives. Qwen3.8, even with
    enable_thinking=False, sometimes appends a trailing sentence after the
    JSON (unlike the original qwen3-vl-30b-fp8 judge this rubric/prompt was
    written against) -- if that trailing text contains its own stray '}'
    (e.g. a parenthetical), the greedy regex overshoots past the real
    object's closing brace and json.loads fails with "Extra data". Scanning
    brace depth from the first '{' and stopping the instant it returns to 0
    is immune to that; only falls back to the greedy regex (matching the
    original judge's behavior) if no balanced object is found at all.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None

DEFAULT_MODEL_PATH = "/scratch/local/ssd/junlin/models/Qwen3.8-27B-bf16"
# Manual device_map, not device_map="auto": accelerate's balancer is overly
# conservative on this box's currently-free GPUs and silently offloads a
# few layers (or lm_head) to disk, which is catastrophic for latency and
# gives no error -- see the earlier bf16-inference benchmark session. This
# 40/24 layer split across GPU7 (fully free, 46GB) and GPU0 (~26-28GB free,
# shared with another long-running process) was verified to place every
# module on-GPU with headroom to spare. Adjust if GPU availability changes.
DEFAULT_DEVICE_MAP_SPLIT = 40
DEFAULT_GPU_IDS = (7, 0)  # physical GPU ids behind CUDA_VISIBLE_DEVICES


def build_device_map(num_layers: int = 64, split: int = DEFAULT_DEVICE_MAP_SPLIT):
    device_map = {
        "model.visual": 0,
        "model.language_model.embed_tokens": 0,
        "model.language_model.norm": 1,
        "model.language_model.rotary_emb": 1,
        "lm_head": 0,
    }
    for i in range(num_layers):
        device_map[f"model.language_model.layers.{i}"] = 0 if i < split else 1
    return device_map


class TransformersVLMJudge:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, temperature: float = 0.0,
                 max_tokens: int = 400, device_map=None,
                 enable_thinking: bool = False, reasoning_effort: str = "low"):
        # enable_thinking=False by default: reasoning_effort="low" was tested
        # and is NOT actually short for this model -- it still produces a
        # long, unstructured visual analysis and blows through max_tokens=400
        # before ever emitting the JSON (measured: 238s, truncated, unparseable).
        # enable_thinking=False (chat template injects an empty <think></think>
        # block, skipping the reasoning step outright) is ~5x faster (46s vs
        # 238s+failure on the same task) and reliably produces clean, valid
        # JSON with sensible criteria-level reasoning in the "reasoning" field.
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device_map or build_device_map(),
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        self.model.eval()
        offloaded = {k: v for k, v in self.model.hf_device_map.items() if v in ("cpu", "disk")}
        if offloaded:
            raise RuntimeError(f"Refusing to judge: modules offloaded off-GPU: {offloaded}")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort

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
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "text", "text": "Image 1 (starting frame):"},
                {"type": "image", "image": Image.open(first_frame_path).convert("RGB")},
                {"type": "text", "text": "Image 2 (candidate generated result -- grade this one):"},
                {"type": "image", "image": Image.open(gen_final_path).convert("RGB")},
                {"type": "text", "text": "Image 3 (ground-truth correct result, for reference):"},
                {"type": "image", "image": Image.open(gt_final_path).convert("RGB")},
            ],
        }]

        last_error = None
        raw_text = None
        for attempt in range(n_retries + 1):
            try:
                inputs = self.processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt",
                    enable_thinking=self.enable_thinking, reasoning_effort=self.reasoning_effort,
                ).to(self.model.device)
                with torch.no_grad():
                    out = self.model.generate(
                        **inputs, max_new_tokens=self.max_tokens,
                        do_sample=self.temperature > 0,
                        temperature=self.temperature if self.temperature > 0 else None,
                    )
                raw_text = self.processor.decode(
                    out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                )
                block = _extract_first_json_object(raw_text)
                if block is None:
                    match = _JSON_BLOCK_RE.search(raw_text)  # last-resort fallback
                    block = match.group(0) if match else None
                if block is None:
                    raise ValueError(f"No JSON object found in response: {raw_text!r}")
                parsed = json.loads(block)
                criteria_dict = dict(parsed["criteria"])
                stray_reasoning = criteria_dict.pop("reasoning", None)  # occasionally
                # emitted nested inside "criteria" instead of as a sibling key
                if stray_reasoning is not None and not parsed.get("reasoning"):
                    parsed["reasoning"] = stray_reasoning
                levels = {}
                observations = {}
                for k, v in criteria_dict.items():
                    if isinstance(v, dict):
                        obs = v.get("observation", "")
                        v = v.get("level", "")
                    else:
                        obs = ""
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
