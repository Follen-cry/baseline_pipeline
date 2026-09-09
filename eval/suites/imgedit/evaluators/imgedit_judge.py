"""Local-transformers port of ImgEdit-Bench's official Basic-Bench judge.

Upstream (`Benchmark/Basic/basic_bench.py`) calls GPT-4o through the OpenAI
API; this module instead loads the `ImgEdit_Judge` checkpoint upstream also
publishes (a Qwen2.5-VL-7B-Instruct fine-tune, ~8.3B params, bf16) and runs it
locally via `transformers.generate()` -- same idea as
`eval/suites/vbvr/evaluators/llm_judge_transformers.py`'s relationship to
`llm_judge_full.py`'s vLLM-server judge, but there was no vLLM-server version
of this judge to begin with, so this is the only implementation.

Message construction (chat-template + `qwen_vl_utils.process_vision_info`,
`min_pixels`/`max_pixels` = upstream's own training-time settings) follows the
demo code in the upstream README's "Setups for ImgEdit-Judge" section
verbatim, not the generic PIL-image chat-content style
`llm_judge_transformers.py` uses for a different (non-Qwen2.5-VL) judge model
-- ImgEdit_Judge was specifically trained against `qwen_vl_utils`' image
preprocessing, so deviating from it risks a train/inference mismatch.

Output format also differs from vbvr's judges: this one is NOT JSON. It's
free text ending in three "Label: N" lines (one per rubric perspective,
1-5 each), exactly like the GPT-4o judge's output --
`extract_scores_and_average` below is `step1_get_avgscore.py`'s parsing
logic ported verbatim (average of every "<label>: <digit>" line found,
whatever the labels are called -- they differ by edit_type, e.g. "Prompt
Compliance" for `adjust` vs "Style Fidelity" for `style` -- see
judge_prompt.json).

    from imgedit_judge import ImgEditJudge, load_prompts
    judge = ImgEditJudge()  # loads the model once, ~15-30s
    prompts = load_prompts(".../singleturn/judge_prompt.json")
    result = judge.score(
        edit_type="adjust", prompt=prompts["adjust"],
        edit_prompt="Change the tortoise's shell texture to a smooth surface.",
        original_path=".../animal/000342021.jpg",
        result_path=".../images/adjust/1082.png",
    )
    print(result["score"], result["raw_response"])
"""
import json
import os
import re
from typing import Any, Dict, Optional

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

DEFAULT_MODEL_PATH = "/scratch/network/ssd2/junlin/models/ImgEdit_Judge"

# Upstream README: "we train our model with this settings" -- not a
# precautionary default, deviating from these risks the judge seeing images
# preprocessed differently than during its own fine-tuning.
MIN_PIXELS = 1016064
MAX_PIXELS = 1354752


def load_prompts(prompts_json_path: str) -> Dict[str, str]:
    with open(prompts_json_path) as f:
        return json.load(f)


_SCORE_VALUE_RE = re.compile(r"^(\d+)(\s*\([^)]*\))?$")  # "4" or "4 (Good)"


def extract_scores_and_average(text: str) -> Optional[float]:
    """Port of step1_get_avgscore.py's extract_scores_and_average, with one
    deliberate loosening: upstream requires the value after "<label>: " to be
    pure digits (`parts[1].isdigit()`), but ImgEdit_Judge (unlike the GPT-4o
    judge upstream's parser was written against) sometimes appends a word
    annotation, e.g. "Instruction Compliance: 2 (Fair)" instead of
    "Instruction Compliance: 2" -- measured on this repo's own runs, this
    hits ~47% of `background` items and ~70% of `compose` items (the judge
    model's own instruction-following gap, not a prompt issue -- the rubric
    text in judge_prompt.json asks for the plain-digit format in both cases),
    silently dropping them from the per-edit_type average under strict
    parsing. The regex below still requires the WHOLE value to be either a
    bare digit or "digit (word...)" -- anchored on both ends, not a loose
    leading-digit match -- specifically so a "Brief reasoning: ..." line that
    happens to start its sentence with a digit can never be mistaken for a
    score line."""
    scores = []
    for line in text.splitlines():
        parts = line.strip().split(": ", 1)
        if len(parts) != 2:
            continue
        m = _SCORE_VALUE_RE.match(parts[1].strip())
        if m:
            scores.append(int(m.group(1)))
    if scores:
        return round(sum(scores) / len(scores), 2)
    return None


class ImgEditJudge:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, device_map="auto",
                 max_new_tokens: int = 512):
        self.processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            # attn_implementation left at the transformers default (sdpa), not
            # upstream's "flash_attention_2" -- this cluster's driver can't
            # run the separately-compiled flash-attn extension (see
            # llm_judge_transformers.py's docstring for the same finding
            # against a different judge model). sdpa ships inside torch.
            device_map=device_map,
        )
        self.model.eval()
        offloaded = {k: v for k, v in getattr(self.model, "hf_device_map", {}).items()
                     if v in ("cpu", "disk")}
        if offloaded:
            raise RuntimeError(f"Refusing to judge: modules offloaded off-GPU: {offloaded}")
        self.max_new_tokens = max_new_tokens

    def score(
        self,
        edit_type: str,
        prompt: str,
        edit_prompt: str,
        original_path: str,
        result_path: str,
    ) -> Dict[str, Any]:
        full_prompt = prompt.replace("<edit_prompt>", edit_prompt)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": full_prompt},
                {"type": "image", "image": original_path},
                {"type": "image", "image": result_path},
            ],
        }]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw_response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]

        return {
            "edit_type": edit_type,
            "raw_response": raw_response,
            "score": extract_scores_and_average(raw_response),
        }
