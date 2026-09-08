"""Minimal VLMEvalKit wrapper for Cosmos3-Nano (understanding side).

Delegates to the shared `Cosmos3Generator` (Evaluation/predict_future/models/
cosmos3_wrapper.py), which loads `Cosmos3OmniForConditionalGeneration`
directly via transformers (see that module's docstring for why vLLM serving,
NVIDIA's documented path, isn't usable on this cluster's driver).

Mirrors InternVLU/SenseNovaU1's `use_custom_prompt`/`build_prompt` contract so
benchmark prompts carry the same answer-format instruction those adapters
inject (e.g. "Answer with the option's letter..." for MCQ) -- without it the
model tends to emit hedged/degenerate text the answer extractor can't map to
an option, which artificially depresses scores.
"""
import os
import sys

from PIL import Image

from .base import BaseModel
from .internvl.utils import build_multi_choice_prompt
from vlmeval.dataset import DATASET_TYPE, DATASET_MODALITY

# Resolve the shared model wrapper via Evaluation/predict_future/models (this
# file is Evaluation/VLMEvalKit/vlmeval/vlm/).
_EVAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _EVAL_ROOT not in sys.path:
    sys.path.insert(0, _EVAL_ROOT)
_PREDICT_FUTURE = os.path.join(_EVAL_ROOT, "predict_future")
if _PREDICT_FUTURE not in sys.path:
    sys.path.insert(0, _PREDICT_FUTURE)


class Cosmos3(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path, max_new_tokens=256, use_custom_prompt=True, **kwargs):
        from models.cosmos3_wrapper import Cosmos3Generator
        self.gen = Cosmos3Generator(model_path=model_path)
        self.max_new_tokens = max_new_tokens
        self._use_custom_prompt = use_custom_prompt

    def use_custom_prompt(self, dataset):
        if not self._use_custom_prompt or dataset is None:
            return False
        if DATASET_MODALITY(dataset) == 'VIDEO':
            return False
        if DATASET_TYPE(dataset) == 'MCQ' and 'LEGO' in dataset:
            return False
        # Omni3DBench's own build_prompt asks for the final answer wrapped in
        # <ans></ans> tags, which its own scorer (Omni3DBench_acc) requires to
        # extract a bare value from a free-form response -- our generic VQA
        # override ("answer in one word/phrase") drops that instruction
        # entirely. Silently gave Nano nonzero MCQ/Numeric scores anyway
        # (Nano naturally answers tersely without reasoning, so its raw
        # output often already equals a bare parseable value) but broke
        # Cosmos3-Edge completely (its answer is always the last line of a
        # long reasoning block, never a bare value) -- confirmed by reading
        # raw prediction text, not inferred from the 0.0 score alone.
        if str(dataset) == 'Omni3DBench':
            return False
        return DATASET_TYPE(dataset) in ('MCQ', 'Y/N', 'VQA')

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)
        dataset_type = DATASET_TYPE(dataset)
        if dataset_type == 'MCQ':
            prompt = build_multi_choice_prompt(line, dataset)
        else:  # Y/N, VQA
            prompt = line['question'] + '\nAnswer the question using a single word or phrase.'
        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message

    def generate_inner(self, message, dataset=None):
        imgs = [Image.open(x["value"]).convert("RGB") for x in message if x["type"] == "image"]
        prompt = "\n".join([x["value"] for x in message if x["type"] == "text"])
        return self.gen.generate_text(
            imgs, prompt, max_new_tokens=self.max_new_tokens, do_sample=False
        )
