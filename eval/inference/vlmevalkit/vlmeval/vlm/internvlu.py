"""Minimal VLMEvalKit wrapper for InternVL-U (understanding side).
Loads the unified pipeline and runs text generation_mode for image+text->text.

To produce results comparable to other VLMEvalKit models (e.g. InternVL, Qwen2.5-VL),
this wrapper implements ``use_custom_prompt`` / ``build_prompt`` so that benchmark
prompts carry the same answer-format instruction those adapters inject (e.g.
"Answer with the option's letter from the given choices directly." for MCQ). Without
it, the model receives the bare dataset prompt and tends to emit hedged / degenerate
text that the answer extractor cannot map to an option, which artificially depresses
scores. See ``vlm/internvl/internvl_chat.py`` for the reference behavior we mirror.
"""
import os
import sys
import torch
from PIL import Image
from .base import BaseModel
from .internvl.utils import build_multi_choice_prompt
from vlmeval.dataset import DATASET_TYPE, DATASET_MODALITY

# Resolve the model inference package (Model_Related) via the central single
# source of truth: Evaluation/paths.py (this file is Evaluation/VLMEvalKit/vlmeval/vlm/).
_EVAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _EVAL_ROOT not in sys.path:
    sys.path.insert(0, _EVAL_ROOT)
from paths import add_internvlu_to_path


class InternVLU(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path, max_new_tokens=2048, use_custom_prompt=True, **kwargs):
        add_internvlu_to_path()
        from internvlu import InternVLUPipeline
        self.pipe = InternVLUPipeline.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        ).to("cuda")
        self.tok = self.pipe.tokenizer
        self.max_new_tokens = max_new_tokens
        self._use_custom_prompt = use_custom_prompt

    # Mirror the standard adapters: emit a per-dataset prompt with the answer-format
    # instruction so the extractor can reliably parse the model's choice.
    def use_custom_prompt(self, dataset):
        if not self._use_custom_prompt or dataset is None:
            return False
        if DATASET_MODALITY(dataset) == 'VIDEO':
            return False
        if DATASET_TYPE(dataset) == 'MCQ' and 'LEGO' in dataset:
            return False
        return DATASET_TYPE(dataset) in ('MCQ', 'Y/N', 'VQA')

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)
        dataset_type = DATASET_TYPE(dataset)
        if dataset_type == 'MCQ':
            prompt = build_multi_choice_prompt(line, dataset)
        elif dataset_type == 'Y/N':
            prompt = line['question'] + '\nAnswer the question using a single word or phrase.'
        else:  # VQA
            prompt = line['question'] + '\nAnswer the question using a single word or phrase.'

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message

    @torch.no_grad()
    def generate_inner(self, message, dataset=None):
        imgs = [Image.open(x["value"]).convert("RGB") for x in message if x["type"] == "image"]
        prompt = "\n".join([x["value"] for x in message if x["type"] == "text"])
        out = self.pipe(
            prompt=[prompt],
            image=[imgs] if imgs else None,
            generation_mode="text",
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        return self.tok.batch_decode(out.generate_output, skip_special_tokens=True)[0].strip()
