"""VLMEvalKit wrapper for SenseNova-U1 (understanding side).

Mirrors the InternVLU adapter: implements ``use_custom_prompt``/``build_prompt``
so benchmark prompts carry the standard answer-format instruction, making the
SenseNova checkpoints comparable to the other adapters under the fair pipeline.

The SFT model thinks in a ``<think>...</think>`` block by default; we strip the
block and return only the committed answer so the rule-based extractor sees the
same kind of text as for other models (applied identically to base and SFTs).
"""
import os
import sys

import torch
from PIL import Image

from .base import BaseModel
from .internvl.utils import build_multi_choice_prompt
from vlmeval.dataset import DATASET_TYPE, DATASET_MODALITY

# Resolve model src + LoRA loader (Model_Related) via Evaluation/paths.py.
_EVAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _EVAL_ROOT not in sys.path:
    sys.path.insert(0, _EVAL_ROOT)
from paths import add_sensenova_to_path

MAX_PIXELS = 1024 * 1024


class SenseNovaU1(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path, max_new_tokens=2048, use_custom_prompt=True, **kwargs):
        add_sensenova_to_path()
        from sensenova_u1.models.neo_unify.utils import load_image_native
        from snu1_lora import load_model_maybe_lora
        self._load_image = load_image_native
        self.model, self.tok = load_model_maybe_lora(model_path, dtype=torch.bfloat16, device="cuda")
        self.max_new_tokens = max_new_tokens
        self._use_custom_prompt = use_custom_prompt

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
        else:
            prompt = line['question'] + '\nAnswer the question using a single word or phrase.'
        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message

    @torch.inference_mode()
    def generate_inner(self, message, dataset=None):
        img_paths = [x["value"] for x in message if x["type"] == "image"]
        prompt = "\n".join([x["value"] for x in message if x["type"] == "text"])

        pixel_values = grid_hw = None
        if img_paths:
            pvs, ghs = [], []
            for p in img_paths:
                pv, gh = self._load_image(p, max_pixels=MAX_PIXELS)
                pvs.append(pv)
                ghs.append(gh)
            pixel_values = torch.cat(pvs, dim=0).to("cuda", dtype=self.model.dtype)
            grid_hw = torch.cat(ghs, dim=0).to("cuda")
            if "<image>" not in prompt:
                prompt = "<image>\n" * len(img_paths) + prompt

        resp = self.model.chat(
            self.tok, pixel_values, prompt,
            dict(max_new_tokens=self.max_new_tokens, do_sample=False),
            grid_hw=grid_hw,
        )
        if "</think>" in resp:
            resp = resp.split("</think>")[-1]
        return resp.strip()
