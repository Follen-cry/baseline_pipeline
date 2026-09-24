"""Shared S4 prompt text -- the part of the human/GPT turn that is identical
across all 5 sources. Source-specific caption-prefix recovery lives in each
source's own `processing/build_s4_caption_prefix.py`; this module only knows
about the generic "describe the progression, then predict" framing that
sits on top of it.

Design (agreed 2026-09-06, see ../recipes/s4_progression/README.md for the
full decision log): S4 = S2's exact (image, cond_image, target_image) triples,
with S2's MCQ-era closing replaced by a shared progression-description +
next-frame-prediction instruction, and S2's fixed placeholder GPT answer
replaced by real VLM-labeled text (progression description + textual
next-frame prediction) ahead of the same "<img>" generation trigger S0-S3
already use.
"""
from __future__ import annotations

_NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def _count_word(n: int) -> str:
    return _NUM_WORDS.get(n, str(n))


def image_block(n: int) -> str:
    return "".join("<image>\n" for _ in range(n))


PROGRESSION_INSTRUCTION_TEMPLATE = (
    'You are shown {count} consecutive frames, in temporal order. First, in '
    '1-2 sentences (no more than about 40 words), describe how the scene '
    'changes across them, based only on what is visible in these {count} '
    'frames. Then, prefaced with "Next frame prediction:", predict in 1-2 '
    'sentences (no more than about 40 words) what the next frame will look '
    'like.'
)

GEN_TRIGGER = "The next frame should look like this: <img>"
NEXT_FRAME_PREDICTION_MARKER = "Next frame prediction:"

# Domain-framing prefix for sources whose S2 caption jumps straight into an
# instruction with no scene-setting sentence (currently just NWM -- see
# recipes/s4_progression/README.md decision log, 2026-09-06).
NWM_DOMAIN_PREFIX_TEMPLATE = (
    "These are {count} consecutive first-person camera frames from an "
    "egocentric navigation video sequence."
)


def progression_instruction(n: int) -> str:
    return PROGRESSION_INSTRUCTION_TEMPLATE.format(count=_count_word(n))


def nwm_domain_prefix(n: int) -> str:
    return NWM_DOMAIN_PREFIX_TEMPLATE.format(count=_count_word(n))


def build_s4_human_prompt(n: int, caption_prefix: str) -> str:
    """caption_prefix: the source's existing task-specific caption text
    (VBVR scene description / EPIC action / NWM domain-prefix + action /
    panda70m video description), with NO closing instruction attached --
    that's what this function appends, replacing whatever S2 used to
    close with.
    """
    caption_prefix = caption_prefix.strip()
    return (
        image_block(n)
        + caption_prefix + "\n\n"
        + progression_instruction(n)
    )


VLM_LABEL_PROMPT_TEMPLATE = (
    "You are given {total} images. Frames 1-{context_last} are consecutive "
    "frames from a video, in temporal order. Frame {last} is the true next "
    "frame. Context: {caption} First, in 1-2 sentences (no more than about "
    "40 words), describe how the scene changes across Frames 1-{context_last}, "
    "in temporal order. Then, in 1-2 sentences (no more than about 40 "
    "words), describe what Frame {last} looks like. Write both as direct, "
    "factual descriptions -- no hedging language, no meta-commentary like "
    '"in this sequence". Return exactly two sentences-groups in this format:\n'
    "Progression: <your progression description>\n"
    "Next frame: <your description of frame {last}>"
)


def _with_trailing_period(text: str) -> str:
    text = text.strip()
    return text if text.endswith((".", "!", "?")) else text + "."


def build_vlm_label_prompt(n: int, caption_prefix: str) -> str:
    """Prompt for the offline VLM captioning pass that produces S4's GT
    text. Caller passes n+1 images (the n context frames, in order, then
    target_image last) matching this prompt's Frame numbering.

    Exact VLM choice / final prompt wording is still open (deferred
    2026-09-06) -- this is a first draft others should feel free to
    iterate on before the labeling pass actually runs.
    """
    total = n + 1
    return VLM_LABEL_PROMPT_TEMPLATE.format(
        total=total, last=total, context_last=n,
        caption=_with_trailing_period(caption_prefix),
    )
