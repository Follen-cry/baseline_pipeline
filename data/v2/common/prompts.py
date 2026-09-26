#!/usr/bin/env python3
"""v2 T0-T4 prompt templates and answer grammar (shared by derive_settings.py and the v2 eval).

One fixed template per setting. Frames are named by their position in the 4-frame window:
F0..F3 (chronological, equally spaced by GAP). Shuffled settings label the shown frames A, B, C.
GAP is written as 0.5 / 1.0 / 2.0 (seconds). Text answers are one line of compact JSON placed
before the generated image; settings without a text answer reply with `<img>` only.

  T0 / T4-A  F0 F1 F2 + GAP                   -> F3
  T1         F0 F1 F2                         -> {"gap"} + F3
  T2         3 of F0..F3 in order + GAP + MISSING (F0 always shown) -> the missing frame
  T3         F0 F1 F2 shuffled as A B C + GAP -> {"order"} + F3
  T4-B       3 of F0..F3 shuffled as A B C (F0 always shown, one of F1..F3 missing)
             -> {"order", "gap", "missing"} + the missing frame

`order` lists the labels from earliest to latest; `missing` is "F1" | "F2" | "F3".
"""
import json

GAPS = (0.5, 1.0, 2.0)
FRAMES = ("F0", "F1", "F2", "F3")
LABELS = ("A", "B", "C")
IMG = "<img>"

_HEAD_ORDERED = ("Here are three consecutive frames sampled from a video clip, in chronological order "
                 "and equally spaced in time.")
_HEAD_SHUFFLED3 = ("Here are three consecutive frames sampled from a video clip, equally spaced in time, "
                   "shown in shuffled order and labeled A, B and C.")
_HEAD_MISSING = ("Here are three observed frames from a four-frame video sequence F0-F3, in chronological "
                 "order and equally spaced in time. One frame position is missing and is specified below.")
_HEAD_SHUFFLED4 = ("Here are three observed frames from a four-frame video sequence F0-F3, equally spaced in "
                   "time, shown in shuffled order and labeled A, B and C. F0 is always observed; one of F1, "
                   "F2, F3 is missing.")

_GAP_LINE = "GAP: {gap} s (time between consecutive frames)"
_GAP_CHOICES = "one of 0.5, 1.0, 2.0"

TASK = {
    "T0": "Generate the next frame F3, which comes one GAP after F2.",
    "T1": ("Predict the temporal gap between consecutive frames, then generate the next frame F3, which "
           "comes one gap after F2.\nAnswer format: first one line of JSON {\"gap\": g}, where g is the gap "
           f"in seconds, {_GAP_CHOICES}; then the frame."),
    "T2": "Generate the missing frame {missing}.",
    "T3": ("Predict their chronological order, then generate the next frame, which comes one GAP after the "
           "latest of them.\nAnswer format: first one line of JSON {\"order\": [...]}, listing the labels "
           "A, B, C from earliest to latest; then the frame."),
    "T4B": ("Predict the chronological order of A, B, C, the temporal gap between consecutive frames of the "
            "full sequence, and which position is missing, then generate the missing frame.\nAnswer format: "
            "first one line of JSON {\"order\": [...], \"gap\": g, \"missing\": \"Fk\"}; order lists A, B, C "
            f"from earliest to latest, g is in seconds ({_GAP_CHOICES}), and missing is one of \"F1\", "
            "\"F2\", \"F3\". Then the frame."),
}


def fmt_gap(gap):
    assert gap in GAPS, gap
    return f"{gap:.1f}"


def answer_json(**kw):
    """Compact one-line JSON with keys in the fixed order order, gap, missing."""
    out = {k: kw[k] for k in ("order", "gap", "missing") if k in kw}
    return json.dumps(out, separators=(",", ":"))


def human_prompt(kind, caption, names, gap=None, missing=None):
    """kind: T0 | T1 | T2 | T3 | T4B. names: the 3 shown image labels, in the order the images are
    given (F-names for T0/T1/T2, A/B/C for T3/T4B)."""
    head = {"T0": _HEAD_ORDERED, "T1": _HEAD_ORDERED, "T2": _HEAD_MISSING,
            "T3": _HEAD_SHUFFLED3, "T4B": _HEAD_SHUFFLED4}[kind]
    lines = [head, "", f"Caption: {caption}"] + [f"{n}: <image>" for n in names]
    if kind in ("T0", "T2", "T3"):
        lines.append(_GAP_LINE.format(gap=fmt_gap(gap)))
    if kind == "T2":
        lines.append(f"MISSING: {missing}")
    task = TASK[kind].format(missing=missing) if kind == "T2" else TASK[kind]  # others hold JSON braces
    lines += ["", task]
    return "\n".join(lines)


def gpt_response(answer=None):
    return f"{answer_json(**answer)}\n{IMG}" if answer else IMG


def parse_answer(text, keys):
    """Parse the JSON line before <img> of a model response. Returns (dict or None, error or None);
    checks that exactly `keys` are present and each value is in its grammar."""
    head = text.split(IMG)[0].strip()
    line = head.splitlines()[0].strip() if head else ""
    try:
        d = json.loads(line)
    except Exception:  # noqa: BLE001
        return None, "not_json"
    if not isinstance(d, dict) or set(d) != set(keys):
        return None, "bad_keys"
    if "order" in d and (not isinstance(d["order"], list) or sorted(d["order"]) != list(LABELS)):
        return None, "bad_order"
    if "gap" in d and d["gap"] not in GAPS:
        return None, "bad_gap"
    if "missing" in d and d["missing"] not in FRAMES[1:]:
        return None, "bad_missing"
    return d, None
