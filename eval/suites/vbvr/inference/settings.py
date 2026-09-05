"""
InternVL-U inference settings for the VBVR eval pipeline.

Scope boundary (deliberately kept narrow): this module owns only the parts
of running inference that are constant across every sample -- generation
mode and image preprocessing. It does NOT own prompt text. Each sample's
prompt is decided once, at data-construction time, and stored as part of
that sample (training jsonl record or eval test-sample record) -- the
inference runner just reads the prompt field off the record it's given, it
never builds or rewrites prompt text itself. Keeping prompt construction out
of this file means a sample means the same thing whether it's replayed from
disk, logged, or re-scored later; the inference step has no side effect on
what "the prompt" was.

The video-to-image prompt rewriting design discussed 2026-08-25 (universal
wrapper + per-task cleanup for ball_bounces_given_time / rotation_puzzle /
maze) is still the intended approach -- it belongs in whatever pipeline
constructs the actual jsonl/sample records (training data and eval test
samples alike), not here.
"""

GENERATION_MODE = "image"  # InternVL-U always runs single-image generation for this eval; never video.

# Applied to every input image (first_frame, or any additional context frames)
# before it reaches the model. Resizes the long edge to 512px, preserving
# aspect ratio -- no padding, no forced square. VBVR-Bench source images are
# not uniformly sized across tasks (e.g. 512x512 vs 800x400), so this is a
# real normalization step, not a no-op.
IMAGE_PREPROCESSING = {
    "target_long_edge": 512,
    "preserve_aspect_ratio": True,
    "pad_to_square": False,
}


def resize_long_edge(image, target_long_edge: int = IMAGE_PREPROCESSING["target_long_edge"]):
    """Resize `image` (HxWxC array) so its longer edge equals target_long_edge,
    preserving aspect ratio. No padding -- output may be non-square."""
    import cv2

    h, w = image.shape[:2]
    scale = target_long_edge / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
    return cv2.resize(image, (new_w, new_h), interpolation=interp)
