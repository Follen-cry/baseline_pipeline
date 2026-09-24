# `eval/suites/v2_temporal_ssl/` — v2 eval suite

Planned: held-out eval for T0-T4 on the v2 window pool's eval split.

- Text answers: GAP / ORDER / MISSING accuracy, parsed with `data/v2/common/prompts.py`.
- Images: PSNR / SSIM / LPIPS / DINO similarity to ground truth, broken down by Δt and missing position k.
- Every T model also runs the shared T0 forecasting eval, as a common yardstick.

Existing suites (vbvr, worldprediction, ...) stay v1 code; v2 checkpoints are added to them as
extra rows. See `docs/v2.md`.
