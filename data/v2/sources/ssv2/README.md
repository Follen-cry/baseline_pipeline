# `data/v2/sources/ssv2/` — Something-Something v2 (v2 source)

> **v2 pool (2026-09-26):** all 10,000 clips of `selection_10k.jsonl` are candidates (no trim to
> 5K). Caption = `label`, verbatim.

## Filtering & sampling summary

**Category-stratified (24 physics templates).**

1. Category filter: 24 physics templates chosen from the label JSON (falling, rolling, sliding, toppling, spinning, projectile, collision, fluids; 15 core + 9 add-ons).
2. Pool: labeled train + validation clips of those templates.
3. Sampling: water-fill over templates, seed 42 — 5 small templates take all they have, the rest 430–431 each (8,336 train + 1,664 validation ids).
4. Extraction: one streaming pass over the local archive, keeping only selected ids.
5. No CV / VLM QC yet (mostly handheld, 240p, 12 fps).
6. Final: `selection_10k.jsonl` = 10,000 (0.86 GB); **to be trimmed to 5K**.

Details below.

## Raw videos (already on disk, nothing to download)

The full SSv2 video release is on torrnode11's local disk, from an earlier MVP setup:
`/scratch/local/ssd/junlin/data/mvp/videos/20bn-something-something-v2-{00,01}` (10.0 GB +
9.44 GB split gzip tar, **220,847 `.webm`**, 12 fps, mostly 240p, median ≈ 4.25 s). It is not
extracted; stream it (`cat …-00 …-01 | tar -xz …`) and keep only selected ids.
`…/mvp/videos/ssv2/*.mp4` holds MVP's 7,142-clip subset, already converted to mp4.

## Labels → `/scratch/network/ssd/junlin/raw/ssv2/labels/`

`labels.json` (174 templates), `train.json` (168,913), `validation.json` (24,777),
`test.json` (27,157 ids, **no labels**). Train/val rows look like
`{"id", "label": "holding potato next to vicks vaporub bottle", "template": "Holding [something]
next to [something]", "placeholders": [...]}`.

- **Provenance:** the official source is Qualcomm (developer login + license acceptance). These
  copies came from the HF mirrors `morpheushoc/something-something-v2` and
  `olarian/something-something-v2`, which were byte-identical on all four files (2026-09-24).
- **Verified against the local archive:** the 220,847 label ids match the 220,847 `.webm` ids
  exactly (none missing either way). Usable labeled pool = train + val = 193,690.
- **License:** SSv2 is under Qualcomm's license. Accept it through the official page before
  any use beyond internal research.

## Selection + extraction (`select_extract_ssv2.py`, seed 42) — done 2026-09-24

- **24 physics templates:** the 15 core ones (falling ×6, rolling ×2, sliding, toppling ×2,
  spinning, projectile, collision, fluid) + 9 add-ons (throwing against, pushing so it spins,
  tipping over, rolling up and back down, surface lifted until sliding, stack collapse,
  spilling onto, pouring out of, dropping onto). The full list with groups is `TEMPLATES` in
  the script.
- **10,000 clips**, water-filled: the 5 small templates contribute everything they have
  (326 / 357 / 375 / 392 / 393, plus 414 for "slides down"), and the rest get 430–431 each.
  8,336 train + 1,664 validation.
- **Extracted** in one streaming pass over the local archive (385 s) →
  `/scratch/network/ssd/junlin/raw/ssv2/videos/<id>.webm` (836 MB), 10,000/10,000 present,
  200/200 sampled clips decode with OpenCV. Measured duration p10/p50/p90 = 2.25 / 3.58 / 5.25 s.
- Manifest: `/scratch/network/ssd/junlin/raw/ssv2/selection_10k.jsonl`
  (`id, split, group, template, label, placeholders, video`).

## Notes for v2 selection

- MVP isn't used for evaluation in the v2 round, so MVP's 7,142 clips are allowed in the pool.
  (If MVP is evaluated later, exclude them; they include all 2,000 `mvp_mini` SSv2 eval videos.)
- `label` fills in the objects but describes the action and its result, so the caption leak
  concern applies.
- Clip length: Δt ≤ 1.0 s is feasible for ~90% of clips, Δt = 2.0 s for ~1%.
