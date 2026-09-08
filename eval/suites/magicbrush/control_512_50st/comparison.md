# MagicBrush — 512×512 control run: base vs. S4

50 randomly sampled single-turn sessions (seed 42), inputs and GT prescaled
to 512×512 before inference/metrics. See [`README.md`](README.md) for the
full protocol. `final_turn` == `all_turn` here (single-turn only).

| model | l1 | l2 | clip-i | dino | clip-t |
|---|---|---|---|---|---|
| InternVL-U (base) | 0.1352 | 0.0481 | 0.8475 | 0.7150 | 0.3101 |
| InternVL-U-s4-pretrain | 0.0745 | 0.0243 | 0.9110 | 0.8588 | 0.3075 |

S4 improves L1 by ~45%, L2 by ~50%, CLIP-I by +6.3pt, DINO by +14.4pt over
base; CLIP-T is essentially flat (-0.3pt), consistent with the main
(native-resolution, full-1053-turn) results.
