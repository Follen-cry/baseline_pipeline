# `data/recipes/v2_temporal_ssl/` — v2 recipe (T0-T4)

Turns per-source window pools into the five v2 settings (planned):

- `merge_pools.py` — merge sources with per-source caps and Δt stratification.
- `derive_settings.py` — seeded pool → `T{0..4}_{train,eval}.jsonl` (ordering, masking,
  shuffling, T4-A/B assignment; same windows across settings).
- `validate.py` — leak checks (caption, `cond_image`) and label-balance checks (GAP, the 6
  ORDER permutations, MISSING k ∈ {1,2,3}).
- `build_all.sh` — end-to-end driver.

Output: `../../datasets/v2_temporal_ssl/`; meta: `../../meta/v2_T{0..4}_{train,eval}_meta.json`.
Setting definitions and open decisions: `docs/v2.md`.
