# `data/v2/recipes/temporal_ssl/` — v2 recipe (T0-T4)

Turns per-source window pools into the five v2 settings (planned):

- `merge_pools.py` — merge sources with per-source caps and Δt stratification.
- `derive_settings.py` — seeded pool → `T{0..4}_{train,eval}.jsonl` (ordering, masking,
  shuffling, T4-A/B assignment, `cond_image` = observed frame nearest the target; same windows
  across settings).
- `validate.py` — label-balance checks (GAP, the 6 ORDER permutations, MISSING k ∈ {1,2,3})
  and path existence.
- `build_all.sh` — end-to-end driver.

Output: `../../datasets/temporal_ssl/`; meta: `../../meta/T{0..4}_{train,eval}_meta.json`.
Setting definitions and decisions: `docs/v2.md`.
