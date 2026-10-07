# `sft/v2/` — v2 (temporal SSL, T0-T4) launchers

| Launcher | Setting | `LM_LOSS_WEIGHT` | Validation sets |
|---|---|---|---|
| `run_t0_gen_sft.sh` | F0 F1 F2 + GAP → F3 | 0.0 | T0 |
| `run_t1_gen_sft.sh` | F0 F1 F2 → `{"gap"}` + F3 | 0.5 | T1 |
| `run_t2_gen_sft.sh` | 3 of F0..F3 + GAP + MISSING → missing frame | 0.0 | T2 |
| `run_t3_gen_sft.sh` | shuffled A B C + GAP → `{"order"}` + F3 | 0.5 | T3 |
| `run_t4_gen_sft.sh` | 50% T4-A (= T0) / 50% T4-B → `{"order","gap","missing"}` + missing frame | 0.5 | T4, T0 |
| `run_t2_1_gen_sft.sh` | T2 restricted to gap ∈ {1.0, 2.0}, `cond_image` forced to F0 | 0.0 | T2.1 |
| `run_t5_gen_sft.sh` | T0's shape, PhysicTran38K-only, gap=1s-only, `cond_image` forced to F0 | 0.0 | T5 |
| `run_t6_gen_sft.sh` | T2's shape, PhysicTran38K-only, gap=1s-only, `cond_image` forced to F0 | 0.0 | T6 |

T2.1 reuses `main`'s pool (`RUN=main`, the default) with 33,046/428 rows instead of T2's
60,000/750. T5 and T6 share a new, dedicated pool (`RUN=phystran_g1`, 9,634/125 rows) built from
PhysicTran38K only at Δt=1.0 only -- see `docs/v2.md`, "Extra settings" for how the pool was built,
the row-count ceiling given the local source, and a caption-leakage caveat for T5.

Each wrapper sets `SETTING` / `LM_LOSS_WEIGHT` / `VAL_SETS` and calls the shared
`run_t_gen_sft.sh`, which holds v1's recipe (`../v1/run_s4_gen_sft.sh`): full-unified,
LLM-LoRA r32, 1 epoch, lr 1e-5, gen_loss_weight 0.5 (warmup 20), imgen_ratio 1.0, global
batch 16 (1 per GPU × 8 GPUs × accumulation 2) → 60,000 rows = **3,750 steps** per setting. It
then fixes the saved processor and merges the LoRA (`<OUTPUT_DIR>-merged`).

**One change from v1: generation-image size.** `GEN_RESIZE_MODE=area`, `GEN_IMAGE_SIZE=512`: the
target and `cond_image` keep their aspect ratio at a fixed area ~512² (scaled up or down, sides
rounded to ×16: 16:9 → 688×384, 1:1 → 512×512) — v1 squashed them to 512×512, so this is v1's pixel
budget without the distortion. Requires `PER_DEVICE_BATCH_SIZE=1` (the trainer refuses otherwise).
`GEN_RESIZE_MODE=square` reproduces v1; `GEN_RESIZE_MODE=keep_aspect GEN_IMAGE_SIZE=1024` = native
resolution (long side ≤ 1024). See `docs/v2.md` decision 6.

- Data: `baseline_pipeline/data/v2/meta/main/T{N}_train_meta.json` → `settings/main/T{N}_train.jsonl`
  (gitignored; rebuild with `data/v2/recipes/temporal_ssl/derive_settings.py` and
  `make_eval_mini.py`).
- Output: `/scratch/network/ssd/junlin/models/internvlu-v2-t{N}-gen` (ssd2 is nearly full).
- Validation (`internvl/train/v2_validation.py`): **6 runs per training** on the setting's 78-row
  evalmini — step 0, every `VAL_STEPS` and at the end, with `VAL_STEPS` = ceil(total steps /
  `VAL_INTERVALS`=5) computed from the meta length, epochs and batch (3,750 steps → every 750:
  0 / 750 / 1500 / 2250 / 3000 / 3750). An explicit `VAL_STEPS` overrides it. `val/<T>/...` to
  tensorboard + wandb (`WANDB_PROJECT=internvlu-v2`, run `t{N}-gen`). ~2 min per set on 8 GPUs
  (T4 validates on 2 sets: T4 and T0).
- The script refuses a `BATCH_SIZE` not divisible by `GPUS × PER_DEVICE_BATCH_SIZE` (the engine
  would silently shrink the global batch, e.g. 16 on 6 GPUs → 12).
- Paths are resolved from the script location, so it runs from this submodule, not from
  `Model_Related/InternVLU/InternVL` (which lacks the v2 validation code).

```bash
conda activate internvlu
bash run_t3_gen_sft.sh                      # full run, 8 GPUs
SMOKE=1 GPUS=2 BATCH_SIZE=4 bash run_t3_gen_sft.sh   # 4 steps, validation at 0/2/4 on 4 rows,
                                            # throwaway dir, wandb offline, no merge unless SMOKE_MERGE=1
```

Setting definitions: `baseline_pipeline/docs/v2.md`. v1 launchers: `../v1/`.
