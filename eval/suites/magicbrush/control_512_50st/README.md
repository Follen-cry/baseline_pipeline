# MagicBrush — 512×512 control run (50 single-turn sessions)

A smaller, fixed-resolution control setting alongside the main
[`../README.md`](../README.md) protocol (which uses each turn's native
resolution and all 1053 turns).

**Scope, different from the main run in three ways:**
1. **Subset**: 50 sessions randomly sampled from the 216 single-turn
   ("independent editing", turn 1 only) sessions in the test split — not all
   1053 turns. Selection is in [`selected_sessions.json`](selected_sessions.json)
   (`random.seed(42)`, `random.sample` over the sorted single-turn img_id
   list, so it's reproducible from `edit_sessions.json` alone).
2. **Resolution**: input images are prescaled to 512×512 before inference
   (`--resize 512` on `gen_magicbrush_internvlu.py`), and both the generated
   and ground-truth images are prescaled to 512×512 before any metric
   (`--resize 512` on `eval_magicbrush.py`) — including L1/L2, which are
   pixel-space and therefore sensitive to this. The 512 resize uses PIL
   bicubic.
3. **Models**: only `InternVL-U` (base) and `InternVL-U-s4-pretrain`
   (`internvlu-s4-gen-merged`) — not all 5 S0-S4 checkpoints.

Since every session here is single-turn, `final_turn` and `all_turn` are
identical by construction (each is the only/last turn of its session).

## Reproduce

```bash
# generation (env: internvlu)
CUDA_VISIBLE_DEVICES=<g> python inference/gen_magicbrush_internvlu.py \
    --data_root /scratch/local/ssd/junlin/data/MagicBrush/test \
    --model_path <ckpt> \
    --out /scratch/local/ssd/junlin/results/MagicBrush_control512_50st/<Model> \
    --resize 512 --img_ids_file control_512_50st/selected_sessions.json --limit 0

# scoring (env: geneval-eval-env)
python evaluators/eval_magicbrush.py \
    --data_root /scratch/local/ssd/junlin/data/MagicBrush/test \
    --generated /scratch/local/ssd/junlin/results/MagicBrush_control512_50st/<Model>/images \
    --save_path /scratch/local/ssd/junlin/results/MagicBrush_control512_50st/<Model> \
    --resize 512 --img_ids_file control_512_50st/selected_sessions.json
```

Generated images stay external (not committed) at
`/scratch/local/ssd/junlin/results/MagicBrush_control512_50st/<Model>/`;
only `results/<Model>/evaluation_metrics.json` and this comparison are
tracked here.

## Results

See [`comparison.md`](comparison.md).
