# `data/v1/recipes/uniform_data_split_regarding_data_source/`

A second WorldPrediction-WM train/eval split, alongside the original one in
`data/v1/sources/worldprediction/` (COIN = eval, CrossTask+EPIC-KITCHENS-100+IKEAASM = train).

**Motivation**: the original split's WM-SFT results (see project memory /
`eval/suites/worldprediction/results/wm_sft_*`) weren't strong, and one known caveat was a
domain gap between train (egocentric-kitchen/furniture-assembly-heavy) and eval (COIN,
third-person how-to video). This recipe removes that specific variable by mixing all 4 usable
sources uniformly into both splits.

## Split
- **Row-level, 80/20 (train:eval = 4:1), independently within each of the 4 sources** (COIN,
  CrossTask, EPIC-KITCHENS-100, IKEAASM — EgoExo4D still excluded, same reason as before: gated,
  no local video). `SPLIT_SEED=42`.
- Both splits contain samples from every source, in the same ~80/20 proportion per source:

  | source | total | train | eval |
  |---|---|---|---|
  | COIN | 171 | 137 | 34 |
  | CrossTask | 81 | 65 | 16 |
  | EPIC-KITCHENS-100 | 193 | 154 | 39 |
  | IKEAASM | 159 | 127 | 32 |
  | **total** | **604** | **483** | **121** |

- **Row-level, not video-level** — a deliberate deviation from every other split in this repo
  (which are video/trajectory/clip-level to guarantee no train/eval video overlap). Caveat, not a
  bug: two rows built from the same underlying source video could land on opposite sides of this
  split. Not checked/enforced here since the split was explicitly requested at the row level.

## No new video processing
Purely a recombination of the 604 already-processed rows from
`data/v1/datasets/worldprediction/{wm_train,wm_eval}.jsonl` (built by
`data/v1/sources/worldprediction/processing/build_worldprediction_wm.py`) — frames, prompts, and
answer letters are unchanged; only train/eval membership is reassigned. `build_uniform_split.py`
also re-derives a filtered WorldPrediction-annotation-format file
(`WorldPrediction-WM-uniform_eval.json`) containing just the 121 eval rows, in the schema the real
eval harness (`eval/suites/worldprediction/run.py`/`evaluator.py`) expects — needed because that
harness reads the original states/candidates/ground_truth schema, not this repo's flat training
jsonl.

## Output
- `data/v1/datasets/uniform_data_split_regarding_data_source/{train,eval}.jsonl` — training data,
  same row schema as `data/v1/datasets/worldprediction/`.
- `data/v1/datasets/uniform_data_split_regarding_data_source/WorldPrediction-WM-uniform_eval.json` —
  eval-harness annotation file.
- `data/v1/meta/uniform_data_split_regarding_data_source_{train,eval}_meta.json`.

## This experiment's scope
First pass trains/evals only **base, S0, S2** InternVL-U checkpoints (not the full S0-S4 sweep),
using the same training recipe as the prior "worldprediction" experiment
(`training/models/internvl-u/internvl_chat/shell/internvlu/sft/v1/run_worldprediction_wm_sft.sh`),
just pointed at this recipe's meta jsons instead.
