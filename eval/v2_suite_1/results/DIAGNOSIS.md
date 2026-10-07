# Diagnosis: why do T0 / T2 / T3 score below base?

Three candidate causes, each tested with a small experiment. The benchmark judge and the T-series training setup were not changed: the degenerate baselines and the no-edit penalty are re-scorings, and the edit fine-tune is a separate probe (`models/diag_editsft/`). Every number below is read from `results/diagnosis/*.csv`; code for each probe is in `eval/v2_suite_1/diagnosis/`.

## Recommendation (ranked)

1. **Cause 3 · training-induced (supervision signal): work on first.** The trained models mostly fail by not editing, and the training data asks them to reproduce a near-identical conditioning frame for about half of all targets. The control check shows the same failure appears within 160 steps whenever fine-tuning targets sit close to the VAE condition, so the fix is in the supervision (which frame conditions, which targets are kept), not necessarily in using video.
2. **Cause 1 · judge: fix alongside, cheaply.** Real flaw on PhyEditBench, but it does not explain the result: penalising no-edit outputs keeps base on top and widens the gap. It must be fixed so progress on Cause 3 is not scored as a regression.
3. **Cause 2 · format mismatch: not supported.** An identical edit fine-tune leaves T0/T2/T3 where they were; the gap closes only because the same fine-tune pulls base down into the T models' failure pattern. The probe's editing data had mostly tiny edits, so this is a weak test of format; a probe with large edits is the natural follow-up.

Across the three probes, the evidence points at one mechanism: the v2 training teaches the model to copy its pixel condition. The trained models' dominant failure is no edit; the training targets are close to a copy of the frame fed as the VAE condition for most rows; and neither a stricter judge nor an editing-format fine-tune changes the picture.

## Control check · is the probe comparison valid?

**Verdict:** The control is valid, and it changes the reading. Base's drop is real, not an artifact, and it is the same copy-like failure the T models show. That makes the Cause 2 probe a weak test of format (its own data induces the failure), and it supports Cause 3's mechanism as a property of the supervision (targets close to the VAE condition) rather than of video content as such.

The control holds:

- A null fine-tune of base (the identical train, save, processor-replace and LoRA-merge path, with learning rate 0) is weight-identical to the original: relative change 0.000 in the VLM and 0.000 in the generation decoder. The 36 decoder output-layer tensors saved as bf16 instead of fp32 are inert: the pipeline loads everything in bf16.
- End to end, the null model reproduces original base exactly: 150 of 150 subset images are pixel-identical (mean difference 0.000). The rewritten configs (scheduler keys, VLM tiling fields) change nothing, so base+FT is a valid same-recipe control.

What the fine-tune did to base:

- What the 160 real steps did to base is the T models' failure pattern: PICABench 0.611 → 0.540, RISEBench 0.429 → 0.365, ImgEdit 0.938 → 0.839, while consistency rises (PhyEditBench 7.44 → 8.04, RISE 3.83 → 4.11) and reasoning falls (RISE 2.73 → 2.40). No-edit outputs go from 6% to 15% and the median edit size from 0.172 to 0.117. PhyEditBench, whose judge rewards copying, is unchanged (0.702 → 0.704).
- The change sits in the generation decoder (relative change 0.0016, largest in the timestep embedder; decoder lr 5e-5) rather than the VLM (0.00012). Likely reasons, in order: half the probe data (MagicBrush) has tiny target edits, so the loss rewards reproducing the input; for an edit row the VAE pixel condition is the input image itself, so on low-change targets the decoder learns to lean on it; and the decoder is updated at 5x the LLM learning rate on a narrow set.

## Cause 1 · Judge unreliability

**Verdict:** The judge is part of the problem for PhyEditBench, but fixing it leaves the conclusion unchanged (trained < base) and makes the gap larger.

Evidence for:

- PhyEditBench's judge rewards doing nothing: the unedited copy scores 0.757 (normalised), above base (0.690) and every trained model (T2 0.735), and an unrelated random image still averages 0.456. On some items an unrelated image gets a near-perfect score (examples in the artifact). Worst on the long-horizon types: Type E copy 0.743 vs base 0.624.
- PICABench's QA cannot tell copy (0.281) from an unrelated image (0.268), so its floor is about 0.27, not 0.

Evidence against:

- On the other benchmarks the judge behaves: random is near the floor (RISEBench 0.012, ImgEdit 0.133), copy sits far below base (RISEBench 0.213 vs 0.438), and the ground truth scores near the ceiling where it exists (PhyEditBench 0.854, RISEBench 0.952).
- Flooring every output that is nearly identical to its input (mean |out - in| < τ) does not change the conclusion; it strengthens it. Target-group ranking without the penalty: -0.026 (T0), -0.010 (T2), -0.025 (T3); with τ = 0.03: -0.129, -0.091, -0.116. The order base > T2 > T3 > T0 holds at τ = 0.02, 0.03 and 0.05.

## Cause 2 · Format mismatch

**Verdict:** Not supported as the main cause. Giving the trained models the editing format does not recover what they lost.

Evidence for:

- After the identical edit fine-tune the measured gap to base shrinks: T0 -0.055 → +0.001, T2 -0.038 → +0.039, T3 -0.050 → -0.004 (150-item subset, normalised).
- T2 is the one model that moves up: +0.020 (CI -0.017 to +0.059).

Evidence against:

- The gap closes because base gets worse, not because the trained models get better. The fine-tune changes base by -0.056 (CI -0.098 to -0.016), T0 by -0.000 and T3 by -0.010. Base's edit magnitude falls from 0.175 to 0.131, towards the trained models' 0.116.
- T0 and T3 stay significantly below the original base even after adapting to the editing format (-0.056 and -0.060, both CIs below 0).

Caveats:

- Small probe: 1,280 pairs, 160 steps, one seed, 150 items. It can rule out a large, quick recovery, not a benefit from a much larger editing mix.
- The control check shows the probe itself turns base into the T models' failure pattern, so the shrinking gap cannot be read as format evidence; the negative verdict rests on the T models not improving. Half the probe data is MagicBrush, whose ground-truth edits are tiny (median change 0.023; 60% below the no-edit threshold) versus PICA-100K (0.169). The PICA-100K half is also narrow: the one downloaded shard (the first of 111, ordered by physics law) contains only the Global law (weather, time of day, season changes), so the probe never shows the model a mechanics, optics or local state edit. Part of base's drop is the probe data rewarding small edits, and the recipe itself (nearest-frame VAE condition for edits = the input) may do the same.

## Cause 3 · Training-induced error

**Verdict:** Supported. The training's supervision signal rewards reproducing the conditioning frame, and that is the failure the trained models show.

Evidence for:

- The losses are mostly non-edits. On the loss set the VLM tags 21% (T0), 22% (T2) and 28% (T3) of trained outputs as no edit, against 3% for base; on all 1,210 items the automatic no-edit rate is 19% / 16% / 18% vs 4%.
- The supervision asks for exactly that. The VAE pixel condition is always the shown frame nearest the target (derive_settings.py:23-24), and the static filter keeps windows where only 0.1% of pixels change (window_motion.py:29). As a result 51% of T0/T3 training targets differ from their condition in under 5% of pixels and 24% in under 1%.
- The data mix is weighted towards the most static sources (merge_pools.py:65): PhysInOne (74% of targets under 5% change), VBVR (78%) and PhysicTran38K (58%) make up 58.8% of windows; SSv2 has the least (12%).

Evidence against:

- No sign of a temporal logic bug: wrong-temporal-direction outputs are rare for every model (T0 1%, T3 1%, base 1%), frame extraction is frame-accurate, and the CFG handling matches inference.
- Losses do not grow from T0 to T2 to T3: T2 is the best of the three (63% correct on the loss set vs T0 58% and T3 49%). So the harm is shared by all settings rather than caused by one task formulation.
- Base drifts more, not less: automatic colour drift 42% vs 22% (T0), blur 43% vs 20%. The training did not degrade image quality.

## Training-code review

| severity | area | location | what | why it matters |
|---|---|---|---|---|
| high | supervision signal | `data/v2/recipes/temporal_ssl/derive_settings.py:23-24, :87, :99-101, :116-121` | cond_image (the VAE pixel condition) is always the shown frame nearest in time to the target (F2 for T0/T1/T3). | The target is close to a pixel-level copy of the condition for most rows (see cause3_target_change.csv), so the cheapest way to lower the flow-matching loss is to reproduce the condition. This matches the no-edit / partial-edit failure mode and the lower edit magnitude. |
| high | frame extraction / filtering | `data/v2/common/window_motion.py:29 (STATIC_THR = 0.001), :12-17` | A window is dropped only if under 0.1% of pixels change; windows that stall part-way are kept. | Admits many near-static targets: 24% of T0/T3 targets change <1% of pixels vs the condition and 51% change <5%; 11.1% of windows are flagged stalled. |
| medium | data proportions | `data/v2/recipes/temporal_ssl/merge_pools.py:65 (WEIGHTS)` | PhysicTran38K 20 + PhysInOne 15 + VBVR 15 of 85 = 58.8% of training windows. | These three sources have the most static targets (58% / 74% / 78% of targets change <5% vs the condition) and two of them are synthetic renders; SSv2 (10/85) has the least (12%). |
| medium | format | `data/v2/common/prompts.py:25-52, :66-78` | Every training row is three frames + a clip caption + 'Generate the next frame ...'; no single-image, instruction-style rows. | The edit prompt format (one image + an imperative instruction) is never seen during the 3,750 steps; the caption describes the whole clip rather than a change to make. |
| low | timestamps | `data/v2/common/window_sampler.py:68` | Frame index = round(t * nominal fps). | Δt labels are approximate for variable-frame-rate sources (SSv2 webm); affects the GAP signal, not editing directly. |
| cleared | CFG conditioning | `internvl/train/dataset_unified.py:290-300, :393-404` | The VAE condition is kept in all CFG-dropout states (text / all). | Deliberate: matches inference, where the pipeline builds the pixel condition once outside the CFG-branch loop. Not a bug. |
| cleared | frame extraction | `data/v2/common/extract_frames.py:73-84` | Frames are read with one sequential cap.grab()/retrieve() pass. | Frame-accurate (no keyframe seeking); decode failures are caught and the window is skipped. |
| cleared | loss | `shell/internvlu/sft/v2/run_t_gen_sft.sh:19, :53, :56-57` | gen_loss_weight 0.5 (warmup 20); lm_loss_weight 0 for T0/T2, 0.5 for T1/T3/T4; INTERNVLU_VAE_COND=1. | Same recipe as the v1 S-series; nothing anomalous on its own. The issue is what the VAE condition is (row 1), not the loss weights. |

## Probe details

- **Cause 1**: `diagnosis/make_degenerate_baselines.py` (random = the input of a task from another benchmark, fixed seed; gt = ground-truth target where available), judged with the unchanged `judge.py`; `diagnosis/cause1_rescore.py` for the penalty (τ ∈ {0.02, 0.03, 0.05}; flagged outputs floored on every judged metric).
- **Cause 2**: `diagnosis/edit_sft/` (data build with perceptual-hash overlap check against every eval image, 27 candidates dropped; launcher = v2 T-series recipe, 2 epochs, 160 steps); fixed subset `cause2_subset_uids.txt` (150 items: PICABench 40, RISE 48, PhyEditBench 25, ImgEdit 27, Anti-Physics 10; MagicBrush excluded because the probe trains on MagicBrush-train); `diagnosis/cause2_analyze.py`.
- **Cause 3**: `diagnosis/failure_modes.py` (loss set = 40 worst items per trained model, union 67; VLM tags with a diagnostic-only prompt, JSON schema, temperature 0; automatic no-edit / colour-drift / blur checks), `diagnosis/pool_target_change.py`, and the code review above.

The visual examples are in the artifact's Diagnosis section.
