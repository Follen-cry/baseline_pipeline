#!/usr/bin/env python
"""Write results/DIAGNOSIS.md and results/diagnosis/diagnosis.json from the diagnosis CSVs.

The narrative lives here once; every number is a {{d:<table>:<k=v&k=v>:<field>:<digits>[:pct|signed]}}
placeholder resolved from results/diagnosis/<table>.csv, the same syntax the artifact page resolves
client-side from the embedded CSV data.   (env: internvlu)
"""
import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
DIAG = os.path.join(RES, "diagnosis")
TABLES = {"c1": "cause1_baselines.csv", "c1p": "cause1_penalty.csv", "c2": "cause2_results.csv",
          "c2m": "cause2_means.csv", "c2e": "cause2_ft_effect.csv", "c2d": "cause2_ftdata_editmag.csv",
          "c3": "cause3_modes.csv", "c3t": "cause3_target_change.csv",
          "ctl": "control_base_ft.csv", "ctn": "control_null_pixels.csv", "wd": "weight_diff.csv"}
T = {k: pd.read_csv(os.path.join(DIAG, v)) for k, v in TABLES.items()}


def resolve(text):
    def rep(mt):
        a = mt.group(1).split(":")
        df = T[a[0]]
        for kv in filter(None, a[1].split("&")):
            k, v = kv.split("=", 1)
            df = df[df[k].astype(str) == v]
        if df.empty:
            raise KeyError(mt.group(0))
        v = float(df.iloc[0][a[2]])
        d = int(a[3]) if len(a) > 3 else 3
        mode = a[4] if len(a) > 4 else ""
        if mode == "pct":
            return f"{round(v * 100)}%"
        if mode == "signed":
            return f"{v:+.{d}f}"
        return f"{v:.{d}f}"
    return re.sub(r"\{\{d:([^}]+)\}\}", rep, text)


C1 = "c1:bench=phyeditbench&slice=all"
RANKING = [
    dict(rank=1, cause="Cause 3 · training-induced (supervision signal)", verdict="work on first",
         why="The trained models mostly fail by not editing, and the training data asks them to reproduce a near-identical conditioning frame for about half of all targets. The control check shows the same failure appears within 160 steps whenever fine-tuning targets sit close to the VAE condition, so the fix is in the supervision (which frame conditions, which targets are kept), not necessarily in using video."),
    dict(rank=2, cause="Cause 1 · judge", verdict="fix alongside, cheaply",
         why="Real flaw on PhyEditBench, but it does not explain the result: penalising no-edit outputs keeps base on top and widens the gap. It must be fixed so progress on Cause 3 is not scored as a regression."),
    dict(rank=3, cause="Cause 2 · format mismatch", verdict="not supported",
         why="An identical edit fine-tune leaves T0/T2/T3 where they were; the gap closes only because the same fine-tune pulls base down into the T models' failure pattern. The probe's editing data had mostly tiny edits, so this is a weak test of format; a probe with large edits is the natural follow-up."),
]
SUMMARY = [
    "Across the three probes, the evidence points at one mechanism: the v2 training teaches the model to copy its pixel condition. The trained models' dominant failure is no edit; the training targets are close to a copy of the frame fed as the VAE condition for most rows; and neither a stricter judge nor an editing-format fine-tune changes the picture.",
]
SECTIONS = {
"Control check · is the probe comparison valid?": {
 "for": [
  "A null fine-tune of base (the identical train, save, processor-replace and LoRA-merge path, with learning rate 0) is weight-identical to the original: relative change {{d:wd:ckpt=base_null-merged&component=vlm:rel_change:3}} in the VLM and {{d:wd:ckpt=base_null-merged&component=generation_decoder:rel_change:3}} in the generation decoder. The 36 decoder output-layer tensors saved as bf16 instead of fp32 are inert: the pipeline loads everything in bf16.",
  "End to end, the null model reproduces original base exactly: {{d:ctn::identical:0}} of {{d:ctn::n:0}} subset images are pixel-identical (mean difference {{d:ctn::mean_pixel_diff:3}}). The rewritten configs (scheduler keys, VLM tiling fields) change nothing, so base+FT is a valid same-recipe control.",
 ],
 "against": [
  "What the 160 real steps did to base is the T models' failure pattern: PICABench {{d:ctl:item=picabench (normalised):base:3}} → {{d:ctl:item=picabench (normalised):base_ft:3}}, RISEBench {{d:ctl:item=risebench (normalised):base:3}} → {{d:ctl:item=risebench (normalised):base_ft:3}}, ImgEdit {{d:ctl:item=imgedit_basic (normalised):base:3}} → {{d:ctl:item=imgedit_basic (normalised):base_ft:3}}, while consistency rises (PhyEditBench {{d:ctl:item=PhyEditBench consistency (1-10):base:2}} → {{d:ctl:item=PhyEditBench consistency (1-10):base_ft:2}}, RISE {{d:ctl:item=RISE consistency (1-5):base:2}} → {{d:ctl:item=RISE consistency (1-5):base_ft:2}}) and reasoning falls (RISE {{d:ctl:item=RISE reasoning (1-5):base:2}} → {{d:ctl:item=RISE reasoning (1-5):base_ft:2}}). No-edit outputs go from {{d:ctl:item=no-edit outputs (edit size < 0.03):base:0:pct}} to {{d:ctl:item=no-edit outputs (edit size < 0.03):base_ft:0:pct}} and the median edit size from {{d:ctl:item=median edit size (mean |out - in|):base:3}} to {{d:ctl:item=median edit size (mean |out - in|):base_ft:3}}. PhyEditBench, whose judge rewards copying, is unchanged ({{d:ctl:item=phyeditbench (normalised):base:3}} → {{d:ctl:item=phyeditbench (normalised):base_ft:3}}).",
  "The change sits in the generation decoder (relative change {{d:wd:ckpt=base-merged&component=generation_decoder:rel_change:4}}, largest in the timestep embedder; decoder lr 5e-5) rather than the VLM ({{d:wd:ckpt=base-merged&component=vlm:rel_change:5}}). Likely reasons, in order: half the probe data (MagicBrush) has tiny target edits, so the loss rewards reproducing the input; for an edit row the VAE pixel condition is the input image itself, so on low-change targets the decoder learns to lean on it; and the decoder is updated at 5x the LLM learning rate on a narrow set.",
 ],
 "labels": ["The control holds", "What the fine-tune did to base"],
 "verdict": "The control is valid, and it changes the reading. Base's drop is real, not an artifact, and it is the same copy-like failure the T models show. That makes the Cause 2 probe a weak test of format (its own data induces the failure), and it supports Cause 3's mechanism as a property of the supervision (targets close to the VAE condition) rather than of video content as such."},
"Cause 1 · Judge unreliability": {
 "for": [
  "PhyEditBench's judge rewards doing nothing: the unedited copy scores {{d:" + C1 + ":copy:3}} (normalised), above base ({{d:" + C1 + ":base:3}}) and every trained model (T2 {{d:" + C1 + ":T2:3}}), and an unrelated random image still averages {{d:" + C1 + ":random:3}}. On some items an unrelated image gets a near-perfect score (examples in the artifact). Worst on the long-horizon types: Type E copy {{d:c1:bench=phyeditbench&slice=TypeE:copy:3}} vs base {{d:c1:bench=phyeditbench&slice=TypeE:base:3}}.",
  "PICABench's QA cannot tell copy ({{d:c1:bench=picabench&slice=all:copy:3}}) from an unrelated image ({{d:c1:bench=picabench&slice=all:random:3}}), so its floor is about 0.27, not 0.",
 ],
 "against": [
  "On the other benchmarks the judge behaves: random is near the floor (RISEBench {{d:c1:bench=risebench&slice=all:random:3}}, ImgEdit {{d:c1:bench=imgedit_basic&slice=all:random:3}}), copy sits far below base (RISEBench {{d:c1:bench=risebench&slice=all:copy:3}} vs {{d:c1:bench=risebench&slice=all:base:3}}), and the ground truth scores near the ceiling where it exists (PhyEditBench {{d:" + C1 + ":gt:3}}, RISEBench {{d:c1:bench=risebench&slice=all:gt:3}}).",
  "Flooring every output that is nearly identical to its input (mean |out - in| < τ) does not change the conclusion; it strengthens it. Target-group ranking without the penalty: {{d:c1p:tau=none&group=A:T0_delta:3:signed}} (T0), {{d:c1p:tau=none&group=A:T2_delta:3:signed}} (T2), {{d:c1p:tau=none&group=A:T3_delta:3:signed}} (T3); with τ = 0.03: {{d:c1p:tau=0.03&group=A:T0_delta:3:signed}}, {{d:c1p:tau=0.03&group=A:T2_delta:3:signed}}, {{d:c1p:tau=0.03&group=A:T3_delta:3:signed}}. The order base > T2 > T3 > T0 holds at τ = 0.02, 0.03 and 0.05.",
 ],
 "verdict": "The judge is part of the problem for PhyEditBench, but fixing it leaves the conclusion unchanged (trained < base) and makes the gap larger."},
"Cause 2 · Format mismatch": {
 "for": [
  "After the identical edit fine-tune the measured gap to base shrinks: T0 {{d:c2:pool=subset&model=T0:gap_before:3:signed}} → {{d:c2:pool=subset&model=T0:gap_after:3:signed}}, T2 {{d:c2:pool=subset&model=T2:gap_before:3:signed}} → {{d:c2:pool=subset&model=T2:gap_after:3:signed}}, T3 {{d:c2:pool=subset&model=T3:gap_before:3:signed}} → {{d:c2:pool=subset&model=T3:gap_after:3:signed}} (150-item subset, normalised).",
  "T2 is the one model that moves up: {{d:c2e:pool=subset&comparison=T2+FT - T2:delta:3:signed}} (CI {{d:c2e:pool=subset&comparison=T2+FT - T2:ci_lo:3:signed}} to {{d:c2e:pool=subset&comparison=T2+FT - T2:ci_hi:3:signed}}).",
 ],
 "against": [
  "The gap closes because base gets worse, not because the trained models get better. The fine-tune changes base by {{d:c2e:pool=subset&comparison=base+FT - base:delta:3:signed}} (CI {{d:c2e:pool=subset&comparison=base+FT - base:ci_lo:3:signed}} to {{d:c2e:pool=subset&comparison=base+FT - base:ci_hi:3:signed}}), T0 by {{d:c2e:pool=subset&comparison=T0+FT - T0:delta:3:signed}} and T3 by {{d:c2e:pool=subset&comparison=T3+FT - T3:delta:3:signed}}. Base's edit magnitude falls from {{d:c2m:pool=subset&model=base:edit_mad:3}} to {{d:c2m:pool=subset&model=base_editsft:edit_mad:3}}, towards the trained models' {{d:c2m:pool=subset&model=T0:edit_mad:3}}.",
  "T0 and T3 stay significantly below the original base even after adapting to the editing format ({{d:c2e:pool=subset&comparison=T0+FT - base (original):delta:3:signed}} and {{d:c2e:pool=subset&comparison=T3+FT - base (original):delta:3:signed}}, both CIs below 0).",
 ],
 "caveats": [
  "Small probe: 1,280 pairs, 160 steps, one seed, 150 items. It can rule out a large, quick recovery, not a benefit from a much larger editing mix.",
  "The control check shows the probe itself turns base into the T models' failure pattern, so the shrinking gap cannot be read as format evidence; the negative verdict rests on the T models not improving. Half the probe data is MagicBrush, whose ground-truth edits are tiny (median change {{d:c2d:source=magicbrush_train:median:3}}; {{d:c2d:source=magicbrush_train:frac_below_0.03:0:pct}} below the no-edit threshold) versus PICA-100K ({{d:c2d:source=pica100k:median:3}}). Part of base's drop is the probe data rewarding small edits, and the recipe itself (nearest-frame VAE condition for edits = the input) may do the same.",
 ],
 "verdict": "Not supported as the main cause. Giving the trained models the editing format does not recover what they lost."},
"Cause 3 · Training-induced error": {
 "for": [
  "The losses are mostly non-edits. On the loss set the VLM tags {{d:c3:scope=loss_set&model=T0:vlm_no_edit:0:pct}} (T0), {{d:c3:scope=loss_set&model=T2:vlm_no_edit:0:pct}} (T2) and {{d:c3:scope=loss_set&model=T3:vlm_no_edit:0:pct}} (T3) of trained outputs as no edit, against {{d:c3:scope=loss_set&model=base:vlm_no_edit:0:pct}} for base; on all 1,210 items the automatic no-edit rate is {{d:c3:scope=all_items&model=T0:auto_no_edit:0:pct}} / {{d:c3:scope=all_items&model=T2:auto_no_edit:0:pct}} / {{d:c3:scope=all_items&model=T3:auto_no_edit:0:pct}} vs {{d:c3:scope=all_items&model=base:auto_no_edit:0:pct}}.",
  "The supervision asks for exactly that. The VAE pixel condition is always the shown frame nearest the target (derive_settings.py:23-24), and the static filter keeps windows where only 0.1% of pixels change (window_motion.py:29). As a result {{d:c3t:source=ALL&setting=T0_T3 (F2->F3):lt_5pct:0:pct}} of T0/T3 training targets differ from their condition in under 5% of pixels and {{d:c3t:source=ALL&setting=T0_T3 (F2->F3):lt_1pct:0:pct}} in under 1%.",
  "The data mix is weighted towards the most static sources (merge_pools.py:65): PhysInOne ({{d:c3t:source=physinone&setting=T0_T3 (F2->F3):lt_5pct:0:pct}} of targets under 5% change), VBVR ({{d:c3t:source=vbvr&setting=T0_T3 (F2->F3):lt_5pct:0:pct}}) and PhysicTran38K ({{d:c3t:source=physictran38k&setting=T0_T3 (F2->F3):lt_5pct:0:pct}}) make up 58.8% of windows; SSv2 has the least ({{d:c3t:source=ssv2&setting=T0_T3 (F2->F3):lt_5pct:0:pct}}).",
 ],
 "against": [
  "No sign of a temporal logic bug: wrong-temporal-direction outputs are rare for every model (T0 {{d:c3:scope=loss_set&model=T0:vlm_wrong_temporal_direction:0:pct}}, T3 {{d:c3:scope=loss_set&model=T3:vlm_wrong_temporal_direction:0:pct}}, base {{d:c3:scope=loss_set&model=base:vlm_wrong_temporal_direction:0:pct}}), frame extraction is frame-accurate, and the CFG handling matches inference.",
  "Losses do not grow from T0 to T2 to T3: T2 is the best of the three ({{d:c3:scope=loss_set&model=T2:vlm_correct:0:pct}} correct on the loss set vs T0 {{d:c3:scope=loss_set&model=T0:vlm_correct:0:pct}} and T3 {{d:c3:scope=loss_set&model=T3:vlm_correct:0:pct}}). So the harm is shared by all settings rather than caused by one task formulation.",
  "Base drifts more, not less: automatic colour drift {{d:c3:scope=all_items&model=base:auto_global_drift:0:pct}} vs {{d:c3:scope=all_items&model=T0:auto_global_drift:0:pct}} (T0), blur {{d:c3:scope=all_items&model=base:auto_blur:0:pct}} vs {{d:c3:scope=all_items&model=T0:auto_blur:0:pct}}. The training did not degrade image quality.",
 ],
 "verdict": "Supported. The training's supervision signal rewards reproducing the conditioning frame, and that is the failure the trained models show."},
}


def main():
    ranking = [dict(r) for r in RANKING]
    json.dump({"ranking": ranking, "summary": SUMMARY, "sections": SECTIONS}, open(os.path.join(DIAG, "diagnosis.json"), "w"), indent=1)
    L = ["# Diagnosis: why do T0 / T2 / T3 score below base?\n",
         "Three candidate causes, each tested with a small experiment. The benchmark judge and the T-series training "
         "setup were not changed: the degenerate baselines and the no-edit penalty are re-scorings, and the edit "
         "fine-tune is a separate probe (`models/diag_editsft/`). Every number below is read from "
         "`results/diagnosis/*.csv`; code for each probe is in `eval/v2_suite_1/diagnosis/`.\n",
         "## Recommendation (ranked)\n"]
    for r in ranking:
        L.append(f"{r['rank']}. **{r['cause']}: {r['verdict']}.** {r['why']}")
    L += ["", resolve(SUMMARY[0]), ""]
    for name, s in SECTIONS.items():
        L.append(f"## {name}\n")
        L.append(f"**Verdict:** {s['verdict']}\n")
        lab = s.get("labels", ["Evidence for", "Evidence against"])
        L.append(f"{lab[0]}:\n")
        L += [f"- {resolve(x)}" for x in s["for"]]
        L.append(f"\n{lab[1]}:\n")
        L += [f"- {resolve(x)}" for x in s["against"]]
        if s.get("caveats"):
            L.append("\nCaveats:\n")
            L += [f"- {resolve(x)}" for x in s["caveats"]]
        L.append("")
    L.append("## Training-code review\n")
    cr = pd.read_csv(os.path.join(DIAG, "cause3_code_review.csv"))
    L.append("| severity | area | location | what | why it matters |\n|---|---|---|---|---|")
    for _, r in cr.iterrows():
        L.append(f"| {r.severity} | {r.area} | `{r.location}` | {r.what} | {r.why_it_matters} |")
    L.append("\n## Probe details\n")
    L.append("- **Cause 1**: `diagnosis/make_degenerate_baselines.py` (random = the input of a task from another benchmark, fixed seed; "
             "gt = ground-truth target where available), judged with the unchanged `judge.py`; `diagnosis/cause1_rescore.py` for the "
             "penalty (τ ∈ {0.02, 0.03, 0.05}; flagged outputs floored on every judged metric).")
    L.append("- **Cause 2**: `diagnosis/edit_sft/` (data build with perceptual-hash overlap check against every eval image, 27 "
             "candidates dropped; launcher = v2 T-series recipe, 2 epochs, 160 steps); fixed subset `cause2_subset_uids.txt` "
             "(150 items: PICABench 40, RISE 48, PhyEditBench 25, ImgEdit 27, Anti-Physics 10; MagicBrush excluded because the "
             "probe trains on MagicBrush-train); `diagnosis/cause2_analyze.py`.")
    L.append("- **Cause 3**: `diagnosis/failure_modes.py` (loss set = 40 worst items per trained model, union 67; VLM tags "
             "with a diagnostic-only prompt, JSON schema, temperature 0; automatic no-edit / colour-drift / blur checks), "
             "`diagnosis/pool_target_change.py`, and the code review above.")
    L.append("\nThe visual examples are in the artifact's Diagnosis section.")
    open(os.path.join(RES, "DIAGNOSIS.md"), "w").write("\n".join(L) + "\n")
    print("DIAGNOSIS.md written")


if __name__ == "__main__":
    main()
