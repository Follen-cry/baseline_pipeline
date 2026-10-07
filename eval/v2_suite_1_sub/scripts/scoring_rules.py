#!/usr/bin/env python
"""Extract the exact judge prompts and scoring code into results/scoring_rules.json (for the page; CPU only, env: internvlu).

Nothing here is hand-copied. Prompts are rendered by the same official modules judge.py imports
(config.judge.benchmark_src), with {placeholders} in place of the per-item fields. Weights and parsers are
taken with inspect.getsource. One worked example per benchmark: the first gallery item of that benchmark
(same RNG as build_artifact.py), its fully filled prompt(s), and the judge's recorded reply for
config.report_models[0] from results/judge_raw.
"""
import inspect
import json
import os
import random
import sys
import types

import pandas as pd

from common import CFG, JUDGE_RAW, RES, manifest
from judge import SRC, STAGE, import_from

EX_MODEL = CFG["report_models"][0]


def raw_rows(bench, uid):
    p = os.path.join(JUDGE_RAW, bench, f"{EX_MODEL}.jsonl")
    rows = {}
    for l in open(p):
        r = json.loads(l)
        if r["uid"] == uid:
            rows[r["key"]] = r
    return list(rows.values())


def example_uid(rows, bench):
    g = CFG["gallery"]
    uids = sorted(r["uid"] for r in rows if r["bench"] == bench)
    pick = set(random.Random(f"{g['seed']}:gallery:{bench}").sample(uids, min(g["per_bench"][bench], len(uids))))
    return next(r for r in rows if r["uid"] in pick)


def phyedit(rows):
    U = import_from(f"{SRC}/PhyEditBench/utils.py", "utils", "phy_utils")
    inv = ["{invariant 1}", "{invariant 2}", "..."]
    images = {"consistency": ["input", "prediction"], "instruction_following": ["input", "GT target", "prediction"],
              "physical_plausibility": ["input", "GT target", "prediction"], "image_quality": ["prediction", "GT target"]}
    prompts = [dict(name=d.replace("_", " ").title(), images=images[d],
                    text=U.build_dimension_prompt(d, "{instruction}", "{explain}", inv)) for d in U.DIMENSIONS]
    ex = example_uid(rows, "phyeditbench")
    got = {r["dimension"]: r for r in raw_rows("phyeditbench", ex["uid"])}
    j = ex["judge"]
    example = dict(uid=ex["uid"], instruction=ex["instruction"], calls=[
        dict(name=d.replace("_", " ").title(), images=images[d],
             prompt=U.build_dimension_prompt(d, ex["instruction"], j["explain"], j["invariants"]),
             reply=json.dumps({"score": got[d]["score"], "reason": got[d]["reason"]}, ensure_ascii=False),
             parsed=f"score {got[d]['score']}") for d in U.DIMENSIONS])
    w = U.DIMENSION_WEIGHTS
    overall = sum(w[d] * got[d]["score"] for d in U.DIMENSIONS)
    example["result"] = " + ".join(f"{w[d]}×{got[d]['score']}" for d in U.DIMENSIONS) + f" = {overall:.2f} / 10"
    return dict(
        key="phyeditbench", name="PhyEditBench", metric="Overall, 1–10",
        calls_per_item=f"{len(U.DIMENSIONS)} independent judge calls, one per dimension",
        system=("You are a strict judge. Output must follow the JSON schema exactly."),
        images="Each image downscaled to ≤768 px on the long side (PNG), sent after the text in the order listed per prompt. "
               "GT target = the benchmark's ground-truth next state.",
        output="JSON constrained by a schema: {\"score\": integer 1–10, \"reason\": string}; max 256 tokens.",
        parse="score = int(score), clipped to 1–10. A reply that is not valid JSON counts as a parse failure and is retried.",
        item_rule="overall = weighted sum of the four dimension scores (official DIMENSION_WEIGHTS below). "
                  "An item needs all four dimensions.",
        bench_rule="Official aggregate: mean of each dimension over the items, then the same weighted sum.",
        code=[("PhyEditBench/utils.py · DIMENSION_WEIGHTS", "DIMENSION_WEIGHTS = " + json.dumps(w, indent=4))],
        notes=["Instruction Following and Physical Plausibility compare against the GT image, so an output that barely changes "
               "the input can still score well when the GT state is close to the input. This is why copy scores "
               "high here (v2_suite_1 DIAGNOSIS, Cause 1).",
               "Image Quality never sees the input; Consistency never sees the GT."],
        prompts=prompts, example=example)


def pica(rows):
    sys.modules.setdefault("vllm", types.SimpleNamespace(LLM=None, SamplingParams=None))
    P = import_from(f"{SRC}/PICABench/PicaEval_qwen.py", "PicaEval_qwen")
    probe = P.create_structured_message([{"role": "user", "content": [{"type": "text", "text": "q"}]}])
    json_instr = probe[-1]["content"][-1]["text"]
    prompts = [dict(name="Per checklist question", images=["output crop around the question's box"],
                    text="{question}" + json_instr)]
    ex = example_uid(rows, "picabench")
    got = sorted(raw_rows("picabench", ex["uid"]), key=lambda r: r["qa_index"])
    calls = [dict(name=f"Question {r['qa_index'] + 1}", images=["output crop"],
                  prompt=r["question"] + json_instr, reply=r.get("model_response") or "",
                  parsed=f"answer {r['model_answer']} · expected {r['answer']} · {'correct' if r['is_correct'] else 'wrong'}")
             for r in got]
    n_ok = sum(bool(r["is_correct"]) for r in got)
    nq = [len(r["judge"]["annotated_qa_pairs"]) for r in rows if r["bench"] == "picabench"]
    return dict(
        key="picabench", name="PICABench", metric="Accuracy, %",
        calls_per_item=f"one judge call per checklist question ({min(nq)}–{max(nq)} questions per item, "
                       f"{sum(nq) / len(nq):.1f} on average in this subset)",
        system="none",
        images="Only the model output, never the input or the instruction. The output is resized to 1024 px on the long side, "
               "cropped to the question's annotated box with 20 px padding, the crop is resized again to 1024 px long side, "
               "then sent as PNG (official crop_box_and_resize mode).",
        output="Text asked to be JSON {\"answer\": \"Yes\"|\"No\", \"explanation\": ...}; max 256 tokens.",
        parse="Official parse_structured_response: JSON first, then a yes/no fallback. The answer is correct when it equals the "
              "annotated answer (case-insensitive). An answer that is neither Yes nor No is retried, and if it still fails "
              "it counts as wrong (official).",
        item_rule="acc = correct answers / questions for that item.",
        bench_rule="Accuracy = mean of the per-item acc (official per-sample average), shown in %.",
        code=[("PicaEval_qwen.py · parse_structured_response", inspect.getsource(P.parse_structured_response)),
              ("PicaEval_qwen.py · _parse_json_response", inspect.getsource(P._parse_json_response))],
        notes=["The checklist questions describe the physically correct result of the edit (e.g. a shadow, a reflection, a "
               "tilted surface). The judge answers each question from the cropped output alone.",
               "Prompts are the superficial PICABench instructions (no physics hint), as in v2_suite_1."],
        prompts=prompts,
        example=dict(uid=ex["uid"], instruction=ex["instruction"], calls=calls,
                     result=f"{n_ok} of {len(got)} correct → acc {n_ok / len(got):.3f}"))


def rise(rows):
    G = import_from(f"{SRC}/RISEBench/gpt_eval.py", "gpt_eval")
    full = {r["index"]: r for r in json.load(open(f"{STAGE}/risebench/datav2_total_w_subtask.json"))}
    items = [full[r["orig_id"]] for r in rows if r["bench"] == "risebench"]
    n_img = sum(not pd.isna(x.get("reasoning_img")) for x in items)
    n_free = sum("consistency_free" in x and not pd.isna(x["consistency_free"]) for x in items)
    prompts = [dict(name="Consistency (judge 1)", images=["input", "output"], text=G.prompt_consist),
               dict(name="Reasoning (judge 2)", images=["output"], text=G.prompt_reasoning),
               dict(name=f"Reasoning, with input image (judge 2; {n_img} of {len(items)} items)", images=["input", "output"],
                    text=G.prompt_reasoning_w_input),
               dict(name="Visual plausibility (judge 3)", images=["output"], text=G.prompt_generation)]
    ex = example_uid(rows, "risebench")
    it = full[ex["orig_id"]]
    got = raw_rows("risebench", ex["uid"])[0]
    w_in = not pd.isna(it.get("reasoning_img"))
    fill = lambda t: t.format(instruct=it["instruction"], reference=it.get("reference", "")) if "{" in t else t
    calls = []
    for key, name, tmpl, imgs in (("judge1", "Consistency (judge 1)", G.prompt_consist, ["input", "output"]),
                                  ("judge2", "Reasoning (judge 2)", G.prompt_reasoning_w_input if w_in else G.prompt_reasoning,
                                   ["input", "output"] if w_in else ["output"]),
                                  ("judge3", "Visual plausibility (judge 3)", G.prompt_generation, ["output"])):
        reply = (got.get("judges") or {}).get(key)
        if reply is None:
            continue
        calls.append(dict(name=name, images=imgs, prompt=fill(tmpl), reply=reply, parsed=f"Final Score → {G.extract(reply)}"))
    s = got["scores"]
    return dict(
        key="risebench", name="RISEBench", metric="Score 1–5 and Accuracy %",
        calls_per_item="3 judge calls: consistency, reasoning, visual plausibility (consistency is skipped for "
                       f"'consistency_free' items: {n_free} of {len(items)} here)",
        system="none",
        images="JPEG, resized to 768 px (official prepare_inputs, image_size=768), sent after the text.",
        output="Free text ending in “**Final Score:** **X**”; max 4096 tokens.",
        parse="Official extract(): the first integer after “Final Score(s)”. Each call gives one 1–5 score. "
              "If any call does not parse, all three are retried together.",
        item_rule="score = 0.3·Consistency + 0.5·Reasoning + 0.2·Plausibility (0.8·Reasoning + 0.2·Plausibility when "
                  "consistency-free); if Reasoning = 1 the score is halved, with a floor of 1. solved (Accuracy) = 1 only when "
                  "every judged dimension is 5.",
        bench_rule="Score = mean per-item score (1–5). Accuracy = share of items solved, in %.",
        code=[("gpt_eval.py · calculate_score", inspect.getsource(G.calculate_score)),
              ("gpt_eval.py · calculate_completion", inspect.getsource(G.calculate_completion)),
              ("gpt_eval.py · extract", inspect.getsource(G.extract))],
        notes=["Reasoning is judged against a text reference, not a reference image: the judge reads the ground-truth "
               "description and checks the output against it.",
               f"Accuracy needs a perfect 5 on every dimension, so it is coarse at n = {len(items)} "
               f"(one item = {100 / len(items):.1f} points)."],
        prompts=prompts,
        example=dict(uid=ex["uid"], instruction=ex["instruction"], reference=it.get("reference"), calls=calls,
                     result=f"Consistency {s['ApprConsistency']}, Reasoning {s['Reasoning']}, Plausibility "
                            f"{s['VisualPlausibility']} → score {s['score']:.2f}, solved {s['complete']}"))


def imgedit(rows):
    M = import_from(f"{SRC}/ImgEdit/Benchmark/Basic/basic_bench.py", "basic_bench")
    S = import_from(f"{SRC}/ImgEdit/Benchmark/Basic/step1_get_avgscore.py", "step1_get_avgscore")
    prompts_all = M.load_prompts(f"{SRC}/ImgEdit/Benchmark/Basic/prompts.json")
    used = sorted({r["judge"]["edit_type"] for r in rows if r["bench"] == "imgedit_basic"})
    counts = {t: sum(r["bench"] == "imgedit_basic" and r["judge"]["edit_type"] == t for r in rows) for t in used}
    prompts = [dict(name=f"{t} ({counts[t]} items)", images=["input", "output"], text=prompts_all[t].strip()) for t in used]
    ex = example_uid(rows, "imgedit_basic")
    got = raw_rows("imgedit_basic", ex["uid"])[0]
    et = ex["judge"]["edit_type"]
    calls = [dict(name=f"Rubric: {et}", images=["input", "output"],
                  prompt=prompts_all[et].replace("<edit_prompt>", ex["instruction"]).strip(),
                  reply=got["raw"], parsed=f"mean of sub-scores = {got['score']}")]
    return dict(
        key="imgedit_basic", name="ImgEdit", metric="Score, 1–5",
        calls_per_item="1 judge call with the rubric for the item's edit type (9 rubrics; 8 occur in this subset)",
        system="none",
        images="Input then output, full resolution, base64 (official call_gpt).",
        output="Free text: a brief reasoning line, then three “<Criterion>: N” lines, N in 1–5.",
        parse="Official extract_scores_and_average: every line of the form “text: integer” is a sub-score; the item score is "
              "their mean, rounded to 2 decimals. No such line = parse failure, retried.",
        item_rule="score = mean of the three rubric sub-scores (each 1–5).",
        bench_rule="Mean item score (1–5).",
        code=[("step1_get_avgscore.py · extract_scores_and_average", inspect.getsource(S.extract_scores_and_average))],
        notes=["Each rubric tells the judge the second and third scores must not exceed the first (edit fidelity), so a "
               "missed edit drags all three down. The instruction is inserted at <edit_prompt>.",
               "ImgEdit Basic is a general-editing control in v2_suite_1 (group C), not a physics benchmark."],
        prompts=prompts,
        example=dict(uid=ex["uid"], instruction=ex["instruction"], calls=calls,
                     result=f"score {got['score']} / 5"))


def main():
    rows = manifest()
    out = dict(
        judge=dict(model="Qwen3-VL-30B-A3B-Instruct-FP8", served=CFG["judge"]["served_name"],
                   server="vLLM, OpenAI-compatible chat completions, --seed 0",
                   temperature=0.0, max_retry=CFG["judge"]["max_retry"], retry_temperature=CFG["judge"]["retry_temperature"],
                   example_model=EX_MODEL),
        benches=[rise(rows), phyedit(rows), pica(rows), imgedit(rows)])
    p = os.path.join(RES, "scoring_rules.json")
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {p}: " + ", ".join(f"{b['name']} {len(b['prompts'])} prompts / {len(b['example']['calls'])} example calls"
                                     for b in out["benches"]))


if __name__ == "__main__":
    main()
