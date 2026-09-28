#!/usr/bin/env python
"""VLM judging for the suite: each benchmark's OFFICIAL judge code, only the API client swapped.

The official modules are imported from the cloned repos under SRC and used as-is:
  phyeditbench  PhyEditBench/utils.py::score_one_dimension (prompts, 768px PNG, json_schema, parsing)
  phyedit_anti  PhyEditBench/gpt_eval_anti.py::vlm_judge + utils.parse_anti_score_content (checklists.jsonl)
  picabench     PICABench/PicaEval_qwen.py (ROI crop_box_and_resize viz, JSON yes/no instruction, parser)
  risebench     RISEBench/gpt_eval.py::eval_vanilla + extract (3 judges, "Final Score" parsing)
  imgedit_basic ImgEdit/Benchmark/Basic/basic_bench.py::call_gpt + prompts.json + step1 parser
  imgedit_uge   ImgEdit/Benchmark/UGE/UGE_bench.py::call_gpt + parser
  magicbrush    same UGE (type-agnostic ImgEdit) rubric
Swapped: the OpenAI client -> our vLLM OpenAI-compatible endpoint(s), model name -> served
judge, temperature forced to 0 on the first attempt.

Parse failures are never dropped: the call is retried (up to MAX_RETRY more times, temperature
0.7, seed=attempt) and every attempt is logged; a row that still fails is kept with parse_ok=False.

One JSONL row per judged unit -> results/judge_raw/<judge_name>/<bench>/<model>.jsonl
  python judge.py --bench picabench --models base,T0,T2,T3 --endpoints http://torrnode12:8031/v1,...
(env: internvlu)
"""
import argparse
import base64
import contextlib
import io
import itertools
import json
import os
import re
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.abspath(os.path.join(HERE, ".."))
SRC = "/scratch/network/ssd/junlin/ssl_eval/src"
STAGE = "/scratch/network/ssd/junlin/ssl_eval/staged"
OUT = os.environ.get("SUITE_OUT", os.path.join(EVAL, "outputs"))
RES = os.path.join(EVAL, "results")
MAX_RETRY = 3
RETRY_TEMP = 0.7


# ---------------------------------------------------------------- client swap
class Endpoint:
    """Round-robin OpenAI-compatible client pool; forces temperature (0 unless retrying)."""

    def __init__(self, urls, served):
        from openai import OpenAI
        self.clients = [OpenAI(base_url=u, api_key="EMPTY", timeout=600) for u in urls]
        self.cyc = itertools.cycle(range(len(self.clients)))
        self.lock = threading.Lock()
        self.served = served
        self.local = threading.local()  # per-thread retry state

    def create(self, **kw):
        with self.lock:
            c = self.clients[next(self.cyc)]
        kw["model"] = self.served
        kw.pop("stream", None)
        att = getattr(self.local, "attempt", 0)
        kw["temperature"] = 0.0 if att == 0 else RETRY_TEMP
        if att:
            kw["seed"] = att
        kw.setdefault("max_tokens", 4096)
        for k in range(8):  # transport errors (not parse errors): back off and retry
            try:
                return c.chat.completions.create(**kw)
            except Exception as e:
                if k == 7:
                    raise
                print(f"[judge] transport error {e!r}; retry", flush=True)
                time.sleep(5 * (k + 1))

    def openai_like(self):
        """Object that quacks like `openai.OpenAI()` for code calling client.chat.completions.create."""
        ep = self
        return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=ep.create)))


def with_retries(ep, fn, ok):
    """Call fn() (temperature 0), then up to MAX_RETRY retries while ok(result) is False."""
    tries = []
    for att in range(MAX_RETRY + 1):
        ep.local.attempt = att
        try:
            res = fn()
            good = ok(res)
            err = None
        except Exception as e:
            res, good, err = None, False, repr(e)
        tries.append(dict(attempt=att, ok=good, err=err))
        if good:
            break
    ep.local.attempt = 0
    return res, tries


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def import_from(path, name, alias=None):
    import importlib.util
    d = os.path.dirname(path)
    sys.path.insert(0, d)
    try:
        spec = importlib.util.spec_from_file_location(alias or name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[alias or name] = mod  # dataclasses resolve their module via sys.modules
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(d)
    # PhyEditBench and RISEBench both ship a top-level `utils` module; never let one shadow the other
    sys.modules.pop("utils", None)
    return mod


# ---------------------------------------------------------------- benchmarks
def units_phyedit(rows, model, ep):
    U = import_from(f"{SRC}/PhyEditBench/utils.py", "utils", "phy_utils")

    class Responses:  # replaces utils.ResponsesClient; same request content, chat-completions transport
        def create_response(self, prompt_text, images, temperature=0.0, max_output_tokens=256,
                            schema_name="score_response", extra_instructions=None, save_raw_path=None):
            content = [{"type": "text", "text": prompt_text}]
            for _, p in images:
                content.append({"type": "image_url", "image_url": {"url": U.load_image_as_data_url(p)}})
            msgs = ([{"role": "system", "content": extra_instructions}] if extra_instructions else []) + \
                   [{"role": "user", "content": content}]
            fmt = U.score_json_schema(schema_name)
            r = ep.create(messages=msgs, max_tokens=max_output_tokens,
                          response_format={"type": "json_schema", "json_schema": {
                              "name": fmt["name"], "schema": fmt["schema"], "strict": fmt["strict"]}})
            return types.SimpleNamespace(output_text=r.choices[0].message.content or "")

    client = Responses()
    for r in rows:
        if r["bench"] != "phyeditbench":
            continue
        pred = os.path.join(OUT, model, r["out_rel"])
        for dim in U.DIMENSIONS:
            def run(r=r, dim=dim, pred=pred):
                def fn():
                    return U.score_one_dimension(client, dim, r["inputs"][0], r["judge"]["ref"], pred,
                                                r["instruction"], r["judge"]["explain"], r["judge"]["invariants"])
                res, tries = with_retries(ep, fn, lambda x: x is not None and "score" in x)
                return dict(dimension=dim, score=res["score"] if res else None,
                            reason=res["reason"] if res else None, parse_ok=res is not None, tries=tries)
            yield f"{r['uid']}|{dim}", r, run


def units_anti(rows, model, ep):
    A = import_from(f"{SRC}/PhyEditBench/gpt_eval_anti.py", "gpt_eval_anti")
    U = import_from(f"{SRC}/PhyEditBench/utils.py", "utils", "phy_utils2")
    client = ep.openai_like()
    dims = ["Instruction_Following", "Physical_Plausibility", "Consistency", "Image_Quality"]
    for r in rows:
        if r["bench"] != "phyedit_anti":
            continue
        pred = os.path.join(OUT, model, r["out_rel"])

        def run(r=r, pred=pred):
            def fn():
                with quiet():
                    txt = A.vlm_judge(client=client, editing_prompt=r["instruction"],
                                      edited_image=U.encode_image_base64(pred),
                                      input_image=U.encode_image_base64(r["inputs"][0]),
                                      check_list=r["judge"]["checklist"], judge_model="gpt-4o")
                return txt, U.parse_anti_score_content(txt)
            # vlm_judge swallows transport errors into all-0 scores ("Python Error"); treat those as failures
            res, tries = with_retries(ep, fn, lambda x: x is not None and all(
                x[1].get(d) is not None for d in dims) and "Python Error" not in x[0])
            parsed = res[1] if res else {}
            return dict(raw=res[0] if res else None, parse_ok=bool(res) and tries[-1]["ok"], tries=tries,
                        **{d: parsed.get(d) for d in dims}, summary=parsed.get("summary"))
        yield r["uid"], r, run


def units_pica(rows, model, ep):
    sys.modules.setdefault("vllm", types.SimpleNamespace(LLM=None, SamplingParams=None))
    P = import_from(f"{SRC}/PICABench/PicaEval_qwen.py", "PicaEval_qwen")
    args = types.SimpleNamespace(viz_padding=20, box_color="red", log_question_changes=False)
    viz_mode = "crop_box_and_resize"
    json_instruction = ("\n\nPlease provide a structured answer in the following JSON format:\n"
                        "{\"answer\": \"Yes\" or \"No\", \"explanation\": \"Brief explanation of your reasoning\"}"
                        "\n\nOutput ONLY valid JSON. No extra text.")
    # sanity: our copy of the instruction must equal the official one
    probe = P.create_structured_message([{"role": "user", "content": [{"type": "text", "text": "q"}]}])
    assert probe[-1]["content"][-1]["text"] == json_instruction
    for r in rows:
        if r["bench"] != "picabench":
            continue
        pred = os.path.join(OUT, model, r["out_rel"])
        viz_dir = os.path.join(RES, "pica_viz", model, f"visualization_annotated_qa_{viz_mode}")
        for qi, qa in enumerate(r["judge"]["annotated_qa_pairs"]):
            def run(r=r, qi=qi, qa=qa, pred=pred, viz_dir=viz_dir):
                q, rel = P.create_visualization_and_question(pred, qa, int(r["orig_id"]), "annotated_qa", qi,
                                                             viz_mode, viz_dir, args)
                vpath = os.path.join(os.path.dirname(viz_dir), rel) if rel else pred
                img = P.resize_image(Image.open(vpath).convert("RGB"))  # as prepare_vllm_batch
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
                content = [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": q},
                           {"type": "text", "text": json_instruction}]

                def fn():
                    resp = ep.create(messages=[{"role": "user", "content": content}], max_tokens=256, top_p=1.0,
                                     stop=["��", "\n��", "\U0001F4D0\n"])
                    txt = resp.choices[0].message.content or ""
                    ans, expl = P.parse_structured_response(txt)
                    return txt, ans, expl
                res, tries = with_retries(ep, fn, lambda x: x is not None and x[1] in ("Yes", "No"))
                txt, ans, expl = res if res else (None, "Unknown", "")
                gt = qa["answer"].lower().strip().rstrip(".")
                return dict(qa_index=qi, question=qa["question"], answer=qa["answer"], model_answer=ans,
                            model_response=txt, is_correct=gt == ans.lower().strip().rstrip("."),
                            parse_ok=ans in ("Yes", "No"), tries=tries)
            yield f"{r['uid']}|q{qi}", r, run


_rise_lock = threading.Lock()


def units_rise(rows, model, ep):
    import pandas as pd
    G = import_from(f"{SRC}/RISEBench/gpt_eval.py", "gpt_eval")
    full = pd.DataFrame(json.load(open(f"{STAGE}/risebench/datav2_total_w_subtask.json")))  # official json, as in main()
    full = full.set_index("index", drop=False)

    def gpt_generate(inputs, model="gpt-4.1-2025-04-14", temperature=0, max_tokens=4096, image_size=768, **kw):
        # official gpt_generate minus the HTTP call: same prepare_inputs(image_size=768), max_tokens 4096
        msgs = G.prepare_inputs(inputs, image_size=image_size)
        resp = ep.create(messages=msgs, max_tokens=max_tokens)
        return 0, resp.choices[0].message.content.strip(), resp
    G.gpt_generate = gpt_generate

    for r in rows:
        if r["bench"] != "risebench":
            continue
        item = full.loc[r["orig_id"]]

        def run(r=r, item=item):
            def fn():
                with quiet():
                    j = G.eval_vanilla(item, f"{STAGE}/risebench/data", os.path.join(OUT, model, "risebench"))
                return j, rise_scores(G, j, item)
            res, tries = with_retries(ep, fn, lambda x: x is not None and x[1] is not None)
            j, sc = res if res else (None, None)
            return dict(judges=j, scores=sc, parse_ok=sc is not None, tries=tries)
        yield r["uid"], r, run


def rise_scores(G, judge, item):
    """Per-sample slice of gpt_eval.main(): extract() + the len==3 / len==2 mapping, verbatim logic."""
    if judge["judge1"] is None:
        s2, s3 = G.extract(judge["judge2"]), G.extract(judge["judge3"])
        score = None if (not s2 or not s3) else [None] + s2 + s3
    elif "judge3" not in judge:
        s1, s2 = G.extract(judge["judge1"]), G.extract(judge["judge2"])
        score = None if (not s1 or not s2) else s1 + s2
    else:
        s1, s2, s3 = G.extract(judge["judge1"]), G.extract(judge["judge2"]), G.extract(judge["judge3"])
        score = None if (not s1 or not s2 or not s3) else s1 + s2 + s3
    if not score:
        return None
    if len(score) == 3:
        cons, rea, qua = score
    elif len(score) == 2:
        rea, cons, qua = 4 * min(score[1], 1) + 1, 4 * min(score[0], 1) + 1, None
    else:
        return None  # official code would mis-assign; count as a parse failure and retry
    row = dict(item)
    row.update(Reasoning=rea, ApprConsistency=cons, VisualPlausibility=qua)
    import pandas as pd
    row = pd.Series(row)
    return dict(Reasoning=rea, ApprConsistency=cons, VisualPlausibility=qua,
                score=float(G.calculate_score(row)), complete=int(G.calculate_completion(row)))


def units_imgedit(rows, model, ep, bench):
    if bench == "imgedit_basic":
        M = import_from(f"{SRC}/ImgEdit/Benchmark/Basic/basic_bench.py", "basic_bench")
        prompts = M.load_prompts(f"{SRC}/ImgEdit/Benchmark/Basic/prompts.json")
    else:
        M = import_from(f"{SRC}/ImgEdit/Benchmark/UGE/UGE_bench.py", "UGE_bench")
    S = import_from(f"{SRC}/ImgEdit/Benchmark/Basic/step1_get_avgscore.py", "step1_get_avgscore")
    M.OpenAI = lambda **kw: ep.openai_like()
    call = getattr(M.call_gpt, "__wrapped__", M.call_gpt)  # UGE wraps call_gpt in tenacity(100x); we retry ourselves
    for r in rows:
        if r["bench"] != bench:
            continue
        pred = os.path.join(OUT, model, r["out_rel"])

        def run(r=r, pred=pred):
            def fn():
                with quiet():
                    if bench == "imgedit_basic":
                        resp = call(r["inputs"][0], pred, r["instruction"], r["judge"]["edit_type"], prompts)
                    else:
                        resp = call(r["inputs"][0], pred, r["instruction"])
                txt = resp.choices[0].message.content
                avg, how = S.extract_scores_and_average(txt), "official"
                if avg is None and bench == "imgedit_uge":
                    # UGE's official prompt never states a score-line format, so the judge writes
                    # "... Score: 4." inline, which the official line parser cannot read. Fallback:
                    # last "Score: N" (N in 1-5). Used only when the official parser returns None.
                    hits = re.findall(r"\bScore\s*:\s*\**\s*([1-5])\b", txt or "")
                    avg, how = (float(hits[-1]), "fallback_score_regex") if hits else (None, None)
                return txt, avg, how
            res, tries = with_retries(ep, fn, lambda x: x is not None and x[1] is not None)
            txt, avg, how = res if res else (None, None, None)
            return dict(raw=txt, score=avg, parse_method=how, parse_ok=avg is not None, tries=tries)
        yield r["uid"], r, run


def make_units(bench, rows, model, ep):
    if bench == "phyeditbench":
        return units_phyedit(rows, model, ep)
    if bench == "phyedit_anti":
        return units_anti(rows, model, ep)
    if bench == "picabench":
        return units_pica(rows, model, ep)
    if bench == "risebench":
        return units_rise(rows, model, ep)
    if bench in ("imgedit_basic", "imgedit_uge"):
        return units_imgedit(rows, model, ep, bench)
    if bench == "magicbrush":  # ImgEdit's type-agnostic (UGE) rubric
        rows = [dict(r, bench="imgedit_uge") if r["bench"] == "magicbrush" else r for r in rows
                if r["bench"] == "magicbrush"]
        return units_imgedit(rows, model, ep, "imgedit_uge")
    raise ValueError(bench)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--models", default="base,T0,T2,T3")
    ap.add_argument("--endpoints", required=True, help="comma-separated OpenAI base URLs")
    ap.add_argument("--served", default="qwen3-vl-30b-fp8")
    ap.add_argument("--judge_name", default="qwen3vl30b_fp8")
    ap.add_argument("--uids", default="", help="optional file with uids to restrict to (sanity subset)")
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    ep = Endpoint(a.endpoints.split(","), a.served)
    rows = [json.loads(l) for l in open(os.path.join(HERE, "manifest.jsonl"))]
    if a.uids:
        keep = set(open(a.uids).read().split())
        rows = [r for r in rows if r["uid"] in keep]
    out_dir = os.path.join(RES, "judge_raw", a.judge_name, a.bench)
    os.makedirs(out_dir, exist_ok=True)
    for model in a.models.split(","):
        path = os.path.join(out_dir, f"{model}.jsonl")
        done = set()
        if os.path.exists(path):
            done = {json.loads(l)["key"] for l in open(path)}
        units = [(k, r, fn) for k, r, fn in make_units(a.bench, rows, model, ep) if k not in done]
        missing = [k for k, r, _ in units if not os.path.exists(os.path.join(OUT, model, r["out_rel"]))]
        if missing:
            print(f"[judge] {model} {a.bench}: {len(missing)} outputs missing, skipping those", flush=True)
            units = [u for u in units if u[0] not in set(missing)]
        print(f"[judge] {a.judge_name} {a.bench} {model}: {len(units)} units ({len(done)} done)", flush=True)
        lock = threading.Lock()
        t0 = time.time()
        with open(path, "a") as f, ThreadPoolExecutor(a.workers) as ex:
            futs = {ex.submit(fn): (k, r) for k, r, fn in units}
            for i, fut in enumerate(as_completed(futs)):
                k, r = futs[fut]
                try:
                    out = fut.result()
                except Exception as e:  # unexpected harness error: log it, keep the row visible
                    out = dict(parse_ok=False, harness_error=repr(e))
                rec = dict(key=k, uid=r["uid"], model=model, bench=a.bench, judge=a.judge_name, **out)
                with lock:
                    f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                    f.flush()
                if (i + 1) % 100 == 0:
                    print(f"[judge] {model} {a.bench} {i + 1}/{len(units)} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
