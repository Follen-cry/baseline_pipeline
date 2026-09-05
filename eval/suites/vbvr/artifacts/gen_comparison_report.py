#!/usr/bin/env python3
"""Generate a 4-model comparison artifact (InternVL-U base, InternVL-U
finetuned, bagel, sensenova) in the same visual language as the earlier
single-model Dual-Gauge Eval Report, but restructured around shared test
instances: since all 4 runs score the identical 900 (id, prompt, GT) pairs,
each sampled instance is shown once with all 4 models' outputs side by
side, rather than 4 separate galleries."""
import base64
import html as htmlmod
import json
import os
import random
import sys
from io import BytesIO

from PIL import Image

sys.path.insert(0, "/scratch/network/ssd2/junlin/ssl_mllm/Evaluation/VBVR-CustomEval/evaluators")
from llm_judge_full import TASK_CRITERIA, build_full_judge_instruction

RUN = "/scratch/network/ssd2/junlin/ssl_mllm/results/vbvr_target_pred_eval"
OUT_PATH = "/scratch/network/ssd2/junlin/ssl_mllm/Evaluation/VBVR-CustomEval/artifacts/model_comparison_report.html"

SEED = 82720260827
N_SHARED_PER_TASK = 20

MODELS = [
    ("base", "InternVL-U (base)", f"{RUN}/base_prescale512_vaecond",
     "/scratch/network/ssd2/junlin/ssl_mllm/Evaluation/VBVR-CustomEval/judge_eval/judge_scored_900.json"),
    ("finetuned", "InternVL-U (fine-tuned)", f"{RUN}/vbvr-gen-noce_prescale512_vaecond",
     f"{RUN}/vbvr-gen-noce_prescale512_vaecond/judge_scored.json"),
    ("bagel", "BAGEL", f"{RUN}/bagel", f"{RUN}/bagel/judge_scored.json"),
    ("sensenova", "SenseNova-U1", f"{RUN}/sensenova", f"{RUN}/sensenova/judge_scored.json"),
]

CATEGORY_OF = {
    "ball_bounces_given_time": "Knowledge", "stable_sort": "Perception",
    "multi_object_placement": "Perception", "grid_shift": "Transformation",
    "rotation_puzzle": "Transformation", "shape_color_then_move": "Abstraction",
    "animal_size_sorting": "Perception", "maze": "Spatiality",
    "2d_geometric_transformation": "Transformation",
}
DOMAIN_OF = {
    "ball_bounces_given_time": "ID", "stable_sort": "ID", "multi_object_placement": "ID",
    "grid_shift": "ID", "rotation_puzzle": "ID", "shape_color_then_move": "OOD",
    "animal_size_sorting": "OOD", "maze": "OOD", "2d_geometric_transformation": "OOD",
}
TASK_ORDER = list(TASK_CRITERIA.keys())

RULE_MECHANISM = {
    "ball_bounces_given_time": "Single criterion: distance between the ball's detected final position and the ground truth's, via Hough-circle / dark-blob detection.",
    "stable_sort": "Contour-based shape detection (type/size/color/position) recovers each shape; classification, order, fidelity, and layout are checked against the sorted target arrangement.",
    "multi_object_placement": "Per-color contour detection for objects and star markers; scores how many objects reached their matching-color star, how centered, and whether counts/sizes/star positions survived.",
    "grid_shift": "Colored-block detection gates the sub-scores on a completeness/pattern-preservation check; only if blocks are still recognizably present does it score movement direction, step count, and final-cell precision.",
    "rotation_puzzle": "Pipe-tile color-mask detection compares tile connectivity, rotation angle, grid position, and opening alignment between the first and final frame.",
    "shape_color_then_move": "Per-shape detection (type/color/position) on both frames checks the untouched reference row stayed pixel-identical, then whether the pattern was correctly completed.",
    "animal_size_sorting": "Contour-based animal detection; checks final left-to-right size order, shared baseline, appearance fidelity, and that no animal went missing.",
    "maze": "Orange/yellow path-marker mask vs. black wall mask checks the drawn path never crosses a wall, reaches both markers, moves only between adjacent cells, and the maze itself is untouched. Gated by a path-coverage check (drawn path pixels vs. the GT's own path pixel count) so short stubs that never reach the end can't score as if they were a full solve.",
    "2d_geometric_transformation": "HSV saturation isolates the colored shape from its grayscale scene; rotation angle (via ellipse fit), position, and size are compared to the dashed target outline.",
}

MODEL_COLOR_VAR = {"base": "--m-base", "finetuned": "--m-finetuned", "bagel": "--m-bagel", "sensenova": "--m-sensenova"}


def esc(s):
    return htmlmod.escape(s or "", quote=True)


def img_data_uri(path, max_side=190, quality=78):
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return None
    w, h = im.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def spearman(xs, ys):
    n = len(xs)
    def rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    vx = sum((rx[i] - mx) ** 2 for i in range(n))
    vy = sum((ry[i] - my) ** 2 for i in range(n))
    return cov / (vx ** 0.5 * vy ** 0.5) if vx and vy else None


def main():
    model_data = {}
    for key, label, run_dir, judge_path in MODELS:
        scored = json.load(open(os.path.join(run_dir, "scored.json")))
        meta = json.load(open(os.path.join(run_dir, "meta.json")))
        judge = json.load(open(judge_path))["results"]
        rule_by_id = {r["id"]: r for r in scored["records"]}
        meta_by_id = {r["id"]: r for r in meta["results"]}
        model_data[key] = {"label": label, "rule": rule_by_id, "judge": judge, "meta": meta_by_id}

    base_meta = model_data["base"]["meta"]
    by_task_ids = {}
    for rid, row in base_meta.items():
        by_task_ids.setdefault(row["task_name"], []).append(rid)

    rng = random.Random(SEED)
    shared_picks = {t: rng.sample(ids, min(N_SHARED_PER_TASK, len(ids))) for t, ids in by_task_ids.items()}

    # ---- stats: per model x task, and overall ----
    task_stats = {}   # task -> model_key -> {rule_mean, judge_mean, rho, n}
    overall_stats = {}  # model_key -> {rule_mean, judge_mean, rho, n}
    for key, _, _, _ in MODELS:
        d = model_data[key]
        all_r, all_j = [], []
        for t in TASK_ORDER:
            rs, js = [], []
            for rid in by_task_ids.get(t, []):
                r = d["rule"].get(rid)
                j = d["judge"].get(rid)
                if r and r["score"] is not None and j and j.get("judge_score") is not None:
                    rs.append(r["score"]); js.append(j["judge_score"])
            rho = spearman(rs, js) if len(rs) > 1 else None
            task_stats.setdefault(t, {})[key] = {
                "rule_mean": sum(rs) / len(rs) if rs else 0,
                "judge_mean": sum(js) / len(js) if js else 0,
                "rho": rho, "n": len(rs),
            }
            all_r += rs; all_j += js
        rho_o = spearman(all_r, all_j) if len(all_r) > 1 else None
        overall_stats[key] = {
            "rule_mean": sum(all_r) / len(all_r) if all_r else 0,
            "judge_mean": sum(all_j) / len(all_j) if all_j else 0,
            "rho": rho_o, "n": len(all_r),
        }

    def readout_cards():
        cards = []
        for key, label, _, _ in MODELS:
            s = overall_stats[key]
            rho_str = f"{s['rho']:.3f}" if s["rho"] is not None else "n/a"
            cards.append(f"""
            <div class="model-card model-{key}">
              <div class="model-card-head"><span class="model-dot dot-{key}"></span>{esc(label)}</div>
              <div class="model-card-stats">
                <div><span class="mcs-label rule-ink">rule</span><span class="mono">{s['rule_mean']:.3f}</span></div>
                <div><span class="mcs-label judge-ink">judge</span><span class="mono">{s['judge_mean']:.3f}</span></div>
                <div><span class="mcs-label">agree &rho;</span><span class="mono">{rho_str}</span></div>
              </div>
            </div>""")
        return "".join(cards)

    def summary_table():
        rows = []
        for t in TASK_ORDER:
            cells = "".join(
                f'<td class="num mono">{task_stats[t][key]["rule_mean"]:.2f}<span class="sep">/</span>{task_stats[t][key]["judge_mean"]:.2f}</td>'
                for key, _, _, _ in MODELS
            )
            rows.append(f"""
            <tr>
              <td><span class="pill pill-cat cat-{esc(CATEGORY_OF[t].lower())}">{esc(CATEGORY_OF[t])}</span></td>
              <td class="mono">{esc(t)}</td>
              <td><span class="pill pill-dom">{esc(DOMAIN_OF[t])}</span></td>
              {cells}
            </tr>""")
        headers = "".join(f'<th class="num">{esc(label)}<br><span class="th-sub">rule / judge</span></th>' for _, label, _, _ in MODELS)
        return f"""
        <div class="table-wrap">
          <table class="summary-table">
            <thead><tr><th>Category</th><th>Task</th><th>Domain</th>{headers}</tr></thead>
            <tbody>{"".join(rows)}</tbody>
          </table>
        </div>"""

    def model_mini_bars(t):
        rows = []
        for key, label, _, _ in MODELS:
            s = task_stats[t][key]
            rows.append(f"""
            <div class="mini-bar-row">
              <span class="mini-bar-label"><span class="model-dot dot-{key}"></span>{esc(label)}</span>
              <div class="mini-bar-track"><div class="mini-bar-fill fill-rule" style="width:{s['rule_mean']*100:.1f}%"></div></div>
              <span class="mono mini-bar-val rule-ink">{s['rule_mean']:.2f}</span>
              <div class="mini-bar-track"><div class="mini-bar-fill fill-judge" style="width:{s['judge_mean']*100:.1f}%"></div></div>
              <span class="mono mini-bar-val judge-ink">{s['judge_mean']:.2f}</span>
            </div>""")
        return "".join(rows)

    def sample_row(t, idx, rid):
        base_row = base_meta[rid]
        input_uri = img_data_uri(base_row["input_image"])
        gt_uri = img_data_uri(base_row["target_image"])
        prompt = base_row["prompt"].split("Task: ", 1)[-1] if "Task: " in base_row["prompt"] else base_row["prompt"]

        model_cells = []
        for key, label, _, _ in MODELS:
            d = model_data[key]
            row = d["meta"].get(rid)
            r = d["rule"].get(rid, {})
            j = d["judge"].get(rid, {})
            gen_uri = img_data_uri(row["generated_image"]) if row else None
            rscore = r.get("score") or 0.0
            jscore = j.get("judge_score") or 0.0
            model_cells.append(f"""
            <div class="mcell">
              <div class="mcell-label"><span class="model-dot dot-{key}"></span>{esc(label)}</div>
              <img src="{gen_uri}" alt="{esc(label)} output" loading="lazy">
              <div class="mcell-scores">
                <span class="rule-ink mono">{rscore:.2f}</span><span class="sep">/</span><span class="judge-ink mono">{jscore:.2f}</span>
              </div>
            </div>""")

        return f"""
        <div class="sample-row">
          <div class="sample-row-head">
            <span class="mono sample-idx">#{idx:02d}</span>
            <span class="sample-id mono">{esc(rid.split('_eval_')[-1])}</span>
          </div>
          <p class="sample-prompt">{esc(prompt)}</p>
          <div class="sample-grid">
            <div class="mcell ref-cell">
              <div class="mcell-label">input</div>
              <img src="{input_uri}" alt="input" loading="lazy">
            </div>
            <div class="mcell ref-cell">
              <div class="mcell-label">ground truth</div>
              <img src="{gt_uri}" alt="ground truth" loading="lazy">
            </div>
            {"".join(model_cells)}
          </div>
        </div>"""

    def task_section(t):
        task_desc, criteria = TASK_CRITERIA[t]
        cat = CATEGORY_OF[t]
        dom = DOMAIN_OF[t]
        rep_prompt = base_meta[shared_picks[t][0]]["prompt"]
        instruction = build_full_judge_instruction(t, rep_prompt)
        crit_bars = "".join(
            f'<div class="wcrit"><div class="wcrit-head"><code>{esc(k)}</code><span class="mono">{w:g}%</span></div>'
            f'<div class="wcrit-defs"><span class="lvl lvl-correct">correct</span>{esc(lv["correct"])}</div>'
            f'<div class="wcrit-defs"><span class="lvl lvl-partial">partial</span>{esc(lv["partial"])}</div>'
            f'<div class="wcrit-defs"><span class="lvl lvl-wrong">wrong</span>{esc(lv["wrong"])}</div></div>'
            for k, w, lv in criteria
        )
        rows = "".join(sample_row(t, i + 1, rid) for i, rid in enumerate(shared_picks[t]))

        return f"""
        <section class="task" id="t-{esc(t)}">
          <header class="task-head">
            <div class="task-head-top">
              <span class="pill pill-cat cat-{esc(cat.lower())}">{esc(cat)}</span>
              <span class="pill pill-dom">{esc(dom)}</span>
              <h2>{esc(t)}</h2>
            </div>
          </header>
          <p class="task-desc">{esc(task_desc)}</p>
          <div class="task-body-cols">
            <div class="method-col">
              <p class="method-label rule-ink">Rule-based method</p>
              <p class="method-text">{esc(RULE_MECHANISM.get(t, ""))}</p>
              {crit_bars}
            </div>
            <div class="method-col">
              <p class="method-label judge-ink">Judge method &mdash; exact prompt sent</p>
              <pre class="prompt-box mono">{esc(instruction)}</pre>
            </div>
          </div>
          <p class="minibar-label">Mean score per model (rule / judge)</p>
          <div class="mini-bars">{model_mini_bars(t)}</div>
          <p class="samples-label">{N_SHARED_PER_TASK} shared test instances, same id across all 4 models (seed {SEED})</p>
          <div class="samples-list">{rows}</div>
        </section>"""

    nav_links = "".join(f'<a href="#t-{esc(t)}">{esc(t)}</a>' for t in TASK_ORDER)
    sections = "".join(task_section(t) for t in TASK_ORDER)
    legend = "".join(
        f'<span class="legend-item"><span class="model-dot dot-{key}"></span>{esc(label)}</span>'
        for key, label, _, _ in MODELS
    )

    html = f"""<title>Four-Model Eval Comparison</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700;800&family=Source+Sans+3:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {{
  --bg: #EEEBE3; --surface: #FBFAF6; --surface-alt: #E4E0D2;
  --ink: #201F1B; --ink-soft: #55524A; --ink-faint: #8A8577; --border: #D6D1C0;
  --rule: #0E7C7B; --rule-soft: #D9EFEE; --judge: #B3236B; --judge-soft: #F7DEEB; --overlap: #4A3B6B;
  --good: #1C6B5E; --warn: #A9700E; --bad: #A8395A;
  --cat-perception: #A8395A; --cat-knowledge: #2A5CA6; --cat-transformation: #B06A1A;
  --cat-abstraction: #6B4FA0; --cat-spatiality: #1C6B5E;
  --m-base: #6B7A8F; --m-finetuned: #C9922B; --m-bagel: #4C8C5B; --m-sensenova: #B0542E;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #16181A; --surface: #1D2020; --surface-alt: #262A29;
    --ink: #ECEAE2; --ink-soft: #B8B4A6; --ink-faint: #837F72; --border: #34383A;
    --rule: #4FD6D2; --rule-soft: #16302F; --judge: #F17BB0; --judge-soft: #3A1E2C; --overlap: #B0A0F0;
    --good: #5FBFA3; --warn: #D9A548; --bad: #E37FA0;
    --cat-perception: #E37FA0; --cat-knowledge: #7DA8E8; --cat-transformation: #E0A356;
    --cat-abstraction: #A98FE0; --cat-spatiality: #5FBFA3;
    --m-base: #9DACC0; --m-finetuned: #E8B95C; --m-bagel: #7FC492; --m-sensenova: #E88A5C;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #16181A; --surface: #1D2020; --surface-alt: #262A29;
  --ink: #ECEAE2; --ink-soft: #B8B4A6; --ink-faint: #837F72; --border: #34383A;
  --rule: #4FD6D2; --rule-soft: #16302F; --judge: #F17BB0; --judge-soft: #3A1E2C; --overlap: #B0A0F0;
  --good: #5FBFA3; --warn: #D9A548; --bad: #E37FA0;
  --cat-perception: #E37FA0; --cat-knowledge: #7DA8E8; --cat-transformation: #E0A356;
  --cat-abstraction: #A98FE0; --cat-spatiality: #5FBFA3;
  --m-base: #9DACC0; --m-finetuned: #E8B95C; --m-bagel: #7FC492; --m-sensenova: #E88A5C;
}}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{ background: var(--bg); color: var(--ink); margin: 0; padding: 0 0 6rem; font-family: "Source Sans 3", system-ui, sans-serif; font-size: 16px; line-height: 1.55; }}
.mono {{ font-family: "JetBrains Mono", ui-monospace, monospace; font-variant-numeric: tabular-nums; }}
.wrap {{ max-width: 1280px; margin: 0 auto; padding: 0 2rem; }}
a {{ color: var(--rule); }}
.rule-ink {{ color: var(--rule); }}
.judge-ink {{ color: var(--judge); }}
.sep {{ color: var(--ink-faint); margin: 0 0.15em; }}

header.top {{ background: var(--surface); border-bottom: 1px solid var(--border); padding: 3.2rem 0 2rem; }}
.kicker {{ font-family: "JetBrains Mono", monospace; font-size: 0.76rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--ink-faint); margin: 0 0 0.8rem; }}
h1 {{ font-family: "Barlow Condensed", system-ui, sans-serif; font-weight: 700; font-size: clamp(2.1rem, 4.6vw, 3.2rem); line-height: 1.02; margin: 0 0 1rem; text-wrap: balance; letter-spacing: -0.01em; }}
.lede {{ color: var(--ink-soft); max-width: 72ch; font-size: 1.03rem; line-height: 1.6; margin: 0 0 1.8rem; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 1.1rem; margin-bottom: 1.6rem; }}
.legend-item {{ display: flex; align-items: center; gap: 0.4rem; font-size: 0.85rem; color: var(--ink-soft); }}
.model-dot {{ width: 0.7rem; height: 0.7rem; border-radius: 50%; display: inline-block; flex-shrink: 0; }}
.dot-base {{ background: var(--m-base); }}
.dot-finetuned {{ background: var(--m-finetuned); }}
.dot-bagel {{ background: var(--m-bagel); }}
.dot-sensenova {{ background: var(--m-sensenova); }}
.model-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 0.9rem; margin-bottom: 1.6rem; }}
.model-card {{ background: var(--surface-alt); border: 1px solid var(--border); border-radius: 10px; padding: 1rem 1.2rem; border-top: 3px solid var(--border); }}
.model-base {{ border-top-color: var(--m-base); }}
.model-finetuned {{ border-top-color: var(--m-finetuned); }}
.model-bagel {{ border-top-color: var(--m-bagel); }}
.model-sensenova {{ border-top-color: var(--m-sensenova); }}
.model-card-head {{ font-family: "Barlow Condensed", system-ui, sans-serif; font-weight: 600; font-size: 1.15rem; display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.6rem; }}
.model-card-stats {{ display: flex; flex-direction: column; gap: 0.25rem; font-size: 0.88rem; }}
.model-card-stats div {{ display: flex; justify-content: space-between; }}
.mcs-label {{ color: var(--ink-faint); text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.04em; }}
nav.tasknav {{ display: flex; flex-wrap: wrap; gap: 0.35rem 1rem; margin-top: 0.4rem; }}
nav.tasknav a {{ font-family: "JetBrains Mono", monospace; font-size: 0.76rem; color: var(--ink-soft); text-decoration: none; border-bottom: 1px dotted var(--border); }}
nav.tasknav a:hover {{ color: var(--rule); border-color: var(--rule); }}

section.intro {{ padding: 2.6rem 0; border-bottom: 1px solid var(--border); }}
.intro-cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2.4rem; }}
.intro-col h3 {{ font-family: "Barlow Condensed", system-ui, sans-serif; font-size: 1.5rem; margin: 0 0 0.7rem; display: flex; align-items: center; gap: 0.5rem; }}
.ink-dot {{ width: 0.8rem; height: 0.8rem; border-radius: 50%; display: inline-block; }}
.ink-dot-rule {{ background: var(--rule); }}
.ink-dot-judge {{ background: var(--judge); }}
.intro-col p {{ color: var(--ink-soft); margin: 0 0 0.8rem; }}
.intro-col ul {{ margin: 0 0 0.8rem; padding-left: 1.2rem; color: var(--ink-soft); }}
.intro-col li {{ margin-bottom: 0.4rem; }}
.lesson-box {{ background: var(--surface-alt); border-left: 3px solid var(--overlap); border-radius: 6px; padding: 0.9rem 1.1rem; margin-top: 1rem; font-size: 0.9rem; color: var(--ink-soft); }}
.lesson-box b {{ color: var(--ink); }}

section.summary {{ padding: 2.6rem 0; border-bottom: 1px solid var(--border); }}
section.summary h2, section.task .task-head h2 {{ font-family: "Barlow Condensed", system-ui, sans-serif; font-weight: 700; }}
section.summary h2 {{ font-size: 1.9rem; margin: 0 0 1.2rem; }}
.table-wrap {{ overflow-x: auto; border: 1px solid var(--border); border-radius: 10px; background: var(--surface); }}
table.summary-table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
table.summary-table th {{ text-align: left; padding: 0.7rem 0.9rem; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--ink-faint); border-bottom: 1px solid var(--border); white-space: nowrap; }}
.th-sub {{ font-size: 0.62rem; text-transform: none; letter-spacing: 0; opacity: 0.75; }}
table.summary-table td {{ padding: 0.55rem 0.9rem; border-bottom: 1px solid var(--border); }}
table.summary-table tr:last-child td {{ border-bottom: none; }}
table.summary-table .num {{ text-align: right; }}

.pill {{ display: inline-block; padding: 0.16rem 0.6rem; border-radius: 999px; font-size: 0.7rem; font-weight: 600; color: #fff; white-space: nowrap; }}
.pill-dom {{ background: var(--ink-faint); }}
.cat-perception {{ background: var(--cat-perception); }}
.cat-knowledge {{ background: var(--cat-knowledge); }}
.cat-transformation {{ background: var(--cat-transformation); }}
.cat-abstraction {{ background: var(--cat-abstraction); }}
.cat-spatiality {{ background: var(--cat-spatiality); }}

section.task {{ padding: 3rem 0; border-bottom: 1px solid var(--border); }}
.task-head-top {{ display: flex; align-items: baseline; gap: 0.7rem; flex-wrap: wrap; margin-bottom: 1rem; }}
.task-head-top h2 {{ font-size: 1.9rem; margin: 0; }}
.task-desc {{ color: var(--ink-soft); max-width: 82ch; margin: 0 0 1.6rem; }}
.task-body-cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-bottom: 1.8rem; align-items: start; }}
.method-label {{ font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; margin: 0 0 0.6rem; }}
.method-text {{ font-size: 0.88rem; color: var(--ink-soft); margin: 0 0 0.9rem; }}
.wcrit {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 0.6rem 0.8rem; margin-bottom: 0.5rem; }}
.wcrit-head {{ display: flex; justify-content: space-between; font-size: 0.83rem; margin-bottom: 0.3rem; }}
.wcrit-defs {{ font-size: 0.74rem; color: var(--ink-faint); margin-bottom: 0.12rem; }}
.prompt-box {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 0.85rem 1rem; font-size: 0.68rem; line-height: 1.5; white-space: pre-wrap; word-break: break-word; max-height: 26rem; overflow-y: auto; color: var(--ink-soft); }}
.lvl {{ display: inline-block; font-family: "JetBrains Mono", monospace; font-size: 0.6rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.02em; padding: 0.06rem 0.35rem; border-radius: 4px; color: #fff; margin-right: 0.35rem; }}
.lvl-correct {{ background: var(--good); }}
.lvl-partial {{ background: var(--warn); }}
.lvl-wrong {{ background: var(--bad); }}

.minibar-label {{ font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--ink-faint); margin: 0 0 0.7rem; }}
.mini-bars {{ display: flex; flex-direction: column; gap: 0.5rem; margin-bottom: 2rem; background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 1rem 1.2rem; }}
.mini-bar-row {{ display: grid; grid-template-columns: 12rem 1fr 2.6rem 1fr 2.6rem; align-items: center; gap: 0.6rem; }}
.mini-bar-label {{ font-size: 0.82rem; display: flex; align-items: center; gap: 0.4rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.mini-bar-track {{ background: var(--surface-alt); border-radius: 4px; height: 0.55rem; overflow: hidden; }}
.mini-bar-fill {{ height: 100%; }}
.fill-rule {{ background: var(--rule); }}
.fill-judge {{ background: var(--judge); }}
.mini-bar-val {{ font-size: 0.78rem; text-align: right; }}

.samples-label {{ font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--ink-faint); margin: 0 0 1rem; }}
.samples-list {{ display: flex; flex-direction: column; gap: 1.4rem; }}
.sample-row {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1rem 1.1rem; }}
.sample-row-head {{ display: flex; align-items: center; gap: 0.7rem; margin-bottom: 0.35rem; }}
.sample-idx {{ font-size: 0.72rem; color: var(--ink-faint); }}
.sample-id {{ font-size: 0.68rem; color: var(--ink-faint); }}
.sample-prompt {{ font-size: 0.76rem; color: var(--ink-soft); line-height: 1.4; margin: 0 0 0.8rem; max-height: 3em; overflow-y: auto; }}
.sample-grid {{ display: grid; grid-template-columns: repeat(6, minmax(120px, 1fr)); gap: 0.6rem; overflow-x: auto; }}
.mcell {{ display: flex; flex-direction: column; gap: 0.3rem; min-width: 110px; }}
.mcell-label {{ font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.02em; color: var(--ink-faint); display: flex; align-items: center; gap: 0.3rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.mcell img {{ width: 100%; aspect-ratio: 1; object-fit: contain; background: var(--surface-alt); border-radius: 6px; border: 1px solid var(--border); }}
.ref-cell img {{ border-style: dashed; }}
.mcell-scores {{ font-size: 0.72rem; text-align: center; }}

footer.meta {{ padding: 2.6rem 0 0; }}
footer.meta p {{ color: var(--ink-faint); font-size: 0.84rem; line-height: 1.7; max-width: 78ch; }}
@media (max-width: 800px) {{ .task-body-cols, .intro-cols {{ grid-template-columns: 1fr; }} .mini-bar-row {{ grid-template-columns: 8rem 1fr 2.2rem 1fr 2.2rem; }} }}
</style>

<header class="top">
  <div class="wrap">
    <p class="kicker">4 models &middot; VBVR-CustomEval &middot; 900 real generations each, identical test set</p>
    <h1>Four models, one bench: a head-to-head on visual reasoning</h1>
    <p class="lede">InternVL-U (base checkpoint), InternVL-U (fine-tuned on this task family), BAGEL, and SenseNova-U1 were each run on the identical 900-sample, 9-task target-frame-prediction split &mdash; same ids, same prompts, same ground truth &mdash; then scored twice: once by a deterministic rule-based detector pipeline, once by a Qwen3-VL-30B-fp8 judge. Because every model shares the exact same test instances, this report shows shared samples side by side per task rather than four separate galleries.</p>
    <div class="legend">{legend}</div>
    <div class="model-cards">{readout_cards()}</div>
    <nav class="tasknav">{nav_links}</nav>
  </div>
</header>

<section class="intro">
  <div class="wrap">
    <div class="intro-cols">
      <div class="intro-col">
        <h3><span class="ink-dot ink-dot-rule"></span>Rule-based method</h3>
        <p>Per task, a small set of weighted sub-criteria (e.g. for <code class="mono">stable_sort</code>: classification, order, fidelity, layout) is computed from classical computer vision &mdash; contour detection, HSV color masks, connected-component analysis &mdash; identically for all four models' outputs. Deterministic, zero marginal cost per sample.</p>
        <p>Includes two generic fixes found by auditing real generations: a <b>visual-coherence gate</b> (discounts scores when a candidate's foreground doesn't resemble any color in that sample's own ground truth &mdash; catches garbled outputs that fool loose per-task detectors) and a <b>maze path-coverage gate</b> (a short path stub can no longer score as if it were a full solve).</p>
      </div>
      <div class="intro-col">
        <h3><span class="ink-dot ink-dot-judge"></span>Judge method</h3>
        <p>A Qwen3-VL-30B-fp8 judge is shown the same three images (starting frame, candidate, ground truth) and classifies each sub-criterion into <span class="lvl lvl-wrong">wrong</span> <span class="lvl lvl-partial">partial</span> <span class="lvl lvl-correct">correct</span> (0 / 0.5 / 1.0), combined with the identical weights the rule-based side uses. Instructed to verify against actual pixel content and compare element-by-element rather than assume the task instruction was followed.</p>
        <div class="lesson-box"><b>A stronger version was tried and reverted.</b> Forcing the judge to write an "observation" before every verdict looked good on individually spot-tested cases, but at full scale it caused confabulation on ~20% of samples (a blank maze image scored a fabricated "complete path" 1.0). Reverted to the lighter instruction, which held up at full scale across all four models here.</div>
      </div>
    </div>
  </div>
</section>

<section class="summary">
  <div class="wrap">
    <h2>All 9 tasks &times; 4 models</h2>
    {summary_table()}
  </div>
</section>

<div class="wrap">
{sections}
</div>

<footer class="meta">
  <div class="wrap">
    <p><b>Models:</b> InternVL-U base and fine-tuned checkpoints, BAGEL-7B-MoT, SenseNova-U1-8B-MoT-SFT &mdash; all single-image generation, 512px output. <b>Judge:</b> <code>qwen3-vl-30b-fp8</code>, temperature 0, images downscaled to 512px on the long edge, one call per sample (900 &times; 4 = 3,600 total judge calls across this report, 0 errors). <b>Sampling:</b> {N_SHARED_PER_TASK} shared test instances per task, Python <code>random.Random({SEED})</code>, identical ids across all 4 models, not cherry-picked. <code>glass_refraction</code> is excluded from all 4 runs: no generations exist for it yet (its DataFactory generator repo is access-restricted).</p>
  </div>
</footer>
"""
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write(html)
    print(f"wrote {OUT_PATH} ({len(html)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
