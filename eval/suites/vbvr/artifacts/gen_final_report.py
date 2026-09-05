#!/usr/bin/env python3
"""Generate the final consolidated eval-results artifact: base InternVL-U's
900 real generations, scored by both the rule-based pipeline and the Qwen
judge (final validated versions, post round-2 fixes), with method
explanations, exact judge prompts, and 10 visual examples per task."""
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

RUN_DIR = "/scratch/network/ssd2/junlin/ssl_mllm/results/vbvr_target_pred_eval/base_prescale512_vaecond"
JUDGE_PATH = "/scratch/network/ssd2/junlin/ssl_mllm/Evaluation/VBVR-CustomEval/judge_eval/judge_scored_900.json"
OUT_PATH = "/scratch/network/ssd2/junlin/ssl_mllm/Evaluation/VBVR-CustomEval/artifacts/final_eval_report.html"

SEED = 82720260827
N_PER_TASK = 10

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
    "maze": "Orange/yellow path-marker mask vs. black wall mask checks the drawn path never crosses a wall, reaches both markers, moves only between adjacent cells, and the maze itself is untouched.",
    "2d_geometric_transformation": "HSV saturation isolates the colored shape from its grayscale scene; rotation angle (via ellipse fit), position, and size are compared to the dashed target outline.",
}

def esc(s):
    return htmlmod.escape(s or "", quote=True)

def img_data_uri(path, max_side=260, quality=80):
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
    scored = json.load(open(os.path.join(RUN_DIR, "scored.json")))
    meta = json.load(open(os.path.join(RUN_DIR, "meta.json")))
    judge = json.load(open(JUDGE_PATH))["results"]
    by_id_meta = {r["id"]: r for r in meta["results"]}
    rule_by_id = {r["id"]: r for r in scored["records"]}

    by_task_ids = {}
    for r in meta["results"]:
        by_task_ids.setdefault(r["task_name"], []).append(r["id"])

    rng = random.Random(SEED)
    picks = {t: rng.sample(ids, min(N_PER_TASK, len(ids))) for t, ids in by_task_ids.items()}

    # per-task + overall stats (full 900)
    task_stats = {}
    all_r, all_j = [], []
    for t in TASK_ORDER:
        ids = by_task_ids.get(t, [])
        rs, js = [], []
        for rid in ids:
            r = rule_by_id.get(rid)
            j = judge.get(rid)
            if r and r["score"] is not None and j and j.get("judge_score") is not None:
                rs.append(r["score"]); js.append(j["judge_score"])
        rho = spearman(rs, js) if len(rs) > 1 else None
        task_stats[t] = {
            "n": len(rs),
            "rule_mean": sum(rs) / len(rs) if rs else 0,
            "judge_mean": sum(js) / len(js) if js else 0,
            "rho": rho,
        }
        all_r += rs; all_j += js
    overall_rho = spearman(all_r, all_j)
    overall = {
        "n": len(all_r),
        "rule_mean": sum(all_r) / len(all_r),
        "judge_mean": sum(all_j) / len(all_j),
        "rho": overall_rho,
    }

    def agreement_class(r, j):
        d = abs(r - j)
        if d < 0.2: return "agree"
        if d < 0.45: return "mixed"
        return "disagree"

    def sample_card(idx, rid):
        row = by_id_meta[rid]
        r = rule_by_id.get(rid, {})
        j = judge.get(rid, {})
        rscore = r.get("score") or 0.0
        jscore = j.get("judge_score") or 0.0
        cls = agreement_class(rscore, jscore)
        crit = j.get("judge_criteria") or {}
        crit_html = "".join(
            f'<div class="crit-row"><code>{esc(k)}</code><span class="lvl lvl-{esc(v)}">{esc(v)}</span></div>'
            for k, v in crit.items()
        )
        input_uri = img_data_uri(row["input_image"])
        gen_uri = img_data_uri(row["generated_image"])
        gt_uri = img_data_uri(row["target_image"])
        prompt = row["prompt"].split("Task: ", 1)[-1] if "Task: " in row["prompt"] else row["prompt"]
        return f"""
        <article class="sample sample-{cls}">
          <div class="sample-top">
            <span class="sample-idx">#{idx:02d}</span>
            <span class="agree-tag agree-tag-{cls}">{ {"agree":"agree","mixed":"partial agreement","disagree":"disagree"}[cls] }</span>
          </div>
          <div class="sample-imgs">
            <figure><img src="{input_uri}" alt="input" loading="lazy"><figcaption>input</figcaption></figure>
            <figure><img src="{gen_uri}" alt="generated" loading="lazy"><figcaption>generated</figcaption></figure>
            <figure><img src="{gt_uri}" alt="ground truth" loading="lazy"><figcaption>ground truth</figcaption></figure>
          </div>
          <p class="sample-prompt">{esc(prompt)}</p>
          <div class="gauges">
            <div class="gauge">
              <div class="gauge-label"><span>rule</span><span class="mono">{rscore:.2f}</span></div>
              <div class="gauge-track"><div class="gauge-fill fill-rule" style="width:{rscore*100:.1f}%;"></div></div>
            </div>
            <div class="gauge">
              <div class="gauge-label"><span>judge</span><span class="mono">{jscore:.2f}</span></div>
              <div class="gauge-track"><div class="gauge-fill fill-judge" style="width:{jscore*100:.1f}%;"></div></div>
            </div>
          </div>
          {f'<div class="crit-grid">{crit_html}</div>' if crit_html else ''}
          <p class="sample-reasoning">&ldquo;{esc(j.get("judge_reasoning") or "")}&rdquo;</p>
        </article>"""

    def task_section(t):
        task_desc, criteria = TASK_CRITERIA[t]
        stats = task_stats[t]
        cat = CATEGORY_OF[t]
        dom = DOMAIN_OF[t]
        ex_ids = picks[t]
        rep_prompt = by_id_meta[ex_ids[0]]["prompt"]
        instruction = build_full_judge_instruction(t, rep_prompt)

        crit_bars = "".join(
            f'<div class="wcrit"><div class="wcrit-head"><code>{esc(k)}</code><span class="mono">{w:g}%</span></div>'
            f'<div class="wcrit-defs"><span class="lvl lvl-correct">correct</span>{esc(lv["correct"])}</div>'
            f'<div class="wcrit-defs"><span class="lvl lvl-partial">partial</span>{esc(lv["partial"])}</div>'
            f'<div class="wcrit-defs"><span class="lvl lvl-wrong">wrong</span>{esc(lv["wrong"])}</div></div>'
            for k, w, lv in criteria
        )
        rho_str = f"{stats['rho']:.3f}" if stats["rho"] is not None else "n/a"
        cards = "".join(sample_card(i + 1, rid) for i, rid in enumerate(ex_ids))

        return f"""
        <section class="task" id="t-{esc(t)}">
          <header class="task-head">
            <div class="task-head-top">
              <span class="pill pill-cat cat-{esc(cat.lower())}">{esc(cat)}</span>
              <span class="pill pill-dom">{esc(dom)}</span>
              <h2>{esc(t)}</h2>
            </div>
            <div class="task-readout">
              <div class="readout"><span class="readout-label">rule mean</span><span class="readout-val mono rule-ink">{stats['rule_mean']:.3f}</span></div>
              <div class="readout"><span class="readout-label">judge mean</span><span class="readout-val mono judge-ink">{stats['judge_mean']:.3f}</span></div>
              <div class="readout"><span class="readout-label">agreement (&rho;)</span><span class="readout-val mono">{rho_str}</span></div>
            </div>
          </header>
          <p class="task-desc">{esc(task_desc)}</p>
          <div class="method-cols">
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
          <p class="samples-label">10 random real generations (seed {SEED})</p>
          <div class="samples-grid">{cards}</div>
        </section>"""

    nav_links = "".join(f'<a href="#t-{esc(t)}">{esc(t)}</a>' for t in TASK_ORDER)
    sections = "".join(task_section(t) for t in TASK_ORDER)

    summary_rows = "".join(f"""
        <tr>
          <td><span class="pill pill-cat cat-{esc(CATEGORY_OF[t].lower())}">{esc(CATEGORY_OF[t])}</span></td>
          <td class="mono">{esc(t)}</td>
          <td class="mono num">{task_stats[t]['rule_mean']:.3f}</td>
          <td class="mono num">{task_stats[t]['judge_mean']:.3f}</td>
          <td class="mono num">{f"{task_stats[t]['rho']:.3f}" if task_stats[t]['rho'] is not None else '&mdash;'}</td>
        </tr>""" for t in TASK_ORDER)

    html = f"""<title>Dual-Gauge Eval Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700;800&family=Source+Sans+3:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {{
  --bg: #EEEBE3;
  --surface: #FBFAF6;
  --surface-alt: #E4E0D2;
  --ink: #201F1B;
  --ink-soft: #55524A;
  --ink-faint: #8A8577;
  --border: #D6D1C0;
  --rule: #0E7C7B;
  --rule-soft: #D9EFEE;
  --judge: #B3236B;
  --judge-soft: #F7DEEB;
  --overlap: #4A3B6B;
  --good: #1C6B5E;
  --warn: #A9700E;
  --bad: #A8395A;
  --cat-perception: #A8395A;
  --cat-knowledge: #2A5CA6;
  --cat-transformation: #B06A1A;
  --cat-abstraction: #6B4FA0;
  --cat-spatiality: #1C6B5E;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #16181A; --surface: #1D2020; --surface-alt: #262A29;
    --ink: #ECEAE2; --ink-soft: #B8B4A6; --ink-faint: #837F72;
    --border: #34383A; --rule: #4FD6D2; --rule-soft: #16302F;
    --judge: #F17BB0; --judge-soft: #3A1E2C; --overlap: #B0A0F0;
    --good: #5FBFA3; --warn: #D9A548; --bad: #E37FA0;
    --cat-perception: #E37FA0; --cat-knowledge: #7DA8E8; --cat-transformation: #E0A356;
    --cat-abstraction: #A98FE0; --cat-spatiality: #5FBFA3;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #16181A; --surface: #1D2020; --surface-alt: #262A29;
  --ink: #ECEAE2; --ink-soft: #B8B4A6; --ink-faint: #837F72;
  --border: #34383A; --rule: #4FD6D2; --rule-soft: #16302F;
  --judge: #F17BB0; --judge-soft: #3A1E2C; --overlap: #B0A0F0;
  --good: #5FBFA3; --warn: #D9A548; --bad: #E37FA0;
  --cat-perception: #E37FA0; --cat-knowledge: #7DA8E8; --cat-transformation: #E0A356;
  --cat-abstraction: #A98FE0; --cat-spatiality: #5FBFA3;
}}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{
  background: var(--bg); color: var(--ink); margin: 0; padding: 0 0 6rem;
  font-family: "Source Sans 3", system-ui, sans-serif; font-size: 16px; line-height: 1.55;
}}
.mono {{ font-family: "JetBrains Mono", ui-monospace, monospace; font-variant-numeric: tabular-nums; }}
.disp {{ font-family: "Barlow Condensed", system-ui, sans-serif; }}
.wrap {{ max-width: 1180px; margin: 0 auto; padding: 0 2rem; }}
a {{ color: var(--rule); }}

header.top {{ background: var(--surface); border-bottom: 1px solid var(--border); padding: 3.2rem 0 2rem; }}
.kicker {{ font-family: "JetBrains Mono", monospace; font-size: 0.76rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--ink-faint); margin: 0 0 0.8rem; }}
h1 {{ font-family: "Barlow Condensed", system-ui, sans-serif; font-weight: 700; font-size: clamp(2.2rem, 5vw, 3.4rem); line-height: 1.02; margin: 0 0 1rem; text-wrap: balance; letter-spacing: -0.01em; }}
.lede {{ color: var(--ink-soft); max-width: 68ch; font-size: 1.05rem; line-height: 1.6; margin: 0 0 1.8rem; }}
.readout-row {{ display: flex; flex-wrap: wrap; gap: 0.9rem; margin-bottom: 1.6rem; }}
.readout-card {{ background: var(--surface-alt); border: 1px solid var(--border); border-radius: 10px; padding: 1rem 1.3rem; min-width: 11rem; }}
.readout-card .rc-label {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--ink-faint); margin-bottom: 0.3rem; }}
.readout-card .rc-val {{ font-family: "Barlow Condensed", system-ui, sans-serif; font-weight: 700; font-size: 2.1rem; line-height: 1; }}
.readout-card .rc-sub {{ font-size: 0.78rem; color: var(--ink-faint); margin-top: 0.25rem; }}
.rc-rule .rc-val {{ color: var(--rule); }}
.rc-judge .rc-val {{ color: var(--judge); }}
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
.lesson-box {{ background: var(--surface-alt); border-left: 3px solid var(--overlap); border-radius: 6px; padding: 0.9rem 1.1rem; margin-top: 1rem; font-size: 0.92rem; color: var(--ink-soft); }}
.lesson-box b {{ color: var(--ink); }}

section.summary {{ padding: 2.6rem 0; border-bottom: 1px solid var(--border); }}
section.summary h2, section.task .task-head h2 {{ font-family: "Barlow Condensed", system-ui, sans-serif; font-weight: 700; }}
section.summary h2 {{ font-size: 1.9rem; margin: 0 0 1.2rem; }}
.table-wrap {{ overflow-x: auto; border: 1px solid var(--border); border-radius: 10px; background: var(--surface); }}
table.summary-table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
table.summary-table th {{ text-align: left; padding: 0.7rem 1rem; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--ink-faint); border-bottom: 1px solid var(--border); }}
table.summary-table td {{ padding: 0.6rem 1rem; border-bottom: 1px solid var(--border); }}
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
.task-readout {{ display: flex; gap: 0.8rem; flex-wrap: wrap; margin-bottom: 1.4rem; }}
.readout {{ background: var(--surface-alt); border: 1px solid var(--border); border-radius: 8px; padding: 0.5rem 0.9rem; display: flex; flex-direction: column; gap: 0.15rem; }}
.readout-label {{ font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--ink-faint); }}
.readout-val {{ font-size: 1.15rem; font-weight: 600; }}
.rule-ink {{ color: var(--rule); }}
.judge-ink {{ color: var(--judge); }}
.task-desc {{ color: var(--ink-soft); max-width: 78ch; margin: 0 0 1.6rem; }}
.method-cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-bottom: 2rem; align-items: start; }}
.method-label {{ font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; margin: 0 0 0.6rem; }}
.method-text {{ font-size: 0.88rem; color: var(--ink-soft); margin: 0 0 0.9rem; }}
.wcrit {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 0.65rem 0.85rem; margin-bottom: 0.6rem; }}
.wcrit-head {{ display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 0.35rem; }}
.wcrit-head code {{ color: var(--ink); }}
.wcrit-defs {{ font-size: 0.76rem; color: var(--ink-faint); margin-bottom: 0.15rem; }}
.prompt-box {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 0.9rem 1rem; font-size: 0.7rem; line-height: 1.5; white-space: pre-wrap; word-break: break-word; max-height: 30rem; overflow-y: auto; color: var(--ink-soft); }}
.lvl {{ display: inline-block; font-family: "JetBrains Mono", monospace; font-size: 0.62rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em; padding: 0.08rem 0.4rem; border-radius: 4px; color: #fff; margin-right: 0.4rem; }}
.lvl-correct {{ background: var(--good); }}
.lvl-partial {{ background: var(--warn); }}
.lvl-wrong {{ background: var(--bad); }}

.samples-label {{ font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--ink-faint); margin: 0 0 0.9rem; }}
.samples-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; }}
.sample {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 0.9rem; border-top: 3px solid var(--border); }}
.sample-agree {{ border-top-color: var(--good); }}
.sample-mixed {{ border-top-color: var(--warn); }}
.sample-disagree {{ border-top-color: var(--bad); }}
.sample-top {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem; }}
.sample-idx {{ font-family: "JetBrains Mono", monospace; font-size: 0.72rem; color: var(--ink-faint); }}
.agree-tag {{ font-size: 0.64rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em; padding: 0.1rem 0.5rem; border-radius: 999px; color: #fff; }}
.agree-tag-agree {{ background: var(--good); }}
.agree-tag-mixed {{ background: var(--warn); }}
.agree-tag-disagree {{ background: var(--bad); }}
.sample-imgs {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.4rem; margin-bottom: 0.6rem; }}
.sample-imgs figure {{ margin: 0; }}
.sample-imgs img {{ width: 100%; aspect-ratio: 1; object-fit: contain; background: var(--surface-alt); border-radius: 5px; border: 1px solid var(--border); }}
.sample-imgs figcaption {{ text-align: center; font-size: 0.6rem; color: var(--ink-faint); margin-top: 0.2rem; text-transform: uppercase; letter-spacing: 0.02em; }}
.sample-prompt {{ font-size: 0.74rem; color: var(--ink-faint); line-height: 1.4; max-height: 3.4em; overflow-y: auto; margin: 0 0 0.7rem; }}
.gauges {{ display: flex; flex-direction: column; gap: 0.4rem; margin-bottom: 0.6rem; }}
.gauge-label {{ display: flex; justify-content: space-between; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--ink-faint); margin-bottom: 0.15rem; }}
.gauge-track {{ background: var(--surface-alt); border-radius: 4px; height: 0.5rem; overflow: hidden; }}
.gauge-fill {{ height: 100%; }}
.fill-rule {{ background: var(--rule); }}
.fill-judge {{ background: var(--judge); }}
.crit-grid {{ display: flex; flex-direction: column; gap: 0.2rem; margin-bottom: 0.6rem; }}
.crit-row {{ display: flex; justify-content: space-between; align-items: center; font-size: 0.72rem; }}
.crit-row code {{ color: var(--ink-soft); }}
.sample-reasoning {{ font-size: 0.74rem; color: var(--ink-faint); font-style: italic; line-height: 1.4; margin: 0; }}

footer.meta {{ padding: 2.6rem 0 0; }}
footer.meta p {{ color: var(--ink-faint); font-size: 0.84rem; line-height: 1.7; max-width: 74ch; }}
.overflow-x {{ overflow-x: auto; }}
@media (max-width: 800px) {{ .method-cols, .intro-cols {{ grid-template-columns: 1fr; }} }}
</style>

<header class="top">
  <div class="wrap">
    <p class="kicker">InternVL-U (base) &middot; VBVR-CustomEval &middot; 900 real generations</p>
    <h1>Two instruments, one bench: scoring InternVL-U's generations</h1>
    <p class="lede">Every one of the 900 real outputs from InternVL-U's base checkpoint (no VBVR fine-tuning) across the 9 active visual-reasoning tasks was measured twice: once by a deterministic pixel/contour detector pipeline, once by a Qwen3-VL-30B-fp8 judge reading the same three images. Same rubric, same weights, two different instruments &mdash; this report explains both, shows the exact prompt the judge saw per task, and samples 10 real examples per task so you can see where the two gauges agree and where they don't.</p>
    <div class="readout-row">
      <div class="readout-card"><div class="rc-label">Samples scored</div><div class="rc-val mono">{overall['n']}/900</div></div>
      <div class="readout-card rc-rule"><div class="rc-label">Rule-based mean</div><div class="rc-val mono">{overall['rule_mean']:.3f}</div></div>
      <div class="readout-card rc-judge"><div class="rc-label">Judge mean</div><div class="rc-val mono">{overall['judge_mean']:.3f}</div></div>
      <div class="readout-card"><div class="rc-label">Agreement (Spearman &rho;)</div><div class="rc-val mono">{overall['rho']:.3f}</div><div class="rc-sub">up from 0.140 pre-fix baseline</div></div>
    </div>
    <nav class="tasknav">{nav_links}</nav>
  </div>
</header>

<section class="intro">
  <div class="wrap">
    <div class="intro-cols">
      <div class="intro-col">
        <h3><span class="ink-dot ink-dot-rule"></span>Rule-based method</h3>
        <p>Per task, a small set of weighted sub-criteria (e.g. for <code class="mono">stable_sort</code>: classification, order, fidelity, layout) is computed from classical computer vision &mdash; contour detection, HSV color masks, connected-component analysis &mdash; run once on the input frame and once on the generated frame. Deterministic, fast, zero cost per sample.</p>
        <p>Two generic fixes were added after auditing real (not synthetic) generations by eye:</p>
        <ul>
          <li><b>Visual-coherence gate</b> (all 9 tasks): real generations from an untrained base model are often visually incoherent in ways clean test images never are &mdash; a task_specific score of 0.995 was found for a sample whose "ball" was actually a thin orange ring with no resemblance to the ground truth's solid pink circle. The gate checks what fraction of the candidate's foreground doesn't resemble any color in that sample's own ground truth (by pixel area AND by connected component, so a small wrong object can't hide inside a large correct one), and discounts the score accordingly.</li>
          <li><b>Maze path-coverage gate</b>: the original detector was blind to path length &mdash; a 5-pixel scribble that touched no wall scored as high as a full solve. Now gated by drawn-path pixels &divide; the ground truth's own path pixel count.</li>
        </ul>
      </div>
      <div class="intro-col">
        <h3><span class="ink-dot ink-dot-judge"></span>Judge method</h3>
        <p>A Qwen3-VL-30B-fp8 judge is shown the same three images (starting frame, candidate, ground truth) and classifies each of the same sub-criteria into <span class="lvl lvl-wrong">wrong</span> <span class="lvl lvl-partial">partial</span> <span class="lvl lvl-correct">correct</span> (mapped to 0 / 0.5 / 1.0), combined with the identical weights the rule-based side uses &mdash; a like-for-like swap of detector code for judge vision, not a different rubric.</p>
        <p>The prompt instructs it to verify against actual pixel content rather than assume the task instruction was followed, and to compare element-by-element (grid cell by grid cell, or step by step) rather than forming a general impression.</p>
        <div class="lesson-box"><b>A stronger version was tried and reverted.</b> Forcing the judge to write an "observation" before every verdict fixed individually-flagged cases in spot tests, but at full scale (all 900) it was a regression: judge means jumped up almost everywhere and rule-vs-judge correlation <i>dropped</i>. A maze image with nothing drawn at all scored a fabricated "complete path of yellow dots" 1.0 &mdash; on 20% of all 900 samples. Forcing a plausible-sounding justification made the model more prone to confabulating support for a verdict, not more grounded. Reverted to the lighter instruction above, which held up at full scale.</div>
      </div>
    </div>
  </div>
</section>

<section class="summary">
  <div class="wrap">
    <h2>All 9 tasks</h2>
    <div class="table-wrap">
      <table class="summary-table">
        <thead><tr><th>Category</th><th>Task</th><th class="num">Rule mean</th><th class="num">Judge mean</th><th class="num">Agreement (&rho;)</th></tr></thead>
        <tbody>{summary_rows}</tbody>
      </table>
    </div>
  </div>
</section>

<div class="wrap">
{sections}
</div>

<footer class="meta">
  <div class="wrap">
    <p><b>Model under test:</b> InternVL-U base checkpoint (no VBVR fine-tuning), single-image generation, 512px output, 512px input prescale. <b>Judge:</b> <code>qwen3-vl-30b-fp8</code>, temperature 0, images downscaled to 512px on the long edge. <b>Sampling:</b> 10 random examples per task, Python <code>random.Random({SEED})</code>, not cherry-picked. Agreement badges: green = rule and judge within 0.2 of each other, amber = within 0.45, red = further apart. <code>glass_refraction</code> is excluded: no real InternVL-U generations exist for it yet (its DataFactory generator repo is access-restricted).</p>
  </div>
</footer>
"""
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write(html)
    print(f"wrote {OUT_PATH} ({len(html)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
