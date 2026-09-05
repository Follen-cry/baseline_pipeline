#!/usr/bin/env python3
"""Build the 4-model comparison artifact for the S0-S3 pretraining-recipe
follow-up: four independently-pretrained InternVL-U checkpoints (S0-S3, each
a different 3-frames-in next-frame-prediction pretraining recipe over the
same ~49k-row mixed real+synthetic dataset -- see data/specs/S0_S3_DATASET_SPEC.md)
were each fine-tuned with an IDENTICAL recipe (LoRA rank 32, lr 1e-5, batch
16, no-ce, all conditioning images through the ViT) on the SAME 2,000-row,
4-task target-prediction set (500/task: multi_object_placement,
rotation_puzzle, shape_color_then_move, 2d_geometric_transformation --
2 ID + 2 OOD, subset of the earlier 9-task set). The only difference across
the four is which S0-S3 pretraining recipe the SFT started from:

  S0 (V2I-F)  : 3 frames -> fixed next frame, no caption
  S1 (V2I-V)  : 3 frames -> variable target (next OR preceding), no caption
  S2 (VC2I-F) : 3 frames + caption -> fixed next frame
  S3 (VC2IA-F): 3 frames + caption + 4 candidates -> fixed next frame + MCQ

All four are evaluated on the identical 400-sample (4 task x 100) eval
split. Adapted from gen_9task_sft_comparison.py's visual language, trimmed
to 4 tasks and (up to) 4 models.

Each model directory under results/vbvr_target_pred_eval/ is expected to
contain scored_new.json (rule-based, from validation/score_target_pred_eval_v2.py
-- the REWRITTEN scorers in VBVR-CustomEval/scorers/) and judge_scored.json
(from judge_eval/score_all_with_judge.py --run-dir).

Switched from the original scored.json on 2026-09-03. The old rule-based
scorers were each dominated by a detection failure that routed silently into a
constant fallback -- rotation_puzzle looked for blue pipes in images whose pipes
are orange/pink/yellow, multi_object_placement enumerated four hardcoded HSV
boxes and could not see magenta/cyan/orange objects, and so on -- so their
numbers were closer to constants than measurements. Set RULE_FILE back to
"scored.json" to regenerate the old version of this report.

    python gen_s0s3_4task_comparison.py
"""
import html as htmlmod
import json
import os
import random
import sys
from io import BytesIO

from PIL import Image

sys.path.insert(0, "/scratch/network/ssd2/junlin/ssl_mllm/Evaluation/VBVR-CustomEval/evaluators")
from llm_judge_full import TASK_CRITERIA, build_full_judge_instruction
from judge_yesno import TASK_QUESTIONS, build_yesno_instruction

RUN = "/scratch/network/ssd2/junlin/ssl_mllm/results/vbvr_target_pred_eval"
OUT_PATH = "/scratch/network/ssd2/junlin/ssl_mllm/Evaluation/VBVR-CustomEval/artifacts/s0s3_4task_comparison.html"

RULE_FILE = "scored_new.json"   # "scored.json" for the superseded scorers

# Tasks scored by the yes/no CHECKLIST judge (judge_yesno.py) instead of the
# weighted wrong/partial/correct rubric judge (llm_judge_full.py).
#
# rotation_puzzle was moved 2026-09-03. The rubric judge had saturated on it:
# over all 500 generations it emitted only three distinct scores, never once
# used the "partial" level in 2,000 criterion verdicts, and two of its four
# criteria were effectively constants (rotation_accuracy correct 498/500,
# position_preservation 499/500) that together pinned 50% of every score at a
# fixed 1.0. The checklist judge asks a small number of independent yes/no
# questions about the candidate against the ground truth only -- no starting
# frame, no task-instruction text -- and scores yes-count / question-count, so
# the question count IS the score granularity and there are no weights to tune.
YESNO_TASKS = {"rotation_puzzle"}
YESNO_FILE = "judge_scored_yesno.json"

SEED = 20260903  # resampled 2026-09-03 alongside the rule-scorer rewrite; the
                 # previous batch (seed 20260902) was drawn against the old
                 # scores and is deliberately not carried over.
N_SHARED_PER_TASK = 10

# Per-task seed override. rotation_puzzle was re-drawn on 2026-09-03 when it
# moved to the yes/no checklist judge: the old batch was sampled against the
# rubric judge's scores, so keeping it would show a set of images chosen under
# the scoring system that was just replaced. Every other task keeps SEED, so
# the rest of the report is unchanged sample-for-sample.
TASK_SEED = {"rotation_puzzle": 40260903}
TASK_N = {"rotation_puzzle": 15}  # per-task sample count override; default N_SHARED_PER_TASK elsewhere

MODELS = [
    ("base", "Base (no pretrain)", f"{RUN}/base_4task500sft", f"{RUN}/base_4task500sft/judge_scored.json"),
    ("s0", "S0 (V2I-F)", f"{RUN}/s0_4task500sft", f"{RUN}/s0_4task500sft/judge_scored.json"),
    ("s1", "S1 (V2I-V)", f"{RUN}/s1_4task500sft", f"{RUN}/s1_4task500sft/judge_scored.json"),
    ("s2", "S2 (VC2I-F)", f"{RUN}/s2_4task500sft", f"{RUN}/s2_4task500sft/judge_scored.json"),
    ("s3", "S3 (VC2IA-F)", f"{RUN}/s3_4task500sft", f"{RUN}/s3_4task500sft/judge_scored.json"),
]

CATEGORY_OF = {
    "multi_object_placement": "Perception", "rotation_puzzle": "Transformation",
    "shape_color_then_move": "Abstraction", "2d_geometric_transformation": "Transformation",
}
DOMAIN_OF = {
    "multi_object_placement": "ID", "rotation_puzzle": "ID",
    "shape_color_then_move": "OOD", "2d_geometric_transformation": "OOD",
}
TASK_ORDER = list(CATEGORY_OF.keys())

RULE_MECHANISM = {
    "multi_object_placement": "Objects and star markers are separated by area (an order of magnitude apart), then each ground-truth object is matched to at most one candidate object with color as a hard gate, so a wrong-colored object earns no positional credit. Placement tolerances scale with each object's own size rather than a fixed pixel radius. Ground truth shows no stars at all -- a correct solve covers them -- so a candidate showing none scores full star-invariance.",
    "rotation_puzzle": "The 2x2 tile grid is recovered from the ground-truth frame and applied to the candidate. Each tile is reduced to an exact discrete state: which of its four edge midpoints a pipe arm actually reaches, measured as a physical distance in pixels. Connectivity is then a question about junctions rather than mask overlap, and dangling arms pointing off-grid are subtracted.",
    "shape_color_then_move": "Objects are found relative to the detected background (a saturation mask collapses when a candidate repaints the scene), assigned to the six fixed grid cells, and each cell scored as a product of position, glyph identity (centred mask IoU) and color -- so one hard failure cannot be averaged away by the other two.",
    "2d_geometric_transformation": "Scored against the ground-truth final frame rather than the input. Orientation error is the rotation that best maps the candidate silhouette onto ground truth, found by exhaustive search over 360 degrees, which stays unambiguous under symmetry. Position is measured against the move the task actually asked for, so a shape left where it started scores zero by construction.",
}

MODEL_COLORS = {"base": "#A8395A", "s0": "#6B7A8F", "s1": "#8C6BAB", "s2": "#C9922B", "s3": "#4C8C5B"}


def esc(s):
    return htmlmod.escape(s or "", quote=True)


def img_data_uri(path, max_side=200, quality=78):
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
    import base64
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


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
        scored = json.load(open(os.path.join(run_dir, RULE_FILE)))
        meta = json.load(open(os.path.join(run_dir, "meta.json")))
        judge = json.load(open(judge_path))["results"]
        yn_path = os.path.join(run_dir, YESNO_FILE)
        judge_yn = json.load(open(yn_path))["results"] if os.path.exists(yn_path) else {}
        rule_by_id = {r["id"]: r for r in scored["records"]}
        meta_by_id = {r["id"]: r for r in meta["results"]}
        model_data[key] = {"label": label, "rule": rule_by_id, "judge": judge,
                           "judge_yesno": judge_yn, "meta": meta_by_id}

    def judge_score(d, task, rid):
        """The judge score for one sample, from whichever judge scores that task.
        Returns None if that judge has no usable score for it."""
        if task in YESNO_TASKS:
            rec = d["judge_yesno"].get(rid)
            return rec.get("score") if rec else None
        rec = d["judge"].get(rid)
        return rec.get("judge_score") if rec else None

    base_meta = model_data[MODELS[0][0]]["meta"]
    by_task_ids = {}
    for rid, row in base_meta.items():
        by_task_ids.setdefault(row["task_name"], []).append(rid)

    # IMPORTANT: this must reproduce the ORIGINAL single-shared-rng loop
    # exactly for every task except the ones in TASK_SEED. The original code
    # advanced one rng object across all tasks in `by_task_ids` dict order, so
    # each task's draw depended on how many draws happened before it -- a
    # per-task fresh Random(SEED) is NOT equivalent and silently redraws every
    # task, not just the overridden one. (Caught 2026-09-03 by diffing a
    # republish against the live version: shape_color_then_move's shown ids
    # had changed even though only rotation_puzzle was meant to move.)
    rng = random.Random(SEED)
    shared_picks = {t: rng.sample(ids, min(N_SHARED_PER_TASK, len(ids))) for t, ids in by_task_ids.items()}
    for t, seed in TASK_SEED.items():
        if t in by_task_ids:
            n = TASK_N.get(t, N_SHARED_PER_TASK)
            shared_picks[t] = random.Random(seed).sample(by_task_ids[t], min(n, len(by_task_ids[t])))

    task_stats, overall_stats = {}, {}
    for key, _, _, _ in MODELS:
        d = model_data[key]
        all_r, all_j = [], []
        for t in TASK_ORDER:
            rs, js = [], []
            for rid in by_task_ids.get(t, []):
                r = d["rule"].get(rid)
                j = judge_score(d, t, rid)
                if r and r["score"] is not None and j is not None:
                    rs.append(r["score"]); js.append(j)
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
            <div class="model-card" style="border-top-color:{MODEL_COLORS[key]}">
              <div class="model-card-head"><span class="model-dot" style="background:{MODEL_COLORS[key]}"></span>{esc(label)}</div>
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
              <span class="mini-bar-label"><span class="model-dot" style="background:{MODEL_COLORS[key]}"></span>{esc(label)}</span>
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
            gen_uri = img_data_uri(row["generated_image"]) if row else None
            rscore = r.get("score") or 0.0
            jscore = judge_score(d, t, rid) or 0.0
            # For the checklist judge, name the questions that came back "no" --
            # that is the whole point of scoring by question rather than weight.
            fail_chips = ""
            if t in YESNO_TASKS:
                rec = d["judge_yesno"].get(rid) or {}
                answers = rec.get("answers") or {}
                failed = [k for k, v in answers.items() if not v]
                if failed:
                    fail_chips = ('<div class="qfails">'
                                  + "".join(f'<span class="qfail">{esc(k)}</span>' for k in failed)
                                  + "</div>")
                elif answers:
                    fail_chips = '<div class="qfails"><span class="qpass">all 4 yes</span></div>'
            # fail_chips is "" for every non-checklist task; skip the line
            # entirely rather than interpolate an empty string, so those
            # tasks' markup stays byte-identical to before this feature existed.
            fail_chips_html = f"\n              {fail_chips}" if fail_chips else ""
            model_cells.append(f"""
            <div class="mcell">
              <div class="mcell-label"><span class="model-dot" style="background:{MODEL_COLORS[key]}"></span>{esc(label)}</div>
              <img src="{gen_uri}" alt="{esc(label)} output" loading="lazy">
              <div class="mcell-scores">
                <span class="rule-ink mono">{rscore:.2f}</span><span class="sep">/</span><span class="judge-ink mono">{jscore:.2f}</span>
              </div>{fail_chips_html}
            </div>""")

        return f"""
        <div class="sample-row">
          <div class="sample-row-head">
            <span class="mono sample-idx">#{idx:02d}</span>
            <span class="sample-id mono">{esc(rid.split('_eval_')[-1])}</span>
          </div>
          <p class="sample-prompt">{esc(prompt)}</p>
          <div class="sample-refs">
            <div class="mcell ref-cell">
              <div class="mcell-label">input</div>
              <img src="{input_uri}" alt="input" loading="lazy">
            </div>
            <div class="mcell ref-cell">
              <div class="mcell-label">ground truth</div>
              <img src="{gt_uri}" alt="ground truth" loading="lazy">
            </div>
          </div>
          <div class="sample-models">
            {"".join(model_cells)}
          </div>
        </div>"""

    def task_section(t):
        cat = CATEGORY_OF[t]
        dom = DOMAIN_OF[t]
        is_yesno = t in YESNO_TASKS
        if is_yesno:
            # The checklist judge takes no task-instruction text and no starting
            # frame, so the prompt it sends is the same for every sample.
            task_desc, questions = TASK_QUESTIONS[t]
            instruction = build_yesno_instruction(t)
            n_q = len(questions)
            crit_bars = (
                f'<p class="wcrit-note">Scored by the <b>yes/no checklist judge</b>: '
                f'{n_q} independent questions, each worth the same, '
                f'score = yes-count / {n_q}. Granularity is therefore 1/{n_q} '
                f'&mdash; {n_q + 1} possible scores, and no weights to tune.</p>'
                + "".join(
                    f'<div class="wcrit"><div class="wcrit-head"><code>{esc(k)}</code>'
                    f'<span class="mono">1/{n_q}</span></div>'
                    f'<div class="wcrit-defs"><span class="lvl lvl-correct">yes</span>{esc(qt)}</div></div>'
                    for k, qt in questions
                )
            )
        else:
            task_desc, criteria = TASK_CRITERIA[t]
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
              <p class="method-label judge-ink">{"Judge method &mdash; yes/no checklist, exact prompt sent" if is_yesno else "Judge method &mdash; exact prompt sent"}</p>
              <pre class="prompt-box mono">{esc(instruction)}</pre>
            </div>
          </div>
          <p class="minibar-label">Mean score per model (rule / judge)</p>
          <div class="mini-bars">{model_mini_bars(t)}</div>
          <p class="samples-label">{TASK_N.get(t, N_SHARED_PER_TASK)} shared test instances, same id across all {len(MODELS)} models (seed {TASK_SEED.get(t, SEED)})</p>
          <div class="samples-list">{rows}</div>
        </section>"""

    nav_links = "".join(f'<a href="#t-{esc(t)}">{esc(t)}</a>' for t in TASK_ORDER)
    sections = "".join(task_section(t) for t in TASK_ORDER)
    legend = "".join(
        f'<span class="legend-item"><span class="model-dot" style="background:{MODEL_COLORS[key]}"></span>{esc(label)}</span>'
        for key, label, _, _ in MODELS
    )

    html = f"""<title>S0-S3 Pretrain Comparison</title>
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
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #16181A; --surface: #1D2020; --surface-alt: #262A29;
    --ink: #ECEAE2; --ink-soft: #B8B4A6; --ink-faint: #837F72; --border: #34383A;
    --rule: #4FD6D2; --rule-soft: #16302F; --judge: #F17BB0; --judge-soft: #3A1E2C; --overlap: #B0A0F0;
    --good: #5FBFA3; --warn: #D9A548; --bad: #E37FA0;
    --cat-perception: #E37FA0; --cat-knowledge: #7DA8E8; --cat-transformation: #E0A356;
    --cat-abstraction: #A98FE0; --cat-spatiality: #5FBFA3;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #16181A; --surface: #1D2020; --surface-alt: #262A29;
  --ink: #ECEAE2; --ink-soft: #B8B4A6; --ink-faint: #837F72; --border: #34383A;
  --rule: #4FD6D2; --rule-soft: #16302F; --judge: #F17BB0; --judge-soft: #3A1E2C; --overlap: #B0A0F0;
  --good: #5FBFA3; --warn: #D9A548; --bad: #E37FA0;
  --cat-perception: #E37FA0; --cat-knowledge: #7DA8E8; --cat-transformation: #E0A356;
  --cat-abstraction: #A98FE0; --cat-spatiality: #5FBFA3;
}}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{ background: var(--bg); color: var(--ink); margin: 0; padding: 0 0 6rem; font-family: "Source Sans 3", system-ui, sans-serif; font-size: 16px; line-height: 1.55; }}
.mono {{ font-family: "JetBrains Mono", ui-monospace, monospace; font-variant-numeric: tabular-nums; }}
.wrap {{ max-width: 1360px; margin: 0 auto; padding: 0 2rem; }}
a {{ color: var(--rule); }}
.rule-ink {{ color: var(--rule); }}
.judge-ink {{ color: var(--judge); }}
.sep {{ color: var(--ink-faint); margin: 0 0.15em; }}

header.top {{ background: var(--surface); border-bottom: 1px solid var(--border); padding: 3.2rem 0 2rem; }}
.kicker {{ font-family: "JetBrains Mono", monospace; font-size: 0.76rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--ink-faint); margin: 0 0 0.8rem; }}
h1 {{ font-family: "Barlow Condensed", system-ui, sans-serif; font-weight: 700; font-size: clamp(2.1rem, 4.6vw, 3.2rem); line-height: 1.02; margin: 0 0 1rem; text-wrap: balance; letter-spacing: -0.01em; }}
.lede {{ color: var(--ink-soft); max-width: 76ch; font-size: 1.03rem; line-height: 1.6; margin: 0 0 1.8rem; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 1.1rem; margin-bottom: 1.6rem; }}
.legend-item {{ display: flex; align-items: center; gap: 0.4rem; font-size: 0.85rem; color: var(--ink-soft); }}
.model-dot {{ width: 0.7rem; height: 0.7rem; border-radius: 50%; display: inline-block; flex-shrink: 0; }}
.model-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 0.9rem; margin-bottom: 1.6rem; }}
.model-card {{ background: var(--surface-alt); border: 1px solid var(--border); border-radius: 10px; padding: 1rem 1.2rem; border-top: 3px solid var(--border); }}
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
.recipe-table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; margin-top: 0.6rem; }}
.recipe-table th, .recipe-table td {{ text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid var(--border); }}
.recipe-table th {{ color: var(--ink-faint); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.03em; }}

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
/* A change-bar, not decoration: the rail marks the one part of the method that
   was replaced. Selector is scoped to .intro-col so it outranks `.intro-col p`
   on specificity rather than leaning on !important. */
.intro-col p.method-note {{ font-size: 0.84rem; line-height: 1.55; color: var(--ink-soft); border-left: 2px solid var(--rule); padding: 0.1rem 0 0.1rem 0.85rem; margin: 0.9rem 0 0; }}
.method-note b {{ color: var(--rule); }}
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
.mini-bar-row {{ display: grid; grid-template-columns: 10rem 1fr 2.6rem 1fr 2.6rem; align-items: center; gap: 0.6rem; }}
.mini-bar-label {{ font-size: 0.82rem; display: flex; align-items: center; gap: 0.4rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.mini-bar-track {{ background: var(--surface-alt); border-radius: 4px; height: 0.55rem; overflow: hidden; }}
.mini-bar-fill {{ height: 100%; }}
.fill-rule {{ background: var(--rule); }}
.fill-judge {{ background: var(--judge); }}
.caveat {{ font-size: 12.5px; line-height: 1.6; margin: 10px 0 0; padding: 10px 12px;
  border-left: 3px solid #C9860B; background: color-mix(in srgb, #C9860B 9%, transparent);
  border-radius: 0 4px 4px 0; }}
.qfails {{ display: flex; flex-wrap: wrap; gap: 3px; justify-content: center; margin-top: 4px; }}
.qfail {{ font-size: 9.5px; font-family: ui-monospace, monospace; padding: 1px 4px; border-radius: 3px;
  background: var(--judge-soft); color: var(--judge); border: 1px solid color-mix(in srgb, var(--judge) 30%, transparent); }}
.qpass {{ font-size: 9.5px; font-family: ui-monospace, monospace; padding: 1px 4px; border-radius: 3px;
  background: var(--rule-soft); color: var(--rule); }}
.wcrit-note {{ font-size: 12.5px; line-height: 1.55; margin: 0 0 10px; padding: 8px 10px;
  border-left: 2px solid var(--judge); background: var(--judge-soft); border-radius: 0 4px 4px 0; }}
.mini-bar-val {{ font-size: 0.78rem; text-align: right; }}

.samples-label {{ font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--ink-faint); margin: 0 0 1rem; }}
.samples-list {{ display: flex; flex-direction: column; gap: 1.4rem; }}
.sample-row {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1rem 1.1rem; }}
.sample-row-head {{ display: flex; align-items: center; gap: 0.7rem; margin-bottom: 0.35rem; }}
.sample-idx {{ font-size: 0.72rem; color: var(--ink-faint); }}
.sample-id {{ font-size: 0.68rem; color: var(--ink-faint); }}
.sample-prompt {{ font-size: 0.76rem; color: var(--ink-soft); line-height: 1.4; margin: 0 0 0.8rem; max-height: 3em; overflow-y: auto; }}
.sample-refs {{ display: flex; gap: 0.6rem; margin-bottom: 0.7rem; padding-bottom: 0.7rem; border-bottom: 1px dashed var(--border); }}
.sample-refs .mcell {{ flex: 0 0 160px; }}
.sample-models {{ display: grid; grid-template-columns: repeat(5, minmax(120px, 1fr)); gap: 0.6rem; overflow-x: auto; }}
.mcell {{ display: flex; flex-direction: column; gap: 0.3rem; min-width: 110px; }}
.mcell-label {{ font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.02em; color: var(--ink-faint); display: flex; align-items: center; gap: 0.3rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.mcell img {{ width: 100%; aspect-ratio: 1; object-fit: contain; background: var(--surface-alt); border-radius: 6px; border: 1px solid var(--border); }}
.ref-cell img {{ border-style: dashed; }}
.mcell-scores {{ font-size: 0.72rem; text-align: center; }}

footer.meta {{ padding: 2.6rem 0 0; }}
footer.meta p {{ color: var(--ink-faint); font-size: 0.84rem; line-height: 1.7; max-width: 78ch; }}
@media (max-width: 900px) {{ .task-body-cols, .intro-cols {{ grid-template-columns: 1fr; }} .mini-bar-row {{ grid-template-columns: 7rem 1fr 2.2rem 1fr 2.2rem; }} .sample-refs .mcell {{ flex-basis: 120px; }} .sample-models {{ grid-template-columns: repeat(3, minmax(100px, 1fr)); }} }}
</style>

<header class="top">
  <div class="wrap">
    <p class="kicker">5 models &middot; VBVR-CustomEval &middot; identical SFT recipe, identical 2,000-row training set, identical 400-sample eval</p>
    <h1>Which pretraining recipe transfers best?</h1>
    <p class="lede">Five InternVL-U checkpoints &mdash; the plain base checkpoint (no S0&ndash;S3 pretraining at all) plus four independently pretrained on the same ~49k-row S0&ndash;S3 mixed real+synthetic dataset with a different next-frame-prediction recipe each (see the table below) &mdash; were then fine-tuned with an <b>identical</b> SFT recipe (LoRA rank 32, lr 1e-5, batch 16, no-ce, all conditioning images through the ViT) on the <b>same</b> 2,000-row, 4-task target-prediction set (500/task: multi_object_placement, rotation_puzzle, shape_color_then_move, 2d_geometric_transformation). All five are evaluated on the identical 400-sample (4 task &times; 100) eval split, scored twice: once by a deterministic rule-based detector pipeline, once by a Qwen3-VL-30B-fp8 judge.</p>
    <table class="recipe-table">
      <thead><tr><th>Setting</th><th>Input</th><th>Target</th><th>Caption</th><th>+Answer</th></tr></thead>
      <tbody>
        <tr><td class="mono">Base</td><td colspan="4">no S0&ndash;S3 pretraining &mdash; SFT applied directly to the plain InternVL-U checkpoint</td></tr>
        <tr><td class="mono">S0 (V2I-F)</td><td>3 frames</td><td>Fixed (next frame)</td><td>none</td><td>&mdash;</td></tr>
        <tr><td class="mono">S1 (V2I-V)</td><td>3 frames</td><td>Variable (next or preceding)</td><td>none</td><td>&mdash;</td></tr>
        <tr><td class="mono">S2 (VC2I-F)</td><td>3 frames + caption</td><td>Fixed (next frame)</td><td>yes</td><td>&mdash;</td></tr>
        <tr><td class="mono">S3 (VC2IA-F)</td><td>3 frames + caption + 4 candidates</td><td>Fixed (next frame)</td><td>yes</td><td>4-way MCQ</td></tr>
      </tbody>
    </table>
    <div class="legend" style="margin-top:1.4rem">{legend}</div>
    <div class="model-cards">{readout_cards()}</div>
    <nav class="tasknav">{nav_links}</nav>
  </div>
</header>

<section class="intro">
  <div class="wrap">
    <div class="intro-cols">
      <div class="intro-col">
        <h3><span class="ink-dot ink-dot-rule"></span>Rule-based method</h3>
        <p>Per task, a small set of weighted sub-criteria is computed from classical computer vision &mdash; contour detection, connected components, CIELAB color matching &mdash; identically for all five models' outputs. Deterministic, zero marginal cost per sample, no learned models.</p>
        <p class="method-note"><b>Rebuilt 2026-09-03.</b> These are the rewritten scorers in <span class="mono">VBVR-CustomEval/scorers/</span>. Every one of the four originals was dominated by a detection failure that routed silently into a constant fallback rather than a measurement &mdash; <span class="mono">rotation_puzzle</span> searched for <i>blue</i> pipes in images whose pipes are rendered orange, pink or yellow per sample, and returned its <span class="mono">0.5</span> constants when the mask came back empty. The rewrites match each ground-truth object to a candidate counterpart with color as a hard gate and score position, pose and shape per object, so an element present <i>somewhere</i> in the image earns no positional credit. Every threshold is derived from a measurement on this split; each task's <span class="mono">&lt;task&gt;_notes.md</span> records the derivation and the remaining failure modes.</p>
      </div>
      <div class="intro-col">
        <h3><span class="ink-dot ink-dot-judge"></span>Judge method</h3>
        <p>A Qwen3-VL-30B-fp8 judge is shown the same three images (starting frame, candidate, ground truth) and classifies each sub-criterion into <span class="lvl lvl-wrong">wrong</span> <span class="lvl lvl-partial">partial</span> <span class="lvl lvl-correct">correct</span> (0 / 0.5 / 1.0), combined with the identical weights the rule-based side uses.</p>
        <p><b>Except <code>rotation_puzzle</code></b>, which as of 2026-09-03 uses a <b>yes/no checklist judge</b> instead. On that task the rubric above had saturated: across all 500 generations it produced only three distinct scores, never used <span class="lvl lvl-partial">partial</span> in 2,000 verdicts, and two of its four criteria were constants (correct on 498/500 and 499/500) that pinned half of every score at a fixed 1.0. The checklist judge sees <b>only two images</b> &mdash; candidate and ground truth, no starting frame &mdash; and is given <b>no task-instruction text</b>, so it cannot grade against what the task asked for rather than what the pixels show. It answers a short list of independent yes/no questions and scores yes-count / question-count.</p>
        <p class="caveat"><b>Read the <code>rotation_puzzle</code> judge column with this caveat.</b> The ground-truth images never draw through the gutters between tiles (measured: 0.000 gutter ink in all 100 references), so a solved ring can be rendered two ways &mdash; GT-style with the gap kept, or as one continuous rectangle drawn through it. The <code>joins_closed</code> question tells the judge that both count, but it does not obey: it answers <i>no</i> on 52&ndash;73% of gutter-crossing candidates versus 14&ndash;27% of GT-style ones. The variants use that rendering at different rates (Base 37% of generations, S1 15%), so the between-model spread below is partly rendering style rather than puzzle-solving. Restricted to GT-style candidates only, the spread falls from <b>0.100 to 0.053</b> &mdash; against 0.039 for the deterministic scorer and 0.032 for the rubric judge on the same subset. The checklist is a real but modest improvement in separation, not the 3&times; the raw numbers suggest.</p>
      </div>
    </div>
  </div>
</section>

<section class="summary">
  <div class="wrap">
    <h2>All 4 tasks &times; 5 models</h2>
    {summary_table()}
  </div>
</section>

<div class="wrap">
{sections}
</div>

<footer class="meta">
  <div class="wrap">
    <p><b>Models:</b> five InternVL-U checkpoints, single-image generation, 512px output; only the pretraining stage (none, for "Base"; S0-S3 otherwise) differs -- the downstream SFT recipe and data are identical across all five. <b>Judge:</b> <code>qwen3-vl-30b-fp8</code>, temperature 0, images downscaled to 512px on the long edge, one call per sample (400 &times; 5 = 2,000 total judge calls across this report; 1 malformed-JSON response each on S1 and Base defaulted to score 0 for that single sample). <b>Sampling:</b> {N_SHARED_PER_TASK} shared test instances per task, Python <code>random.Random({SEED})</code> (<code>rotation_puzzle</code>: {TASK_N["rotation_puzzle"]} instances, re-drawn with seed <code>{TASK_SEED["rotation_puzzle"]}</code> when it moved to the checklist judge), identical ids across all five models, not cherry-picked.</p>
  </div>
</footer>
"""
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write(html)
    print(f"wrote {OUT_PATH} ({len(html)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
