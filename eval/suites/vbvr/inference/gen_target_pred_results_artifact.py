#!/usr/bin/env python3
"""Build the InternVL-U target-frame-prediction eval-results artifact:
real base-model generations (inference/run_target_pred_eval.py) scored by
the current image-only pipeline (validation/score_target_pred_eval.py).

    python gen_target_pred_results_artifact.py \
        --scored /path/to/scored.json --out target_pred_results.html
"""
import argparse
import base64
import html as htmlmod
import json
import os
import random
from io import BytesIO

from PIL import Image

GALLERY_SEED = 42
GALLERY_N = 10

CATEGORY_ORDER = ["Abstraction", "Knowledge", "Perception", "Spatiality", "Transformation"]
CAT_VAR = {
    "Abstraction": "--cat-abstraction",
    "Knowledge": "--cat-knowledge",
    "Perception": "--cat-perception",
    "Spatiality": "--cat-spatiality",
    "Transformation": "--cat-transformation",
}


def esc(s):
    return htmlmod.escape(s or "", quote=True)


def img_data_uri(path, max_side=360, quality=82):
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


def load(scored_path):
    scored = json.load(open(scored_path))
    run_dir = os.path.dirname(os.path.abspath(scored_path))
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    by_id = {r["id"]: r for r in meta["results"]}
    return scored, meta, by_id


def load_judge(judge_scored_path):
    if not judge_scored_path or not os.path.exists(judge_scored_path):
        return {}
    return json.load(open(judge_scored_path))["results"]


def pick_random(records, seed=GALLERY_SEED, n=GALLERY_N):
    """n freshly-random samples per task (not cherry-picked best/worst/
    representative) -- a fixed seed makes the pick reproducible across runs
    on the same underlying record list."""
    scored_recs = sorted([r for r in records if r["score"] is not None], key=lambda r: r["id"])
    if len(scored_recs) <= n:
        return scored_recs
    return random.Random(seed).sample(scored_recs, n)


def agreement_badge(rule_score, judge_score):
    diff = judge_score - rule_score
    if diff >= 0.5:
        return '<span class="badge badge-severe">Judge scored much higher</span>'
    if abs(diff) >= 0.5:
        return '<span class="badge badge-severe">Judge scored much lower</span>'
    if abs(diff) >= 0.25:
        return '<span class="badge badge-warn">Disagreement</span>'
    return '<span class="badge badge-ok">Close agreement</span>'


def sample_card(idx, rec, by_id, judge_by_id):
    row = by_id[rec["id"]]
    input_uri = img_data_uri(row["input_image"])
    gen_uri = img_data_uri(row["generated_image"])
    gt_uri = img_data_uri(row["target_image"])
    prompt = row["prompt"].split("Task: ", 1)[-1] if "Task: " in row["prompt"] else row["prompt"]

    judge = judge_by_id.get(rec["id"])
    judge_block = '<p class="judge-missing">No judge score available.</p>'
    badge = ""
    if judge and judge.get("judge_criteria"):
        crit_rows = "".join(
            f'<div class="crow"><code>{esc(k)}</code><span class="level-tag level-{esc(v)}">{esc(v)}</span></div>'
            for k, v in judge["judge_criteria"].items()
        )
        badge = agreement_badge(rec["score"], judge["judge_score"])
        judge_block = f"""
        <div class="score-box accent-box">
          <div class="score-label">Judge (Qwen3-VL-30B)</div>
          <div class="score-num mono">{judge['judge_score']:.3f}</div>
          <div class="crit-breakdown">{crit_rows}</div>
        </div>"""
        reasoning = f"""<p class="judge-reasoning"><b>Judge reasoning:</b> &ldquo;{esc(judge.get('judge_reasoning') or '')}&rdquo;</p>"""
    else:
        reasoning = ""

    return f"""
    <div class="sample">
      <div class="sample-head">
        <span class="sample-label">Sample {idx} &middot; <span class="mono">{esc(rec['id'])}</span></span>
        {badge}
      </div>
      <div class="sample-imgs">
        <figure><img src="{input_uri}" alt="input" loading="lazy"><figcaption>input</figcaption></figure>
        <figure><img src="{gen_uri}" alt="InternVL-U output" loading="lazy"><figcaption>InternVL-U output</figcaption></figure>
        <figure><img src="{gt_uri}" alt="ground truth" loading="lazy"><figcaption>ground truth</figcaption></figure>
      </div>
      <p class="sample-prompt">{esc(prompt)}</p>
      <div class="score-compare">
        <div class="score-box">
          <div class="score-label">Rule-based</div>
          <div class="score-num mono">{rec['score']:.3f}</div>
        </div>
        {judge_block}
      </div>
      {reasoning}
    </div>"""


def task_row(task_name, summary, max_score_for_scale=1.0):
    mean = summary["mean_score"] or 0.0
    pct = round(mean / max_score_for_scale * 1000) / 10
    cat = summary["category"]
    cat_var = CAT_VAR.get(cat, "--accent")
    return f"""
        <div class="bar-row">
          <div class="bar-label">
            <span class="pill" style="background:var({cat_var});">{esc(cat)}</span>
            <code class="mono task-name">{esc(task_name)}</code>
          </div>
          <div class="bar-track">
            <div class="bar-fill" style="width:{pct}%;"></div>
          </div>
          <div class="bar-value mono">{mean:.3f}</div>
        </div>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--judge-scored", default=None,
                     help="path to judge_eval/judge_scored_900.json (per-datapoint judge scores)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    scored, meta, by_id = load(args.scored)
    judge_by_id = load_judge(args.judge_scored)
    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "target_pred_results.html")

    overall = scored["overall"]
    per_task = scored["per_task"]
    records_by_task = {}
    for r in scored["records"]:
        records_by_task.setdefault(r["task_name"], []).append(r)

    id_tasks = sorted([t for t, s in per_task.items() if s["domain"] == "ID"],
                       key=lambda t: per_task[t]["mean_score"] or 0, reverse=True)
    ood_tasks = sorted([t for t, s in per_task.items() if s["domain"] == "OOD"],
                        key=lambda t: per_task[t]["mean_score"] or 0, reverse=True)

    id_mean = per_task and (sum((per_task[t]["mean_score"] or 0) * per_task[t]["n"] for t in id_tasks) /
                             sum(per_task[t]["n"] for t in id_tasks)) if id_tasks else None
    ood_mean = (sum((per_task[t]["mean_score"] or 0) * per_task[t]["n"] for t in ood_tasks) /
                sum(per_task[t]["n"] for t in ood_tasks)) if ood_tasks else None

    bars_id = "".join(task_row(t, per_task[t]) for t in id_tasks)
    bars_ood = "".join(task_row(t, per_task[t]) for t in ood_tasks)

    legend = "".join(
        f'<span class="pill" style="background:var({CAT_VAR[c]});">{esc(c)}</span>'
        for c in CATEGORY_ORDER if any(s["category"] == c for s in per_task.values())
    )

    gallery_sections = []
    for domain_label, tasks in [("In-Domain", id_tasks), ("Out-of-Domain", ood_tasks)]:
        cards = []
        for t in tasks:
            picks = pick_random(records_by_task[t])
            sample_html = "".join(sample_card(i + 1, rec, by_id, judge_by_id) for i, rec in enumerate(picks))
            cards.append(f"""
      <div class="task-block">
        <div class="task-block-head">
          <span class="pill" style="background:var({CAT_VAR.get(per_task[t]['category'], '--accent')});">{esc(per_task[t]['category'])}</span>
          <code class="mono task-name">{esc(t)}</code>
          <span class="task-block-stat mono">mean {per_task[t]['mean_score']:.3f} &middot; n={per_task[t]['n']}</span>
        </div>
        <div class="samples">{sample_html}</div>
      </div>""")
        gallery_sections.append(f"""
    <section class="domain-section">
      <h2>{domain_label}</h2>
      {"".join(cards)}
    </section>""")

    run_meta = scored["run_meta"]

    task_order = id_tasks + ood_tasks
    table_records = [
        {"id": r["id"], "task": r["task_name"], "category": r["category"],
         "domain": r["domain"], "score": r["score"]}
        for r in scored["records"]
    ]
    task_filter_btns = "".join(
        f'<button class="filter-btn" data-task="{esc(t)}">{esc(t)}</button>' for t in task_order
    )
    records_json = json.dumps(table_records)
    cat_var_json = json.dumps(CAT_VAR)

    table_js = """
<script type="application/json" id="records-data">__RECORDS__</script>
<script type="application/json" id="cat-var-data">__CATVAR__</script>
<script>
(function () {
  var records = JSON.parse(document.getElementById('records-data').textContent);
  var catVar = JSON.parse(document.getElementById('cat-var-data').textContent);
  var state = { task: 'all', domain: 'all', sortDir: null };
  var tbody = document.getElementById('data-table-body');
  var countEl = document.getElementById('table-count');
  var scoreHeader = document.getElementById('score-header');

  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  function render() {
    var rows = records.filter(function (r) {
      if (state.task !== 'all' && r.task !== state.task) return false;
      if (state.domain !== 'all' && r.domain !== state.domain) return false;
      return true;
    });
    if (state.sortDir) {
      rows = rows.slice().sort(function (a, b) {
        var av = a.score == null ? -1 : a.score;
        var bv = b.score == null ? -1 : b.score;
        return state.sortDir === 'asc' ? av - bv : bv - av;
      });
    }
    tbody.innerHTML = rows.map(function (r) {
      var scoreStr = r.score == null ? '&mdash;' : r.score.toFixed(3);
      var v = catVar[r.category] || '--accent';
      return '<tr>' +
        '<td class="td-id mono">' + esc(r.id) + '</td>' +
        '<td class="mono">' + esc(r.task) + '</td>' +
        '<td><span class="pill" style="background:var(' + v + ');">' + esc(r.category) + '</span></td>' +
        '<td>' + esc(r.domain) + '</td>' +
        '<td class="td-score mono">' + scoreStr + '</td>' +
        '</tr>';
    }).join('');
    countEl.textContent = 'Showing ' + rows.length + ' of ' + records.length;
  }

  document.querySelectorAll('.filter-btn[data-task]').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var v = btn.getAttribute('data-task');
      state.task = (state.task === v) ? 'all' : v;
      document.querySelectorAll('.filter-btn[data-task]').forEach(function (b) {
        b.classList.toggle('active', b.getAttribute('data-task') === state.task);
      });
      render();
    });
  });

  document.querySelectorAll('.filter-btn[data-domain]').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var v = btn.getAttribute('data-domain');
      state.domain = (state.domain === v) ? 'all' : v;
      document.querySelectorAll('.filter-btn[data-domain]').forEach(function (b) {
        b.classList.toggle('active', b.getAttribute('data-domain') === state.domain);
      });
      render();
    });
  });

  scoreHeader.addEventListener('click', function () {
    state.sortDir = state.sortDir === 'desc' ? 'asc' : (state.sortDir === 'asc' ? null : 'desc');
    scoreHeader.textContent = 'Score' + (state.sortDir === 'asc' ? ' \\u2191' : state.sortDir === 'desc' ? ' \\u2193' : '');
    render();
  });

  render();
})();
</script>
""".replace("__RECORDS__", records_json).replace("__CATVAR__", cat_var_json)

    html = f"""<title>Target-Frame Prediction Results</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {{
  --bg: #F2F4F8;
  --surface: #FFFFFF;
  --surface-alt: #E8ECF3;
  --ink: #12161F;
  --ink-soft: #4A5566;
  --ink-faint: #808DA1;
  --border: #D5DBE6;
  --accent: #2A78D6;
  --accent-soft: #E1EDFB;
  --code-bg: #EAEEF5;
  --cat-abstraction: #6B4FA0;
  --cat-knowledge: #2A5CA6;
  --cat-perception: #A8395A;
  --cat-spatiality: #1C6B5E;
  --cat-transformation: #B06A1A;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #12151C;
    --surface: #1A1F2A;
    --surface-alt: #202634;
    --ink: #E9EDF5;
    --ink-soft: #ABB6C8;
    --ink-faint: #7C879C;
    --border: #2C3444;
    --accent: #5FA3EA;
    --accent-soft: #1B2A3F;
    --code-bg: #232A38;
    --cat-abstraction: #A98FE0;
    --cat-knowledge: #7DA8E8;
    --cat-perception: #E37FA0;
    --cat-spatiality: #5FBFA3;
    --cat-transformation: #E0A356;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #12151C;
  --surface: #1A1F2A;
  --surface-alt: #202634;
  --ink: #E9EDF5;
  --ink-soft: #ABB6C8;
  --ink-faint: #7C879C;
  --border: #2C3444;
  --accent: #5FA3EA;
  --accent-soft: #1B2A3F;
  --code-bg: #232A38;
  --cat-abstraction: #A98FE0;
  --cat-knowledge: #7DA8E8;
  --cat-perception: #E37FA0;
  --cat-spatiality: #5FBFA3;
  --cat-transformation: #E0A356;
}}
* {{ box-sizing: border-box; }}
body {{
  background: var(--bg);
  color: var(--ink);
  font-family: "IBM Plex Sans", system-ui, sans-serif;
  margin: 0;
  padding: 0 0 5rem;
}}
.mono {{ font-family: "IBM Plex Mono", ui-monospace, monospace; font-variant-numeric: tabular-nums; }}
.wrap {{ max-width: 1100px; margin: 0 auto; padding: 0 2rem; }}
header.top {{ border-bottom: 1px solid var(--border); background: var(--surface); padding: 3rem 0 2.25rem; }}
.eyebrow {{
  font-family: "IBM Plex Mono", monospace; font-size: 0.78rem; letter-spacing: 0.09em;
  text-transform: uppercase; color: var(--accent); margin: 0 0 0.9rem;
  display: flex; align-items: center; gap: 0.6rem;
}}
.eyebrow::before {{ content: ""; width: 1.6rem; height: 1px; background: var(--accent); display: inline-block; }}
h1 {{ font-size: clamp(1.8rem, 3.2vw, 2.4rem); line-height: 1.15; margin: 0 0 0.9rem; text-wrap: balance; }}
.lede {{ color: var(--ink-soft); max-width: 68ch; line-height: 1.6; margin: 0 0 1.6rem; }}
.stat-row {{ display: flex; flex-wrap: wrap; gap: 0.9rem; margin-top: 1.4rem; }}
.stat {{
  background: var(--surface-alt); border: 1px solid var(--border); border-radius: 10px;
  padding: 0.9rem 1.1rem; min-width: 9rem;
}}
.stat-label {{ font-size: 0.74rem; color: var(--ink-faint); text-transform: uppercase; letter-spacing: 0.06em; }}
.stat-value {{ font-family: "IBM Plex Mono", monospace; font-size: 1.7rem; font-weight: 600; color: var(--ink); font-variant-numeric: tabular-nums; }}
.stat-value.accent {{ color: var(--accent); }}
code {{ background: var(--code-bg); border-radius: 4px; padding: 0.05em 0.4em; font-size: 0.92em; }}
.pill {{
  display: inline-block; padding: 0.18rem 0.6rem; border-radius: 999px; font-size: 0.72rem;
  font-weight: 600; color: #fff; letter-spacing: 0.01em; white-space: nowrap;
}}
.legend {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1.2rem 0 0; }}
section.chart-section {{ padding: 2.4rem 0; border-bottom: 1px solid var(--border); }}
section.chart-section h2, section.domain-section h2 {{ font-size: 1.3rem; margin: 0 0 1.2rem; }}
.chart-group {{ margin-bottom: 2rem; }}
.chart-group-title {{ font-size: 0.85rem; color: var(--ink-faint); text-transform: uppercase; letter-spacing: 0.06em; margin: 0 0 0.8rem; }}
.bar-row {{
  display: grid; grid-template-columns: 15rem 1fr 4.5rem; align-items: center; gap: 1rem;
  padding: 0.5rem 0.6rem; border-radius: 6px;
}}
.bar-row:hover {{ background: var(--surface-alt); }}
.bar-label {{ display: flex; align-items: center; gap: 0.55rem; min-width: 0; }}
.bar-label .pill {{ flex-shrink: 0; }}
.task-name {{ font-size: 0.86rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.bar-track {{ background: var(--surface-alt); border: 1px solid var(--border); border-radius: 5px; height: 1.15rem; overflow: hidden; }}
.bar-fill {{ background: linear-gradient(90deg, var(--accent) 0%, var(--accent) 100%); height: 100%; border-radius: 4px 0 0 4px; }}
.bar-value {{ text-align: right; font-size: 0.86rem; color: var(--ink-soft); }}
section.domain-section {{ padding: 2.4rem 0; border-bottom: 1px solid var(--border); }}
.task-block {{ margin-bottom: 2.2rem; }}
.task-block-head {{ display: flex; align-items: center; gap: 0.7rem; margin-bottom: 0.9rem; flex-wrap: wrap; }}
.task-block-stat {{ color: var(--ink-faint); font-size: 0.82rem; margin-left: auto; }}
.samples {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 1rem; }}
.sample {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 0.9rem; }}
.sample-head {{ display: flex; justify-content: space-between; align-items: center; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.6rem; }}
.sample-label {{ font-size: 0.76rem; color: var(--ink-faint); text-transform: uppercase; letter-spacing: 0.04em; }}
.sample-score {{ font-size: 1rem; font-weight: 600; color: var(--accent); }}
.sample-imgs {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; margin-bottom: 0.7rem; }}
.sample-imgs figure {{ margin: 0; }}
.sample-imgs img {{ width: 100%; aspect-ratio: 1; object-fit: contain; background: var(--surface-alt); border-radius: 6px; border: 1px solid var(--border); }}
.sample-imgs figcaption {{ text-align: center; font-size: 0.68rem; color: var(--ink-faint); margin-top: 0.3rem; text-transform: uppercase; letter-spacing: 0.03em; }}
.sample-prompt {{ font-size: 0.82rem; color: var(--ink-soft); line-height: 1.5; margin: 0 0 0.7rem; max-height: 6.5em; overflow-y: auto; }}
.badge {{ font-size: 0.66rem; font-weight: 600; padding: 0.15rem 0.55rem; border-radius: 999px; color: #fff; white-space: nowrap; }}
.badge-ok {{ background: #1C6B5E; }}
.badge-warn {{ background: #A9700E; }}
.badge-severe {{ background: #A8395A; }}
.score-compare {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; margin-bottom: 0.6rem; }}
.score-box {{ background: var(--surface-alt); border: 1px solid var(--border); border-radius: 8px; padding: 0.55rem 0.75rem; }}
.score-box.accent-box {{ border-color: var(--accent); }}
.score-label {{ font-size: 0.66rem; color: var(--ink-faint); text-transform: uppercase; letter-spacing: 0.03em; margin-bottom: 0.15rem; }}
.score-num {{ font-size: 1.2rem; font-weight: 600; }}
.crit-breakdown {{ margin-top: 0.45rem; display: flex; flex-direction: column; gap: 0.15rem; }}
.crow {{ display: flex; justify-content: space-between; align-items: center; font-size: 0.7rem; color: var(--ink-soft); }}
.level-tag {{ display: inline-block; font-family: "IBM Plex Mono", monospace; font-size: 0.62rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em; padding: 0.08rem 0.35rem; border-radius: 4px; color: #fff; }}
.level-correct {{ background: #1C6B5E; }}
.level-partial {{ background: #A9700E; }}
.level-wrong {{ background: #A8395A; }}
.judge-reasoning {{ font-size: 0.76rem; color: var(--ink-soft); line-height: 1.5; font-style: italic; margin: 0; }}
.judge-missing {{ font-size: 0.76rem; color: var(--ink-faint); font-style: italic; margin: 0 0 0.6rem; }}
footer.meta {{ padding: 2.4rem 0; }}
footer.meta p {{ color: var(--ink-faint); font-size: 0.85rem; line-height: 1.7; max-width: 74ch; }}
.overflow-x {{ overflow-x: auto; }}
section.table-section {{ padding: 2.4rem 0; border-bottom: 1px solid var(--border); }}
.table-toolbar {{ display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; margin-bottom: 1rem; }}
.table-toolbar .sep {{ width: 1px; height: 1.3rem; background: var(--border); margin: 0 0.3rem; }}
.filter-btn {{
  font-family: "IBM Plex Mono", monospace; font-size: 0.74rem; padding: 0.3rem 0.7rem;
  border-radius: 999px; border: 1px solid var(--border); background: var(--surface);
  color: var(--ink-soft); cursor: pointer; white-space: nowrap;
}}
.filter-btn:hover {{ border-color: var(--accent); color: var(--ink); }}
.filter-btn.active {{ background: var(--accent); border-color: var(--accent); color: #fff; }}
.table-count {{ margin-left: auto; font-size: 0.78rem; color: var(--ink-faint); }}
.table-wrap {{ max-height: 32rem; overflow-y: auto; border: 1px solid var(--border); border-radius: 10px; background: var(--surface); }}
table.data-table {{ width: 100%; border-collapse: collapse; font-size: 0.84rem; }}
table.data-table thead th {{
  position: sticky; top: 0; background: var(--surface-alt); text-align: left;
  padding: 0.6rem 0.9rem; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em;
  color: var(--ink-faint); border-bottom: 1px solid var(--border); z-index: 1;
}}
table.data-table th.sortable {{ cursor: pointer; user-select: none; }}
table.data-table th.sortable:hover {{ color: var(--ink); }}
table.data-table td {{ padding: 0.5rem 0.9rem; border-bottom: 1px solid var(--border); }}
table.data-table tbody tr:hover {{ background: var(--surface-alt); }}
table.data-table tbody tr:last-child td {{ border-bottom: none; }}
.td-score {{ text-align: right; }}
.td-id {{ color: var(--ink-faint); font-size: 0.8rem; }}
</style>

<header class="top">
  <div class="wrap">
    <p class="eyebrow">InternVL-U &middot; VBVR-CustomEval</p>
    <h1>Target-Frame Prediction: Eval Results</h1>
    <p class="lede">Real generations from InternVL-U (base checkpoint, no VBVR fine-tuning) on the 1-step-target-frame-prediction eval split &mdash; {overall['n']} samples, 100 per active task, across {len(per_task)} of the 10 <code>LOCKED_TASKS</code> (<code>glass_refraction</code> has no eval split yet: its DataFactory generator repo is access-restricted, see README). Each sample: one conditioning image in, one generated image out. Every sample is scored two ways: the existing rule-based image-only pipeline (<code>task_specific</code> dimension, same convention as <code>validation/test_harness.py</code>) and a pure Qwen3-VL-30B-fp8 judge scoring the same criteria as wrong/partial/correct (see the companion "Rule vs. Judge Scoring" artifact for the full per-task rubric and prompt text). The gallery below shows {GALLERY_N} freshly-random samples per task (seed {GALLERY_SEED}, not cherry-picked) with both scores side by side.</p>
    <div class="stat-row">
      <div class="stat"><div class="stat-label">Overall mean</div><div class="stat-value accent mono">{overall['mean_score']:.3f}</div></div>
      <div class="stat"><div class="stat-label">Samples scored</div><div class="stat-value mono">{overall['n_scored']}/{overall['n']}</div></div>
      <div class="stat"><div class="stat-label">In-Domain mean</div><div class="stat-value mono">{(id_mean if id_mean is not None else 0):.3f}</div></div>
      <div class="stat"><div class="stat-label">Out-of-Domain mean</div><div class="stat-value mono">{(ood_mean if ood_mean is not None else 0):.3f}</div></div>
    </div>
    <div class="legend">{legend}</div>
  </div>
</header>

<section class="chart-section">
  <div class="wrap">
    <h2>Mean task_specific score per task</h2>
    <div class="overflow-x">
      <div class="chart-group">
        <p class="chart-group-title">In-Domain</p>
        {bars_id}
      </div>
      <div class="chart-group">
        <p class="chart-group-title">Out-of-Domain</p>
        {bars_ood}
      </div>
    </div>
  </div>
</section>

<div class="wrap">
  {"".join(gallery_sections)}
</div>

<section class="table-section">
  <div class="wrap">
    <h2>All {overall['n']} scores</h2>
    <div class="table-toolbar">
      <button class="filter-btn active" data-task="all">All tasks</button>
      {task_filter_btns}
      <span class="sep"></span>
      <button class="filter-btn" data-domain="ID">ID</button>
      <button class="filter-btn" data-domain="OOD">OOD</button>
      <span class="table-count" id="table-count"></span>
    </div>
    <div class="table-wrap">
      <table class="data-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>Task</th>
            <th>Category</th>
            <th>Domain</th>
            <th class="sortable" id="score-header">Score</th>
          </tr>
        </thead>
        <tbody id="data-table-body"></tbody>
      </table>
    </div>
  </div>
</section>
{table_js}

<footer class="meta">
  <div class="wrap">
    <p><b>Model:</b> {esc(run_meta.get('model', 'InternVL-U (base)'))} &middot; <b>checkpoint:</b> <code>{esc(run_meta.get('model_path',''))}</code> &middot; <b>seed:</b> {run_meta.get('seed')} &middot; <b>output size:</b> {run_meta.get('gen_size')}px &middot; <b>input prescale:</b> {run_meta.get('prescale_input')}px &middot; <b>conditioning:</b> {"ViT + VAE (generation_mode='image' default)" if run_meta.get('vae_cond_last_frame') else "VLM hidden states only"}.</p>
    <p>Rule-based scoring: <code>Evaluation/VBVR-CustomEval/evaluators/image_evaluator.py</code>, <code>task_specific_only=True</code>. Judge scoring: <code>Evaluation/VBVR-CustomEval/evaluators/llm_judge_full.py</code> (<code>qwen3-vl-30b-fp8</code>, wrong/partial/correct per criterion, combined with the same weights the rule-based evaluator uses). Gallery picks are {GALLERY_N} random samples per task (Python <code>random.Random({GALLERY_SEED})</code>, deterministic, not cherry-picked). See <code>Evaluation/VBVR-CustomEval/artifacts</code> for the rule-based scoring-rule breakdown and known image-only-mode limitations, and the "Rule vs. Judge Scoring" artifact for the judge's exact per-task prompts.</p>
  </div>
</footer>
"""

    with open(out_path, "w") as f:
        f.write(html)
    print(f"wrote {out_path} {len(html)} chars")


if __name__ == "__main__":
    main()
