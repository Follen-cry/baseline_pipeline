#!/usr/bin/env python3
"""Build the 3-model target-frame-prediction eval-results comparison artifact:
InternVL-U (base) vs. BAGEL-7B-MoT vs. SenseNova-U1-8B-MoT-SFT, all run over
the identical 900-sample 1-step-target-frame-prediction eval split (bagel and
sensenova read InternVL-U's own meta.json as the source of truth for
prompts/ids, so all three see byte-identical inputs and targets -- see
Evaluation/vbvr_baseline_runners/run_target_pred_eval_{bagel,sensenova}.py).

Each model directory under results/vbvr_target_pred_eval/ is expected to
contain scored.json (rule-based, from validation/score_target_pred_eval.py)
and judge_scored.json (from judge_eval/score_all_with_judge.py --run-dir).
InternVL-U's judge file is the historical judge_eval/judge_scored_900.json
path instead (score_all_with_judge.py's default).

    python gen_target_pred_comparison_artifact.py --out comparison.html
"""
import argparse
import base64
import html as htmlmod
import json
import os
import random
from io import BytesIO

from PIL import Image

RESULTS_ROOT = "/scratch/network/ssd2/junlin/ssl_mllm/results/vbvr_target_pred_eval"
JUDGE_EVAL_DIR = "/scratch/network/ssd2/junlin/ssl_mllm/Evaluation/VBVR-CustomEval/judge_eval"

MODELS = [
    {"key": "internvlu", "label": "InternVL-U (base)",
     "run_dir": os.path.join(RESULTS_ROOT, "base_prescale512_vaecond"),
     "judge_path": os.path.join(JUDGE_EVAL_DIR, "judge_scored_900.json"),
     "color": "--model-a"},
    {"key": "bagel", "label": "BAGEL-7B-MoT",
     "run_dir": os.path.join(RESULTS_ROOT, "bagel"),
     "judge_path": os.path.join(RESULTS_ROOT, "bagel", "judge_scored.json"),
     "color": "--model-b"},
    {"key": "sensenova", "label": "SenseNova-U1-8B-MoT-SFT",
     "run_dir": os.path.join(RESULTS_ROOT, "sensenova"),
     "judge_path": os.path.join(RESULTS_ROOT, "sensenova", "judge_scored.json"),
     "color": "--model-c"},
]

GALLERY_SEED = 42
GALLERY_N = 3

CATEGORY_ORDER = ["Abstraction", "Knowledge", "Perception", "Spatiality", "Transformation"]
CAT_VAR = {
    "Abstraction": "--cat-abstraction",
    "Knowledge": "--cat-knowledge",
    "Perception": "--cat-perception",
    "Spatiality": "--cat-spatiality",
    "Transformation": "--cat-transformation",
}


def esc(s):
    return htmlmod.escape(str(s) if s is not None else "", quote=True)


_img_cache = {}


def img_data_uri(path, max_side=300, quality=80):
    if path in _img_cache:
        return _img_cache[path]
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        _img_cache[path] = None
        return None
    w, h = im.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    uri = f"data:image/jpeg;base64,{b64}"
    _img_cache[path] = uri
    return uri


def load_model(m):
    scored = json.load(open(os.path.join(m["run_dir"], "scored.json")))
    meta = json.load(open(os.path.join(m["run_dir"], "meta.json")))
    by_id = {r["id"]: r for r in meta["results"]}
    rule_by_id = {r["id"]: r for r in scored["records"]}
    judge = json.load(open(m["judge_path"]))["results"] if os.path.exists(m["judge_path"]) else {}
    return {"scored": scored, "meta": meta, "by_id": by_id, "rule_by_id": rule_by_id, "judge_by_id": judge}


def fnum(x, d=3):
    return f"{x:.{d}f}" if isinstance(x, (int, float)) else "&mdash;"


def bar_group(task_name, category, per_model_vals, max_val=1.0):
    cat_var = CAT_VAR.get(category, "--accent")
    bars = ""
    for m, v in zip(MODELS, per_model_vals):
        pct = round((v or 0) / max_val * 100, 1)
        bars += f"""
          <div class="mbar-row">
            <span class="mbar-name mono" style="color:var({m['color']});">{esc(m['label'])}</span>
            <div class="mbar-track"><div class="mbar-fill" style="width:{pct}%;background:var({m['color']});"></div></div>
            <span class="mbar-val mono">{fnum(v)}</span>
          </div>"""
    return f"""
    <div class="task-group">
      <div class="task-group-head">
        <span class="pill" style="background:var({cat_var});">{esc(category)}</span>
        <code class="mono task-name">{esc(task_name)}</code>
      </div>
      {bars}
    </div>"""


def sample_card(task_name, sample_id, data_by_model):
    row0 = data_by_model[MODELS[0]["key"]]["by_id"][sample_id]
    input_uri = img_data_uri(row0["input_image"])
    gt_uri = img_data_uri(row0["target_image"])
    prompt = row0["prompt"].split("Task: ", 1)[-1] if "Task: " in row0["prompt"] else row0["prompt"]

    model_cols = ""
    for m in MODELS:
        d = data_by_model[m["key"]]
        row = d["by_id"][sample_id]
        gen_uri = img_data_uri(row["generated_image"])
        rule_rec = d["rule_by_id"].get(sample_id)
        judge_rec = d["judge_by_id"].get(sample_id)
        rule_score = rule_rec["score"] if rule_rec else None
        judge_score = judge_rec["judge_score"] if judge_rec else None
        reasoning = esc(judge_rec.get("judge_reasoning") or "") if judge_rec else ""
        model_cols += f"""
        <div class="mcol">
          <div class="mcol-head mono" style="color:var({m['color']});">{esc(m['label'])}</div>
          <figure><img src="{gen_uri}" alt="{esc(m['label'])} output" loading="lazy"></figure>
          <div class="mcol-scores">
            <span class="mono">rule {fnum(rule_score)}</span>
            <span class="mono">judge {fnum(judge_score)}</span>
          </div>
          <p class="mcol-reasoning">{reasoning}</p>
        </div>"""

    return f"""
    <div class="sample">
      <div class="sample-head">
        <span class="sample-label mono">{esc(sample_id)}</span>
      </div>
      <p class="sample-prompt">{esc(prompt)}</p>
      <div class="sample-grid">
        <div class="mcol ref-col">
          <div class="mcol-head mono">input &rarr; target</div>
          <div class="ref-imgs">
            <figure><img src="{input_uri}" alt="input" loading="lazy"><figcaption>input</figcaption></figure>
            <figure><img src="{gt_uri}" alt="target" loading="lazy"><figcaption>target</figcaption></figure>
          </div>
        </div>
        {model_cols}
      </div>
    </div>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "target_pred_comparison.html"))
    args = ap.parse_args()

    data = {m["key"]: load_model(m) for m in MODELS}

    per_task = {m["key"]: data[m["key"]]["scored"]["per_task"] for m in MODELS}
    task_names = list(per_task[MODELS[0]["key"]].keys())
    task_meta = {t: per_task[MODELS[0]["key"]][t] for t in task_names}
    id_tasks = sorted([t for t in task_names if task_meta[t]["domain"] == "ID"],
                       key=lambda t: per_task["sensenova"][t]["mean_score"] or 0, reverse=True)
    ood_tasks = sorted([t for t in task_names if task_meta[t]["domain"] == "OOD"],
                        key=lambda t: per_task["sensenova"][t]["mean_score"] or 0, reverse=True)

    def judge_mean_for(mkey, task):
        recs = [r for r in data[mkey]["judge_by_id"].values()
                if r.get("judge_score") is not None and
                data[mkey]["rule_by_id"].get(r["id"], {}).get("task_name") == task]
        return sum(r["judge_score"] for r in recs) / len(recs) if recs else None

    judge_per_task = {m["key"]: {t: judge_mean_for(m["key"], t) for t in task_names} for m in MODELS}

    rule_bars_id = "".join(bar_group(t, task_meta[t]["category"],
                                      [per_task[m["key"]][t]["mean_score"] for m in MODELS]) for t in id_tasks)
    rule_bars_ood = "".join(bar_group(t, task_meta[t]["category"],
                                       [per_task[m["key"]][t]["mean_score"] for m in MODELS]) for t in ood_tasks)
    judge_bars_id = "".join(bar_group(t, task_meta[t]["category"],
                                       [judge_per_task[m["key"]][t] for m in MODELS]) for t in id_tasks)
    judge_bars_ood = "".join(bar_group(t, task_meta[t]["category"],
                                        [judge_per_task[m["key"]][t] for m in MODELS]) for t in ood_tasks)

    overall_stats = ""
    for m in MODELS:
        d = data[m["key"]]
        rule_overall = d["scored"]["overall"]
        judge_scores = [r["judge_score"] for r in d["judge_by_id"].values() if r.get("judge_score") is not None]
        judge_mean = sum(judge_scores) / len(judge_scores) if judge_scores else None
        overall_stats += f"""
        <div class="model-stat" style="border-color:var({m['color']});">
          <div class="model-stat-name mono" style="color:var({m['color']});">{esc(m['label'])}</div>
          <div class="model-stat-row"><span>Judge mean</span><span class="mono accent-num">{fnum(judge_mean)}</span></div>
          <div class="model-stat-row"><span>Rule mean</span><span class="mono">{fnum(rule_overall['mean_score'])}</span></div>
          <div class="model-stat-row"><span>Scored</span><span class="mono">{rule_overall['n_scored']}/{rule_overall['n']}</span></div>
        </div>"""

    legend = "".join(
        f'<span class="pill" style="background:var({CAT_VAR[c]});">{esc(c)}</span>'
        for c in CATEGORY_ORDER if any(task_meta[t]["category"] == c for t in task_names)
    )

    gallery_sections = []
    rnd = random.Random(GALLERY_SEED)
    for domain_label, tasks in [("In-Domain", id_tasks), ("Out-of-Domain", ood_tasks)]:
        blocks = []
        for t in tasks:
            all_ids = sorted([r["id"] for r in data[MODELS[0]["key"]]["scored"]["records"] if r["task_name"] == t])
            picks = rnd.sample(all_ids, min(GALLERY_N, len(all_ids)))
            cards = "".join(sample_card(t, sid, data) for sid in picks)
            blocks.append(f"""
      <div class="task-block">
        <div class="task-block-head">
          <span class="pill" style="background:var({CAT_VAR.get(task_meta[t]['category'], '--accent')});">{esc(task_meta[t]['category'])}</span>
          <code class="mono task-name">{esc(t)}</code>
        </div>
        {cards}
      </div>""")
        gallery_sections.append(f"""
    <section class="domain-section">
      <h2>{domain_label}</h2>
      {"".join(blocks)}
    </section>""")

    # full data table (rule + judge per record per model)
    table_records = []
    for m in MODELS:
        d = data[m["key"]]
        for r in d["scored"]["records"]:
            jr = d["judge_by_id"].get(r["id"])
            table_records.append({
                "id": r["id"], "task": r["task_name"], "category": r["category"],
                "domain": r["domain"], "model": m["label"],
                "rule": r["score"], "judge": jr["judge_score"] if jr else None,
            })
    task_order = id_tasks + ood_tasks
    task_filter_btns = "".join(
        f'<button class="filter-btn" data-task="{esc(t)}">{esc(t)}</button>' for t in task_order
    )
    model_filter_btns = "".join(
        f'<button class="filter-btn" data-model="{esc(m["label"])}" style="--fbc:var({m["color"]});">{esc(m["label"])}</button>'
        for m in MODELS
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
  var state = { task: 'all', domain: 'all', model: 'all', sortKey: null, sortDir: null };
  var tbody = document.getElementById('data-table-body');
  var countEl = document.getElementById('table-count');

  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  function render() {
    var rows = records.filter(function (r) {
      if (state.task !== 'all' && r.task !== state.task) return false;
      if (state.domain !== 'all' && r.domain !== state.domain) return false;
      if (state.model !== 'all' && r.model !== state.model) return false;
      return true;
    });
    if (state.sortKey) {
      rows = rows.slice().sort(function (a, b) {
        var av = a[state.sortKey] == null ? -1 : a[state.sortKey];
        var bv = b[state.sortKey] == null ? -1 : b[state.sortKey];
        return state.sortDir === 'asc' ? av - bv : bv - av;
      });
    }
    tbody.innerHTML = rows.map(function (r) {
      var ruleStr = r.rule == null ? '&mdash;' : r.rule.toFixed(3);
      var judgeStr = r.judge == null ? '&mdash;' : r.judge.toFixed(3);
      var v = catVar[r.category] || '--accent';
      return '<tr>' +
        '<td class="td-id mono">' + esc(r.id) + '</td>' +
        '<td class="mono">' + esc(r.task) + '</td>' +
        '<td><span class="pill" style="background:var(' + v + ');">' + esc(r.category) + '</span></td>' +
        '<td>' + esc(r.domain) + '</td>' +
        '<td class="mono">' + esc(r.model) + '</td>' +
        '<td class="td-score mono">' + ruleStr + '</td>' +
        '<td class="td-score mono">' + judgeStr + '</td>' +
        '</tr>';
    }).join('');
    countEl.textContent = 'Showing ' + rows.length + ' of ' + records.length;
  }

  function wireGroup(selector, key) {
    document.querySelectorAll(selector).forEach(function (btn) {
      btn.addEventListener('click', function () {
        var v = btn.getAttribute('data-' + key);
        state[key] = (state[key] === v) ? 'all' : v;
        document.querySelectorAll(selector).forEach(function (b) {
          b.classList.toggle('active', b.getAttribute('data-' + key) === state[key]);
        });
        render();
      });
    });
  }
  wireGroup('.filter-btn[data-task]', 'task');
  wireGroup('.filter-btn[data-domain]', 'domain');
  wireGroup('.filter-btn[data-model]', 'model');

  document.querySelectorAll('th.sortable').forEach(function (th) {
    th.addEventListener('click', function () {
      var key = th.getAttribute('data-key');
      if (state.sortKey === key) {
        state.sortDir = state.sortDir === 'desc' ? 'asc' : (state.sortDir === 'asc' ? null : 'desc');
        if (!state.sortDir) state.sortKey = null;
      } else {
        state.sortKey = key; state.sortDir = 'desc';
      }
      document.querySelectorAll('th.sortable').forEach(function (h) {
        h.textContent = h.getAttribute('data-label');
      });
      if (state.sortKey) {
        th.textContent = th.getAttribute('data-label') + (state.sortDir === 'asc' ? ' \\u2191' : ' \\u2193');
      }
      render();
    });
  });

  render();
})();
</script>
""".replace("__RECORDS__", records_json).replace("__CATVAR__", cat_var_json)

    html = f"""<title>Target-Frame Prediction: Model Comparison</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {{
  --bg: #F2F4F8; --surface: #FFFFFF; --surface-alt: #E8ECF3; --ink: #12161F;
  --ink-soft: #4A5566; --ink-faint: #808DA1; --border: #D5DBE6;
  --accent: #2A78D6; --accent-soft: #E1EDFB; --code-bg: #EAEEF5;
  --cat-abstraction: #6B4FA0; --cat-knowledge: #2A5CA6; --cat-perception: #A8395A;
  --cat-spatiality: #1C6B5E; --cat-transformation: #B06A1A;
  --model-a: #2A78D6; --model-b: #C1662F; --model-c: #1C8A6E;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #12151C; --surface: #1A1F2A; --surface-alt: #202634; --ink: #E9EDF5;
    --ink-soft: #ABB6C8; --ink-faint: #7C879C; --border: #2C3444;
    --accent: #5FA3EA; --accent-soft: #1B2A3F; --code-bg: #232A38;
    --cat-abstraction: #A98FE0; --cat-knowledge: #7DA8E8; --cat-perception: #E37FA0;
    --cat-spatiality: #5FBFA3; --cat-transformation: #E0A356;
    --model-a: #5FA3EA; --model-b: #E39A66; --model-c: #5FCBA9;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #12151C; --surface: #1A1F2A; --surface-alt: #202634; --ink: #E9EDF5;
  --ink-soft: #ABB6C8; --ink-faint: #7C879C; --border: #2C3444;
  --accent: #5FA3EA; --accent-soft: #1B2A3F; --code-bg: #232A38;
  --cat-abstraction: #A98FE0; --cat-knowledge: #7DA8E8; --cat-perception: #E37FA0;
  --cat-spatiality: #5FBFA3; --cat-transformation: #E0A356;
  --model-a: #5FA3EA; --model-b: #E39A66; --model-c: #5FCBA9;
}}
* {{ box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--ink); font-family: "IBM Plex Sans", system-ui, sans-serif; margin: 0; padding: 0 0 5rem; }}
.mono {{ font-family: "IBM Plex Mono", ui-monospace, monospace; font-variant-numeric: tabular-nums; }}
.wrap {{ max-width: 1180px; margin: 0 auto; padding: 0 2rem; }}
header.top {{ border-bottom: 1px solid var(--border); background: var(--surface); padding: 3rem 0 2.25rem; }}
.eyebrow {{ font-family: "IBM Plex Mono", monospace; font-size: 0.78rem; letter-spacing: 0.09em; text-transform: uppercase; color: var(--accent); margin: 0 0 0.9rem; display: flex; align-items: center; gap: 0.6rem; }}
.eyebrow::before {{ content: ""; width: 1.6rem; height: 1px; background: var(--accent); display: inline-block; }}
h1 {{ font-size: clamp(1.8rem, 3.2vw, 2.4rem); line-height: 1.15; margin: 0 0 0.9rem; text-wrap: balance; }}
.lede {{ color: var(--ink-soft); max-width: 74ch; line-height: 1.6; margin: 0 0 1.6rem; }}
.model-stats {{ display: flex; flex-wrap: wrap; gap: 1rem; margin-top: 1.2rem; }}
.model-stat {{ background: var(--surface-alt); border: 1px solid var(--border); border-left: 3px solid; border-radius: 8px; padding: 0.9rem 1.1rem; min-width: 15rem; flex: 1; }}
.model-stat-name {{ font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem; }}
.model-stat-row {{ display: flex; justify-content: space-between; font-size: 0.84rem; color: var(--ink-soft); padding: 0.15rem 0; }}
.accent-num {{ color: var(--ink); font-weight: 600; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1.2rem 0 0; }}
.pill {{ display: inline-block; padding: 0.18rem 0.6rem; border-radius: 999px; font-size: 0.72rem; font-weight: 600; color: #fff; letter-spacing: 0.01em; white-space: nowrap; }}
code {{ background: var(--code-bg); border-radius: 4px; padding: 0.05em 0.4em; font-size: 0.92em; }}
section.chart-section {{ padding: 2.4rem 0; border-bottom: 1px solid var(--border); }}
section.chart-section h2, section.domain-section h2 {{ font-size: 1.3rem; margin: 0 0 0.4rem; }}
.chart-sub {{ color: var(--ink-faint); font-size: 0.86rem; margin: 0 0 1.4rem; }}
.chart-group {{ margin-bottom: 1.8rem; }}
.chart-group-title {{ font-size: 0.85rem; color: var(--ink-faint); text-transform: uppercase; letter-spacing: 0.06em; margin: 0 0 0.9rem; }}
.task-group {{ border: 1px solid var(--border); border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.7rem; background: var(--surface); }}
.task-group-head {{ display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.6rem; }}
.task-name {{ font-size: 0.86rem; }}
.mbar-row {{ display: grid; grid-template-columns: 12rem 1fr 3.5rem; align-items: center; gap: 0.7rem; padding: 0.15rem 0; }}
.mbar-name {{ font-size: 0.78rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.mbar-track {{ background: var(--surface-alt); border: 1px solid var(--border); border-radius: 5px; height: 0.9rem; overflow: hidden; }}
.mbar-fill {{ height: 100%; border-radius: 3px 0 0 3px; }}
.mbar-val {{ text-align: right; font-size: 0.78rem; color: var(--ink-soft); }}
section.domain-section {{ padding: 2.4rem 0; border-bottom: 1px solid var(--border); }}
.task-block {{ margin-bottom: 2.2rem; }}
.task-block-head {{ display: flex; align-items: center; gap: 0.7rem; margin-bottom: 1rem; flex-wrap: wrap; }}
.sample {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 0.9rem; margin-bottom: 1rem; }}
.sample-head {{ margin-bottom: 0.4rem; }}
.sample-label {{ font-size: 0.76rem; color: var(--ink-faint); text-transform: uppercase; letter-spacing: 0.04em; }}
.sample-prompt {{ font-size: 0.82rem; color: var(--ink-soft); line-height: 1.5; margin: 0 0 0.8rem; }}
.sample-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.7rem; }}
.mcol {{ background: var(--surface-alt); border-radius: 8px; padding: 0.6rem; }}
.mcol-head {{ font-size: 0.72rem; font-weight: 600; margin-bottom: 0.4rem; text-align: center; }}
.mcol figure {{ margin: 0; }}
.mcol img {{ width: 100%; aspect-ratio: 1; object-fit: contain; background: var(--surface); border-radius: 6px; border: 1px solid var(--border); }}
.ref-imgs {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.3rem; }}
.ref-imgs figcaption {{ text-align: center; font-size: 0.6rem; color: var(--ink-faint); margin-top: 0.2rem; text-transform: uppercase; }}
.mcol-scores {{ display: flex; justify-content: space-between; font-size: 0.72rem; color: var(--ink-soft); margin-top: 0.4rem; }}
.mcol-reasoning {{ font-size: 0.68rem; color: var(--ink-faint); font-style: italic; line-height: 1.4; margin: 0.4rem 0 0; max-height: 4.5em; overflow-y: auto; }}
footer.meta {{ padding: 2.4rem 0; }}
footer.meta p {{ color: var(--ink-faint); font-size: 0.85rem; line-height: 1.7; max-width: 80ch; }}
.overflow-x {{ overflow-x: auto; }}
section.table-section {{ padding: 2.4rem 0; }}
.table-toolbar {{ display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; margin-bottom: 1rem; }}
.table-toolbar .sep {{ width: 1px; height: 1.3rem; background: var(--border); margin: 0 0.3rem; }}
.filter-btn {{ font-family: "IBM Plex Mono", monospace; font-size: 0.74rem; padding: 0.3rem 0.7rem; border-radius: 999px; border: 1px solid var(--border); background: var(--surface); color: var(--ink-soft); cursor: pointer; white-space: nowrap; }}
.filter-btn:hover {{ border-color: var(--accent); color: var(--ink); }}
.filter-btn.active {{ background: var(--accent); border-color: var(--accent); color: #fff; }}
.filter-btn[data-model].active {{ background: var(--fbc, var(--accent)); border-color: var(--fbc, var(--accent)); }}
.table-count {{ margin-left: auto; font-size: 0.78rem; color: var(--ink-faint); }}
.table-wrap {{ max-height: 32rem; overflow-y: auto; border: 1px solid var(--border); border-radius: 10px; background: var(--surface); }}
table.data-table {{ width: 100%; border-collapse: collapse; font-size: 0.84rem; }}
table.data-table thead th {{ position: sticky; top: 0; background: var(--surface-alt); text-align: left; padding: 0.6rem 0.9rem; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--ink-faint); border-bottom: 1px solid var(--border); z-index: 1; }}
table.data-table th.sortable {{ cursor: pointer; user-select: none; }}
table.data-table th.sortable:hover {{ color: var(--ink); }}
table.data-table td {{ padding: 0.5rem 0.9rem; border-bottom: 1px solid var(--border); }}
table.data-table tbody tr:hover {{ background: var(--surface-alt); }}
table.data-table tbody tr:last-child td {{ border-bottom: none; }}
.td-score {{ text-align: right; }}
.td-id {{ color: var(--ink-faint); font-size: 0.8rem; }}
@media (max-width: 900px) {{ .sample-grid {{ grid-template-columns: repeat(2, 1fr); }} }}
</style>

<header class="top">
  <div class="wrap">
    <p class="eyebrow">VBVR-CustomEval &middot; 3-Model Comparison</p>
    <h1>Target-Frame Prediction: Eval Results</h1>
    <p class="lede">Three base checkpoints (no VBVR fine-tuning) run over the identical 900-sample 1-step-target-frame-prediction eval split &mdash; 100 samples per task, 9 of the 10 <code>LOCKED_TASKS</code> (<code>glass_refraction</code> has no eval split yet). All three models see byte-identical conditioning images and prompts, read from the same source rows. Every sample is scored two ways: the rule-based image-only pipeline (<code>task_specific</code> dimension) and a pure Qwen3-VL-30B-fp8 judge (wrong/partial/correct per criterion). Charts below use the judge score as primary (rule-based shown as a secondary check); the gallery shows {GALLERY_N} random samples per task with all three models' outputs side by side.</p>
    <div class="model-stats">{overall_stats}</div>
    <div class="legend">{legend}</div>
  </div>
</header>

<section class="chart-section">
  <div class="wrap">
    <h2>Judge score per task, by model</h2>
    <p class="chart-sub">Primary metric &mdash; Qwen3-VL-30B-fp8 judge, same weighted criteria as the rule-based evaluator.</p>
    <div class="overflow-x">
      <div class="chart-group">
        <p class="chart-group-title">In-Domain</p>
        {judge_bars_id}
      </div>
      <div class="chart-group">
        <p class="chart-group-title">Out-of-Domain</p>
        {judge_bars_ood}
      </div>
    </div>
  </div>
</section>

<section class="chart-section">
  <div class="wrap">
    <h2>Rule-based score per task, by model</h2>
    <p class="chart-sub">Secondary / sanity check &mdash; image-only pipeline, <code>task_specific_only=True</code>.</p>
    <div class="overflow-x">
      <div class="chart-group">
        <p class="chart-group-title">In-Domain</p>
        {rule_bars_id}
      </div>
      <div class="chart-group">
        <p class="chart-group-title">Out-of-Domain</p>
        {rule_bars_ood}
      </div>
    </div>
  </div>
</section>

<div class="wrap">
  {"".join(gallery_sections)}
</div>

<section class="table-section">
  <div class="wrap">
    <h2>All {len(table_records)} scores ({len(MODELS)} models &times; 900 samples)</h2>
    <div class="table-toolbar">
      <button class="filter-btn active" data-task="all">All tasks</button>
      {task_filter_btns}
      <span class="sep"></span>
      <button class="filter-btn" data-domain="ID">ID</button>
      <button class="filter-btn" data-domain="OOD">OOD</button>
      <span class="sep"></span>
      {model_filter_btns}
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
            <th>Model</th>
            <th class="sortable" data-key="rule" data-label="Rule">Rule</th>
            <th class="sortable" data-key="judge" data-label="Judge">Judge</th>
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
    <p><b>Models:</b> InternVL-U base (<code>{esc(data['internvlu']['scored']['run_meta'].get('model_path',''))}</code>, ViT+VAE conditioning) &middot; BAGEL-7B-MoT (<code>{esc(data['bagel']['scored']['run_meta'].get('model_path',''))}</code>) &middot; SenseNova-U1-8B-MoT-SFT (<code>{esc(data['sensenova']['scored']['run_meta'].get('model_path',''))}</code>). All three: seed 42, single conditioning image in / one generated image out, run over BAGEL's and SenseNova's respective run scripts pointed at InternVL-U's meta.json as the shared source of rows.</p>
    <p>Rule-based scoring: <code>Evaluation/VBVR-CustomEval/evaluators/image_evaluator.py</code>, <code>task_specific_only=True</code>. Judge scoring: <code>Evaluation/VBVR-CustomEval/evaluators/llm_judge_full.py</code> (<code>qwen3-vl-30b-fp8</code>), run per-model via <code>judge_eval/score_all_with_judge.py --run-dir &lt;model_dir&gt;</code>. Gallery picks: {GALLERY_N} random samples per task (Python <code>random.Random({GALLERY_SEED})</code>, deterministic, not cherry-picked), identical sample ids across all three models since inputs/targets are shared.</p>
  </div>
</footer>
"""

    with open(args.out, "w") as f:
        f.write(html)
    print(f"wrote {args.out} ({len(html)/1e6:.2f} MB, {len(html)} chars)")


if __name__ == "__main__":
    main()
