#!/usr/bin/env python3
"""Build the 3-model comparison artifact for the "does starting from a
bigger next-frame-pretrained checkpoint help" experiment: all three models
were fine-tuned with IDENTICAL settings (LoRA rank 32, lr 1e-5, batch 16,
no-ce, all conditioning images through the ViT) on the SAME new 4,500-row
9-task target-prediction training set (500/task: 5 ID tasks subsampled from
their existing 2,000-row pool, 4 OOD tasks freshly generated since they'd
never had a train split before) -- the only difference is the starting
checkpoint:
  - "frombase": InternVL-U base checkpoint -> SFT on the 4,500 rows
  - "from100k": the checkpoint pretrained on 100,000 next-frame rows (10
    in-domain tasks) -> further SFT on the same 4,500 rows
  - "from140k": the checkpoint pretrained on 140,000 next-frame rows (the
    same 100k plus 40,000 freshly-generated rows for the 4 tasks that used
    to be held out as OOD, now folded in as ordinary training data) ->
    further SFT on the same 4,500 rows
All three are then evaluated on the identical 900-sample (9 task x 100)
eval split used everywhere else in this project. Adapted from
gen_comparison_report.py's 4-model version -- same visual language, trimmed
to 10 shared examples/task (was 20/4-models); "frombase"/"from100k" were the
original 2-model version, "from140k" added in a follow-up session.

Each model directory under results/vbvr_target_pred_eval/ is expected to
contain scored.json (rule-based, from validation/score_target_pred_eval.py)
and judge_scored.json (from judge_eval/score_all_with_judge.py --run-dir).

    python gen_9task_sft_comparison.py
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

RUN = "/scratch/network/ssd2/junlin/ssl_mllm/results/vbvr_target_pred_eval"
OUT_PATH = "/scratch/network/ssd2/junlin/ssl_mllm/Evaluation/VBVR-CustomEval/artifacts/9task_sft_comparison.html"

SEED = 92820260828
N_SHARED_PER_TASK = 10
# 2026-08-30: maze and 2d_geometric_transformation each get 10 additional
# shared examples on top of the original N_SHARED_PER_TASK, drawn from an
# independent per-task RNG stream (see main()) so the original 10 picks for
# every task -- including these two -- stay bit-identical to what's already
# published; only new rows are appended.
EXTRA_SAMPLES = {"maze": 10, "2d_geometric_transformation": 10}

MODELS = [
    ("frombase", "InternVL-U (base &rarr; 4,500-row SFT)", f"{RUN}/frombase_9task4500",
     f"{RUN}/frombase_9task4500/judge_scored.json"),
    ("from100k", "InternVL-U (100k-ckpt &rarr; 4,500-row SFT)", f"{RUN}/from100k_9task4500",
     f"{RUN}/from100k_9task4500/judge_scored.json"),
    ("from140k", "InternVL-U (140k-ckpt &rarr; 4,500-row SFT)", f"{RUN}/from140k_9task4500",
     f"{RUN}/from140k_9task4500/judge_scored.json"),
]

# 2026-08-31: second judge, Qwen3.8-27B run locally via `transformers`
# (evaluators/llm_judge_transformers.py) instead of a vLLM server -- vLLM
# itself cannot currently serve this model on this cluster (see
# Evaluation/GenEditEvalKit/serve_qwen38_27b*.sh for why: the community
# AWQ-INT4 quant crashes vLLM's Marlin repack kernel; the official bf16
# checkpoint's tensor-parallel path needs CUDA-13-only compiled extensions
# this cluster's driver can't run). Scored only the SAME shared_picks
# subset this report already displays (110 ids/model, 330 total judge
# calls) -- a real, measured single call is 40-70s under the fastest
# working config found (enable_thinking=False; the model's own "low"
# reasoning-effort setting is not actually short and blew through
# max_tokens before reaching the JSON), so the full 900x3 corpus would be
# a ~35-40 hour job. See score_shared_with_qwen38.py. Because this is a
# subset, not the full corpus, its stats are kept in their own section/
# columns rather than merged into the existing full-900 rule/judge
# numbers above -- those must stay exactly what they were.
QWEN38_JUDGE_FILES = {key: f"{run_dir}/judge_scored_qwen38.json" for key, _, run_dir, _ in MODELS}

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
# NOT list(TASK_CRITERIA.keys()) -- that now has 10 entries (shape_outline_then_move
# was added to TASK_CRITERIA/the eval jsonl after the historical 900-sample runs;
# it's not one of the 9 active LOCKED_TASKS this comparison covers). Use
# CATEGORY_OF's key order instead, which is exactly the 9 intended tasks.
TASK_ORDER = list(CATEGORY_OF.keys())

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


def esc(s):
    return htmlmod.escape(s or "", quote=True)


def img_data_uri(path, max_side=220, quality=78):
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
    b64 = base64_encode(buf.getvalue())
    return f"data:image/jpeg;base64,{b64}"


def base64_encode(data):
    import base64
    return base64.b64encode(data).decode("ascii")


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
        qwen38_path = QWEN38_JUDGE_FILES[key]
        qwen38 = json.load(open(qwen38_path))["results"] if os.path.exists(qwen38_path) else {}
        model_data[key] = {"label": label, "rule": rule_by_id, "judge": judge, "meta": meta_by_id,
                            "qwen38": qwen38}

    base_meta = model_data[MODELS[0][0]]["meta"]
    by_task_ids = {}
    for rid, row in base_meta.items():
        by_task_ids.setdefault(row["task_name"], []).append(rid)

    rng = random.Random(SEED)
    shared_picks = {t: rng.sample(ids, min(N_SHARED_PER_TASK, len(ids))) for t, ids in by_task_ids.items()}

    # Append EXTRA_SAMPLES additional picks for specific tasks, via an
    # independent RNG stream per task -- keeps every task's original
    # N_SHARED_PER_TASK picks (computed above, untouched) bit-identical to
    # what's already published; only new, non-overlapping rows are appended.
    for t, n_extra in EXTRA_SAMPLES.items():
        ids = by_task_ids.get(t, [])
        already = set(shared_picks.get(t, []))
        remaining = [i for i in ids if i not in already]
        extra_rng = random.Random(f"{SEED}:extra:{t}")
        extra_picks = extra_rng.sample(remaining, min(n_extra, len(remaining)))
        shared_picks[t] = shared_picks.get(t, []) + extra_picks

    # ---- stats: per model x task, and overall ----
    task_stats = {}
    overall_stats = {}
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

    # ---- qwen38-judge stats, SHARED SUBSET ONLY (110 ids/model, not the
    # full per-task pool the block above uses) -- kept in separate dicts so
    # task_stats/overall_stats above (which drive the pre-existing summary
    # table, mini-bars, and readout cards) are untouched. rule_mean/judge_mean
    # here are also recomputed on just the subset, specifically so they're
    # apples-to-apples comparable with qwen38_mean in the same table/row --
    # do not read these as replacements for the full-900 numbers above.
    subset_task_stats = {}
    subset_overall_stats = {}
    for key, _, _, _ in MODELS:
        d = model_data[key]
        all_r, all_j, all_q = [], [], []
        for t in TASK_ORDER:
            rs, js, qs = [], [], []
            for rid in shared_picks.get(t, []):
                r = d["rule"].get(rid)
                j = d["judge"].get(rid)
                q = d["qwen38"].get(rid)
                if r and r["score"] is not None and j and j.get("judge_score") is not None:
                    rs.append(r["score"]); js.append(j["judge_score"])
                if q and q.get("judge_score") is not None:
                    qs.append(q["judge_score"])
            subset_task_stats.setdefault(t, {})[key] = {
                "rule_mean": sum(rs) / len(rs) if rs else None,
                "judge_mean": sum(js) / len(js) if js else None,
                "qwen38_mean": sum(qs) / len(qs) if qs else None,
                "n": len(rs), "n_qwen38": len(qs),
            }
            all_r += rs; all_j += js; all_q += qs
        subset_overall_stats[key] = {
            "rule_mean": sum(all_r) / len(all_r) if all_r else None,
            "judge_mean": sum(all_j) / len(all_j) if all_j else None,
            "qwen38_mean": sum(all_q) / len(all_q) if all_q else None,
            "n": len(all_r), "n_qwen38": len(all_q),
        }

    def readout_cards():
        cards = []
        for key, label, _, _ in MODELS:
            s = overall_stats[key]
            sq = subset_overall_stats[key]
            rho_str = f"{s['rho']:.3f}" if s["rho"] is not None else "n/a"
            qwen38_row = ""
            if sq["qwen38_mean"] is not None:
                qwen38_row = (
                    f'<div><span class="mcs-label judge2-ink">qwen3.8</span>'
                    f'<span class="mono">{sq["qwen38_mean"]:.3f}</span></div>'
                    f'<div class="mcs-subset-note">subset n={sq["n_qwen38"]}</div>'
                )
            cards.append(f"""
            <div class="model-card model-{key}">
              <div class="model-card-head"><span class="model-dot dot-{key}"></span>{label}</div>
              <div class="model-card-stats">
                <div><span class="mcs-label rule-ink">rule</span><span class="mono">{s['rule_mean']:.3f}</span></div>
                <div><span class="mcs-label judge-ink">judge</span><span class="mono">{s['judge_mean']:.3f}</span></div>
                <div><span class="mcs-label">agree &rho;</span><span class="mono">{rho_str}</span></div>
                {qwen38_row}
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
        headers = "".join(f'<th class="num">{label}<br><span class="th-sub">rule / judge</span></th>' for _, label, _, _ in MODELS)
        return f"""
        <div class="table-wrap">
          <table class="summary-table">
            <thead><tr><th>Category</th><th>Task</th><th>Domain</th>{headers}</tr></thead>
            <tbody>{"".join(rows)}</tbody>
          </table>
        </div>"""

    def subset_comparison_table():
        """3-way comparison (rule / original qwen3-vl-30b-fp8 judge / new
        Qwen3.8-27B judge) on ONLY the 110-id shared subset per model --
        deliberately a separate table from summary_table() above, which
        stays full-900. Every number in this table is computed over the
        identical id set, so it's a fair apples-to-apples read, unlike
        naively eyeballing this against the full-900 table."""
        any_qwen38 = any(subset_overall_stats[key]["qwen38_mean"] is not None for key, _, _, _ in MODELS)
        if not any_qwen38:
            return ""
        rows = []
        for t in TASK_ORDER:
            cells = []
            for key, _, _, _ in MODELS:
                s = subset_task_stats[t][key]
                rv = f"{s['rule_mean']:.2f}" if s["rule_mean"] is not None else "&mdash;"
                jv = f"{s['judge_mean']:.2f}" if s["judge_mean"] is not None else "&mdash;"
                qv = f"{s['qwen38_mean']:.2f}" if s["qwen38_mean"] is not None else "&mdash;"
                cells.append(f'<td class="num mono">{rv}<span class="sep">/</span>{jv}<span class="sep">/</span>'
                             f'<span class="judge2-ink">{qv}</span></td>')
            rows.append(f"""
            <tr>
              <td><span class="pill pill-cat cat-{esc(CATEGORY_OF[t].lower())}">{esc(CATEGORY_OF[t])}</span></td>
              <td class="mono">{esc(t)}</td>
              <td class="mono">n={subset_task_stats[t][MODELS[0][0]]['n']}</td>
              {"".join(cells)}
            </tr>""")
        headers = "".join(f'<th class="num">{label}<br><span class="th-sub">rule / judge / qwen3.8</span></th>'
                           for _, label, _, _ in MODELS)
        return f"""
        <div class="table-wrap subset-table">
          <table class="summary-table">
            <thead><tr><th>Category</th><th>Task</th><th>Subset n</th>{headers}</tr></thead>
            <tbody>{"".join(rows)}</tbody>
          </table>
        </div>"""

    def model_mini_bars(t):
        rows = []
        for key, label, _, _ in MODELS:
            s = task_stats[t][key]
            rows.append(f"""
            <div class="mini-bar-row">
              <span class="mini-bar-label"><span class="model-dot dot-{key}"></span>{label}</span>
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
            q = d["qwen38"].get(rid, {})
            qscore = q.get("judge_score")
            qwen38_line = (
                f'<div class="mcell-scores mcell-scores-q"><span class="judge2-ink mono">qwen3.8 {qscore:.2f}</span></div>'
                if qscore is not None else ""
            )
            model_cells.append(f"""
            <div class="mcell">
              <div class="mcell-label"><span class="model-dot dot-{key}"></span>{label}</div>
              <img src="{gen_uri}" alt="{esc(label.replace('&rarr;', '->'))} output" loading="lazy">
              <div class="mcell-scores">
                <span class="rule-ink mono">{rscore:.2f}</span><span class="sep">/</span><span class="judge-ink mono">{jscore:.2f}</span>
              </div>
              {qwen38_line}
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
          <p class="samples-label">{len(shared_picks[t])} shared test instances, same id across all three models (seed {SEED}{f' + extra:{t}' if t in EXTRA_SAMPLES else ''})</p>
          <div class="samples-list">{rows}</div>
        </section>"""

    subset_table = subset_comparison_table()
    # shared_picks is keyed by every task_name present in base_meta (includes
    # e.g. shape_outline_then_move, which isn't one of the 9 active
    # TASK_ORDER tasks this report covers) -- sum only over TASK_ORDER so
    # this count matches what's actually scored/displayed (110, not 120).
    subset_n_total = sum(len(shared_picks.get(t, [])) for t in TASK_ORDER)
    subset_section = "" if not subset_table else f"""
    <h2 class="subset-h2">Same tasks, on the {subset_n_total}-sample shown subset only &mdash; 3-way</h2>
    <p class="subset-caveat">Added 2026-08-31. This table and the <span class="judge2-ink">qwen3.8</span> score
    under every visual example below are the <b>same {subset_n_total}-id subset</b> this page already displays
    &mdash; not the full 900/task pool the table above averages over. rule and judge here are recomputed on that
    identical subset (not copied from above) so all three numbers in a cell are a fair, apples-to-apples 3-way
    read for exactly these samples. Do not compare this table's rule/judge columns against the full-900 table
    above as if the difference were about the qwen3.8 judge &mdash; some of it is just subset variance from n
    dropping to 10&ndash;20/task. See &ldquo;How this comparison was built,&rdquo; phase 11, for why the new judge
    only covers this subset (a real, measured single call is 40&ndash;70s, making the full 2,700-call corpus a
    ~35&ndash;40 hour job).</p>
    {subset_table}"""

    nav_links = "".join(f'<a href="#t-{esc(t)}">{esc(t)}</a>' for t in TASK_ORDER)
    sections = "".join(task_section(t) for t in TASK_ORDER)
    legend = "".join(
        f'<span class="legend-item"><span class="model-dot dot-{key}"></span>{label}</span>'
        for key, label, _, _ in MODELS
    )

    html = f"""<title>4,500-Row SFT Comparison</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700;800&family=Source+Sans+3:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {{
  --bg: #EEEBE3; --surface: #FBFAF6; --surface-alt: #E4E0D2;
  --ink: #201F1B; --ink-soft: #55524A; --ink-faint: #8A8577; --border: #D6D1C0;
  --rule: #0E7C7B; --rule-soft: #D9EFEE; --judge: #B3236B; --judge-soft: #F7DEEB; --overlap: #4A3B6B;
  --judge2: #1C6BA0; --judge2-soft: #DCEAF5;
  --good: #1C6B5E; --warn: #A9700E; --bad: #A8395A;
  --cat-perception: #A8395A; --cat-knowledge: #2A5CA6; --cat-transformation: #B06A1A;
  --cat-abstraction: #6B4FA0; --cat-spatiality: #1C6B5E;
  --m-frombase: #6B7A8F; --m-from100k: #C9922B; --m-from140k: #4C7A5A;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #16181A; --surface: #1D2020; --surface-alt: #262A29;
    --ink: #ECEAE2; --ink-soft: #B8B4A6; --ink-faint: #837F72; --border: #34383A;
    --rule: #4FD6D2; --rule-soft: #16302F; --judge: #F17BB0; --judge-soft: #3A1E2C; --overlap: #B0A0F0;
    --judge2: #6FB4E8; --judge2-soft: #17293A;
    --good: #5FBFA3; --warn: #D9A548; --bad: #E37FA0;
    --cat-perception: #E37FA0; --cat-knowledge: #7DA8E8; --cat-transformation: #E0A356;
    --cat-abstraction: #A98FE0; --cat-spatiality: #5FBFA3;
    --m-frombase: #9DACC0; --m-from100k: #E8B95C; --m-from140k: #8FCBA8;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #16181A; --surface: #1D2020; --surface-alt: #262A29;
  --ink: #ECEAE2; --ink-soft: #B8B4A6; --ink-faint: #837F72; --border: #34383A;
  --rule: #4FD6D2; --rule-soft: #16302F; --judge: #F17BB0; --judge-soft: #3A1E2C; --overlap: #B0A0F0;
  --judge2: #6FB4E8; --judge2-soft: #17293A;
  --good: #5FBFA3; --warn: #D9A548; --bad: #E37FA0;
  --cat-perception: #E37FA0; --cat-knowledge: #7DA8E8; --cat-transformation: #E0A356;
  --cat-abstraction: #A98FE0; --cat-spatiality: #5FBFA3;
  --m-frombase: #9DACC0; --m-from100k: #E8B95C; --m-from140k: #8FCBA8;
}}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{ background: var(--bg); color: var(--ink); margin: 0; padding: 0 0 6rem; font-family: "Source Sans 3", system-ui, sans-serif; font-size: 16px; line-height: 1.55; }}
.mono {{ font-family: "JetBrains Mono", ui-monospace, monospace; font-variant-numeric: tabular-nums; }}
.wrap {{ max-width: 1280px; margin: 0 auto; padding: 0 2rem; }}
a {{ color: var(--rule); }}
.rule-ink {{ color: var(--rule); }}
.judge-ink {{ color: var(--judge); }}
.judge2-ink {{ color: var(--judge2); }}
.sep {{ color: var(--ink-faint); margin: 0 0.15em; }}

header.top {{ background: var(--surface); border-bottom: 1px solid var(--border); padding: 3.2rem 0 2rem; }}
.kicker {{ font-family: "JetBrains Mono", monospace; font-size: 0.76rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--ink-faint); margin: 0 0 0.8rem; }}
h1 {{ font-family: "Barlow Condensed", system-ui, sans-serif; font-weight: 700; font-size: clamp(2.1rem, 4.6vw, 3.2rem); line-height: 1.02; margin: 0 0 1rem; text-wrap: balance; letter-spacing: -0.01em; }}
.lede {{ color: var(--ink-soft); max-width: 72ch; font-size: 1.03rem; line-height: 1.6; margin: 0 0 1.8rem; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 1.1rem; margin-bottom: 1.6rem; }}
.legend-item {{ display: flex; align-items: center; gap: 0.4rem; font-size: 0.85rem; color: var(--ink-soft); }}
.model-dot {{ width: 0.7rem; height: 0.7rem; border-radius: 50%; display: inline-block; flex-shrink: 0; }}
.dot-frombase {{ background: var(--m-frombase); }}
.dot-from100k {{ background: var(--m-from100k); }}
.dot-from140k {{ background: var(--m-from140k); }}
.model-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 0.9rem; margin-bottom: 1.6rem; }}
.model-card {{ background: var(--surface-alt); border: 1px solid var(--border); border-radius: 10px; padding: 1rem 1.2rem; border-top: 3px solid var(--border); }}
.model-frombase {{ border-top-color: var(--m-frombase); }}
.model-from100k {{ border-top-color: var(--m-from100k); }}
.model-from140k {{ border-top-color: var(--m-from140k); }}
.model-card-head {{ font-family: "Barlow Condensed", system-ui, sans-serif; font-weight: 600; font-size: 1.15rem; display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.6rem; }}
.model-card-stats {{ display: flex; flex-direction: column; gap: 0.25rem; font-size: 0.88rem; }}
.model-card-stats div {{ display: flex; justify-content: space-between; }}
.mcs-label {{ color: var(--ink-faint); text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.04em; }}
.mcs-subset-note {{ font-size: 0.66rem; color: var(--ink-faint); text-align: right; margin-top: -0.15rem; }}
nav.tasknav {{ display: flex; flex-wrap: wrap; gap: 0.35rem 1rem; margin-top: 0.4rem; }}
nav.tasknav a {{ font-family: "JetBrains Mono", monospace; font-size: 0.76rem; color: var(--ink-soft); text-decoration: none; border-bottom: 1px dotted var(--border); }}
nav.tasknav a:hover {{ color: var(--rule); border-color: var(--rule); }}

section.intro {{ padding: 2.6rem 0; border-bottom: 1px solid var(--border); }}
.intro-cols {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 2.4rem; }}
.intro-col h3 {{ font-family: "Barlow Condensed", system-ui, sans-serif; font-size: 1.5rem; margin: 0 0 0.7rem; display: flex; align-items: center; gap: 0.5rem; }}
.ink-dot {{ width: 0.8rem; height: 0.8rem; border-radius: 50%; display: inline-block; }}
.ink-dot-rule {{ background: var(--rule); }}
.ink-dot-judge {{ background: var(--judge); }}
.ink-dot-judge2 {{ background: var(--judge2); }}
.intro-col p {{ color: var(--ink-soft); margin: 0 0 0.8rem; }}
.intro-col ul {{ margin: 0 0 0.8rem; padding-left: 1.2rem; color: var(--ink-soft); }}
.intro-col li {{ margin-bottom: 0.4rem; }}
.lesson-box {{ background: var(--surface-alt); border-left: 3px solid var(--overlap); border-radius: 6px; padding: 0.9rem 1.1rem; margin-top: 1rem; font-size: 0.9rem; color: var(--ink-soft); }}
.lesson-box b {{ color: var(--ink); }}

section.summary {{ padding: 2.6rem 0; border-bottom: 1px solid var(--border); }}
section.summary h2, section.task .task-head h2 {{ font-family: "Barlow Condensed", system-ui, sans-serif; font-weight: 700; }}
section.summary h2 {{ font-size: 1.9rem; margin: 0 0 1.2rem; }}
.subset-h2 {{ font-size: 1.4rem; margin: 2.2rem 0 0.8rem; }}
.subset-caveat {{ background: var(--judge2-soft); border-left: 3px solid var(--judge2); border-radius: 6px;
  padding: 0.8rem 1rem; font-size: 0.84rem; color: var(--ink-soft); margin: 0 0 1.2rem; max-width: 90ch; }}
.subset-caveat b {{ color: var(--ink); }}
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
.mini-bar-row {{ display: grid; grid-template-columns: 14rem 1fr 2.6rem 1fr 2.6rem; align-items: center; gap: 0.6rem; }}
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
.sample-grid {{ display: grid; grid-template-columns: repeat(5, minmax(140px, 1fr)); gap: 0.6rem; overflow-x: auto; }}
.mcell {{ display: flex; flex-direction: column; gap: 0.3rem; min-width: 130px; }}
.mcell-label {{ font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.02em; color: var(--ink-faint); display: flex; align-items: center; gap: 0.3rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.mcell img {{ width: 100%; aspect-ratio: 1; object-fit: contain; background: var(--surface-alt); border-radius: 6px; border: 1px solid var(--border); }}
.ref-cell img {{ border-style: dashed; }}
.mcell-scores {{ font-size: 0.72rem; text-align: center; }}
.mcell-scores-q {{ font-size: 0.68rem; text-align: center; margin-top: -0.1rem; }}

footer.meta {{ padding: 2.6rem 0 0; }}
footer.meta p {{ color: var(--ink-faint); font-size: 0.84rem; line-height: 1.7; max-width: 78ch; }}

section.repro {{ padding: 3.2rem 0 2.6rem; border-bottom: 1px solid var(--border); }}
.repro-head h2 {{ font-family: "Barlow Condensed", system-ui, sans-serif; font-weight: 700; font-size: 1.9rem; margin: 0 0 0.4rem; }}
.repro-head p {{ color: var(--ink-soft); max-width: 74ch; margin: 0 0 2rem; font-size: 0.95rem; }}
.repro-phases {{ display: flex; flex-direction: column; gap: 0; }}
.repro-phase {{ display: grid; grid-template-columns: 3.4rem 1fr; gap: 1.3rem; padding: 1.6rem 0; border-top: 1px solid var(--border); }}
.repro-phase:first-child {{ border-top: none; padding-top: 0; }}
.repro-phase-num {{ font-family: "Barlow Condensed", system-ui, sans-serif; font-weight: 800; font-size: 2rem; color: var(--ink-faint); line-height: 1.1; }}
.repro-phase h3 {{ font-family: "Barlow Condensed", system-ui, sans-serif; font-weight: 700; font-size: 1.3rem; margin: 0 0 0.6rem; }}
.repro-phase p {{ color: var(--ink-soft); margin: 0 0 0.7rem; font-size: 0.9rem; max-width: 76ch; }}
.repro-phase ul {{ margin: 0 0 0.7rem; padding-left: 1.1rem; color: var(--ink-soft); font-size: 0.9rem; }}
.repro-phase li {{ margin-bottom: 0.4rem; max-width: 74ch; }}
.repro-why {{ font-size: 0.84rem; color: var(--ink-faint); border-left: 2px solid var(--overlap); padding: 0.15rem 0 0.15rem 0.8rem; margin: 0.7rem 0; max-width: 72ch; }}
.repro-why b {{ color: var(--ink-soft); }}
.repro-code {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 0.7rem 0.9rem; font-size: 0.7rem; line-height: 1.55; overflow-x: auto; white-space: pre; color: var(--ink-soft); margin: 0.5rem 0 0.8rem; }}
.repro-incident {{ background: var(--judge-soft); border-left: 3px solid var(--judge); border-radius: 6px; padding: 0.75rem 0.95rem; font-size: 0.86rem; color: var(--ink-soft); margin-top: 0.7rem; max-width: 76ch; }}
.repro-incident b {{ color: var(--ink); }}
.repro-note {{ font-size: 0.8rem; color: var(--ink-faint); margin-top: 2rem; max-width: 76ch; }}
@media (max-width: 800px) {{ .task-body-cols, .intro-cols {{ grid-template-columns: 1fr; }} .mini-bar-row {{ grid-template-columns: 8rem 1fr 2.2rem 1fr 2.2rem; }} .sample-grid {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }} .repro-phase {{ grid-template-columns: 1fr; }} .repro-phase-num {{ display: none; }} }}
</style>

<header class="top">
  <div class="wrap">
    <p class="kicker">3 models &middot; VBVR-CustomEval &middot; identical SFT recipe, identical 4,500-row training set, identical 900-sample eval</p>
    <h1>Does starting from a bigger next-frame checkpoint help?</h1>
    <p class="lede">All three models were fine-tuned with the exact same recipe (LoRA rank 32 on the LLM, lr 1e-5, batch 16, no-ce / image-loss-only supervision, all conditioning images through the ViT) on the same new 4,500-row training set &mdash; 9 tasks &times; 500 rows each (the 5 ID tasks subsampled from their existing 2,000-row pool; the 4 OOD tasks freshly generated for this run, since they'd never had a train split before). The only difference is the starting checkpoint: one run starts from the InternVL-U base model, one continues from the checkpoint fine-tuned on 100,000 VBVR next-frame rows, and the third continues from a checkpoint fine-tuned on 140,000 next-frame rows (the same 100k plus 40,000 freshly-generated rows for the 4 tasks that used to be held out as OOD, now folded in as ordinary training data). All three are evaluated on the identical 900-sample (9 task &times; 100), 9-task target-frame-prediction eval split, scored twice: once by a deterministic rule-based detector pipeline, once by a Qwen3-VL-30B-fp8 judge.</p>
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
        <p>Per task, a small set of weighted sub-criteria (e.g. for <code class="mono">stable_sort</code>: classification, order, fidelity, layout) is computed from classical computer vision &mdash; contour detection, HSV color masks, connected-component analysis &mdash; identically for both models' outputs. Deterministic, zero marginal cost per sample.</p>
      </div>
      <div class="intro-col">
        <h3><span class="ink-dot ink-dot-judge"></span>Judge method</h3>
        <p>A Qwen3-VL-30B-fp8 judge is shown the same three images (starting frame, candidate, ground truth) and classifies each sub-criterion into <span class="lvl lvl-wrong">wrong</span> <span class="lvl lvl-partial">partial</span> <span class="lvl lvl-correct">correct</span> (0 / 0.5 / 1.0), combined with the identical weights the rule-based side uses.</p>
      </div>
      <div class="intro-col">
        <h3><span class="ink-dot ink-dot-judge2"></span>Second judge &mdash; Qwen3.8-27B</h3>
        <p>Added 2026-08-31: the identical rubric/instruction/weights, but asked of Qwen3.8-27B (a brand-new hybrid-linear-attention VL model) instead, run locally via <code>transformers</code> rather than vLLM &mdash; vLLM itself cannot currently serve this checkpoint on this cluster (community INT4 quant crashes a Marlin kernel; official bf16's tensor-parallel path needs CUDA-13-only compiled extensions this cluster's driver can't run). Scored on the same {N_SHARED_PER_TASK}&ndash;20-sample subset shown below, not the full 900/task pool &mdash; see the caveat box further down.</p>
      </div>
    </div>
  </div>
</section>

<section class="summary">
  <div class="wrap">
    <h2>All 9 tasks &times; 3 models</h2>
    {summary_table()}
    {subset_section}
  </div>
</section>

<div class="wrap">
{sections}
</div>

<section class="repro">
  <div class="wrap">
    <div class="repro-head">
      <h2>How this comparison was built</h2>
      <p>A step-by-step log of the sessions that produced this report &mdash; what was done, the exact paths/commands, and why each step was necessary &mdash; kept here so the same SFT comparison can be reproduced (or extended to a further checkpoint) without re-deriving any of it. Phases 01-08 are the original 2-model (frombase/from100k) build; phase 09 documents the from140k addition.</p>
    </div>
    <div class="repro-phases">

      <div class="repro-phase">
        <div class="repro-phase-num">01</div>
        <div>
          <h3>Scope the eval set: which tasks are actually active</h3>
          <p><code>evaluators/image_evaluator.py</code>'s <code>LOCKED_TASKS</code> names 10 tasks (5 ID + 5 OOD) as the ones actually used for InternVL-U training/eval, out of 15 implemented evaluator classes. Of those 10, only <b>9</b> have real generations/eval data: <code>glass_refraction</code> (Knowledge, OOD) has no eval split because its DataFactory generator repo is access-restricted. The 9 active tasks:</p>
          <ul>
            <li><b>ID (5, has a training split):</b> ball_bounces_given_time, grid_shift, multi_object_placement, rotation_puzzle, stable_sort</li>
            <li><b>OOD (4, had no training split until this session):</b> shape_color_then_move, animal_size_sorting, maze, 2d_geometric_transformation</li>
          </ul>
          <p>Two training-data pools exist for the ID tasks, from two independent pipelines: <code>data/datasets/vbvr_next_frame/next_frame_train.jsonl</code> (10 tasks &times; 10,000 rows, 3-frame-in &rarr; 1-out, used for the earlier 100k-row baseline) and <code>data/datasets/vbvr_target_pred_id/target_pred_id_train.jsonl</code> (5 of those tasks &times; 2,000 rows, 1-frame-in &rarr; 1-out &mdash; the task type this report is about). The OOD tasks had <b>zero</b> rows in either pool.</p>
        </div>
      </div>

      <div class="repro-phase">
        <div class="repro-phase-num">02</div>
        <div>
          <h3>Generate the missing OOD training data</h3>
          <p>Ran each OOD task's own DataFactory generator for 500 new samples/task, into a <b>new</b> output directory with a <b>new</b> seed &mdash; both deliberate, to avoid corrupting the existing 100-sample eval split:</p>
          <div class="repro-code">python VBVR-DataGeneration/target_pred/vbvr_target_pred_generate_ood_train.py
  # -&gt; TASKS_OOD, seed=342, out_dir=/scratch/network/ssd/junlin/vbvr_target_pred/ood_train500/{{task}}/raw
  # (existing eval split: seed=242, out_dir=.../ood/{{task}}/raw -- untouched)</div>
          <div class="repro-why"><b>Why a new seed and a new directory, not just a bigger --num-raw on the existing one:</b> the generator's <code>generate_dataset()</code> always numbers samples <code>{{domain}}_{{i:08d}}</code> starting at <code>i=0</code>, so re-running it against the same output directory <b>overwrites</b> whatever was already there. Reusing seed 242 with a larger count risks the first 100 draws colliding with the already-scored eval set; a disjoint seed (342, following the project's 142=ID/242=OOD-eval pattern) into a disjoint directory (<code>ood_train500/</code> not <code>ood/</code>) guarantees zero overlap by construction, verified afterward by diffing image paths between the two jsonls (0 overlap).</div>
          <p>Built the jsonl for that raw data with a small new script that reuses the project's existing <code>build_rows()</code> (prompt-wrapping, video-language audit) but writes <code>split="train"</code> to a new file instead of overwriting the eval jsonl:</p>
          <div class="repro-code">python VBVR-DataGeneration/target_pred/vbvr_target_pred_build_ood_train.py
  # -&gt; data/datasets/vbvr_target_pred_ood/target_pred_ood_train.jsonl  (2,000 rows: 4 tasks x 500)</div>
        </div>
      </div>

      <div class="repro-phase">
        <div class="repro-phase-num">03</div>
        <div>
          <h3>Assemble the final 4,500-row training set</h3>
          <p>For the 5 ID tasks, subsampled 500/task (fixed per-task seed <code>tp_9task_subsample:{{task}}</code>) out of the existing 2,000-row train pool. For the 4 OOD tasks, used all 500 freshly generated rows. Every row was then converted to the <b>no_ce</b> format &mdash; a top-level <code>"no_ce": true</code> field plus collapsing the gpt turn from a templated sentence to the bare <code>&lt;img&gt;</code> marker &mdash; to match how the earlier 100k-row baseline was trained (confirmed by diffing that run's with-CE vs no-CE jsonls beforehand; the two default-built jsonls here had never been converted).</p>
          <div class="repro-code">python VBVR-DataGeneration/target_pred/vbvr_target_pred_merge_9task.py
  # -&gt; data/datasets/vbvr_target_pred_9task/target_pred_9task_train_no_ce.jsonl  (4,500 rows)
  # -&gt; data/meta/vbvr_target_pred_9task_meta.json</div>
          <div class="repro-why"><b>Why no_ce matters here:</b> VBVR prompts are synthesized instructions, not real captions, so training the text-CE loss on them risks eroding language ability for no benefit. Collapsing the gpt turn to a constant token makes that loss term contribute nothing informative, leaving the image (flow-matching) loss as the real training signal &mdash; this was an explicit requirement for both new runs, not just an artifact of copying the old recipe.</div>
        </div>
      </div>

      <div class="repro-phase">
        <div class="repro-phase-num">04</div>
        <div>
          <h3>Recover the exact SFT recipe used previously</h3>
          <p>Rather than guess hyperparameters, located and read the actual training script and the actual finished checkpoint from the earlier 100k-row run:</p>
          <ul>
            <li>Script: <code>Model_Related/InternVLU/InternVL/internvl_chat/shell/internvlu/sft/run_vbvr_gen_sft.sh</code> &mdash; LoRA rank 32 on the LLM only (never fully unfrozen), lr 1e-5, effective batch size 16, gen-loss weight 0.5 after a 20-step warmup, all conditioning images through the ViT (the code default, not a flag).</li>
            <li>Base checkpoint: <code>/scratch/network/ssd2/junlin/models/InternVL-U</code>.</li>
            <li>Prior 100k-row checkpoint (merged, inference-ready): <code>/scratch/network/ssd/junlin/models/internvlu-vbvr-gen-noce-merged</code> &mdash; this is baseline (b)'s starting point.</li>
          </ul>
        </div>
      </div>

      <div class="repro-phase">
        <div class="repro-phase-num">05</div>
        <div>
          <h3>Train both baselines with identical settings</h3>
          <p>Same script, same <code>META_PATH</code> (the new 4,500-row meta), same hyperparameters for both runs &mdash; only <code>INTERNVLU_CKPT</code> and <code>OUTPUT_DIR</code> differ:</p>
          <div class="repro-code">CUDA_VISIBLE_DEVICES=2,3 GPUS=2 \
META_PATH=data/meta/vbvr_target_pred_9task_meta.json \
OUTPUT_DIR=.../internvlu-vbvr-9task4500-frombase \
bash shell/internvlu/sft/run_vbvr_gen_sft.sh          # (a) from base -- INTERNVLU_CKPT defaults to the base snapshot

CUDA_VISIBLE_DEVICES=2,3 GPUS=2 \
META_PATH=data/meta/vbvr_target_pred_9task_meta.json \
OUTPUT_DIR=.../internvlu-vbvr-9task4500-from100k \
INTERNVLU_CKPT=.../internvlu-vbvr-gen-noce-merged \
bash shell/internvlu/sft/run_vbvr_gen_sft.sh          # (b) continues from the 100k-row checkpoint</div>
          <p>GPUs 2,3 on torrnode8 only (not 6,7, which another user's job was lightly using at the time) &mdash; the two runs went sequentially, not in parallel, to avoid splitting the 2 free GPUs further. <code>gradient_accumulation_steps</code> is computed as <code>BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS</code>, so dropping from the original 4 GPUs to 2 here automatically doubled it (4&rarr;8), preserving the effective batch size of 16 with no config changes needed. <code>EPOCHS</code> was kept at the literal value 1 from the original script (&asymp;282 steps for 4,500 rows) rather than scaled up to match the original run's 6,250 total steps &mdash; a deliberate choice to keep every non-data-volume setting identical, not an oversight.</p>
          <div class="repro-incident"><b>False alarm, not a bug:</b> the first logged step reported ~127s/it (&asymp;10h ETA), which looked alarming. It turned out tqdm's first-iteration timing includes one-time model/data loading overhead; steady state settled to ~28s/step (&asymp;2h10m/run), consistent with the earlier 100k-row run's own <code>trainer_state.json</code> (<code>train_runtime=64,023s / 6,250 steps = 10.24s/step on 4 GPUs</code> &rarr; ~2x slower on 2 GPUs is the expected, unremarkable number).</div>
        </div>
      </div>

      <div class="repro-phase">
        <div class="repro-phase-num">06</div>
        <div>
          <h3>Generate on the 900-sample eval split</h3>
          <div class="repro-code">python inference/run_target_pred_eval.py --gpu &lt;n&gt; --ckpt &lt;merged-pipeline-dir&gt; --tag &lt;name&gt;
  # --prescale-input defaults to 512 (long-edge resize) -- left at default, matching every other model's runs</div>
          <div class="repro-incident"><b>1,000 samples were generated, not 900:</b> <code>target_pred_id_eval.jsonl</code> still carries 100 leftover eval rows for <code>shape_outline_then_move</code>, a task dropped from <code>LOCKED_TASKS</code> back on 2026-08-25 for being redundant with another task. <code>run_target_pred_eval.py</code>'s <code>load_rows()</code> just reads the whole ID+OOD eval jsonls, so it has no way to know that task isn't active. Harmless (the extra 100 generations/scores per model are simply excluded by this report's hardcoded 9-task list) but worth ~11% extra compute &mdash; worth deleting those rows from the source jsonl before the next run.</div>
          <div class="repro-incident"><b>Node contention doubled inference time, then got fixed mid-run:</b> torrnode8's CPU was saturated by an unrelated user's job (load average ~127 on 64 cores) while its GPUs sat mostly idle, starving this script's CPU-side image preprocessing. Moved both models' inference to less-contended GPUs/nodes (torrnode12, then torrnode8's freed GPU3, then torrnode12 again) once the training GPUs freed up. <code>run_target_pred_eval.py</code> skips any <code>&lt;id&gt;.png</code> that already exists on disk, so killing and relaunching on a different node/GPU loses zero completed samples.</div>
        </div>
      </div>

      <div class="repro-phase">
        <div class="repro-phase-num">07</div>
        <div>
          <h3>Score both models</h3>
          <div class="repro-code">python validation/score_target_pred_eval.py --run-dir &lt;run_dir&gt;              # rule-based, ~3 min for both combined
python judge_eval/score_all_with_judge.py --run-dir &lt;run_dir&gt; --workers 8   # judge, ~20 min each, sequential</div>
          <div class="repro-incident"><b>Judge scoring failed 100% on the first pass</b> (2,000/2,000 "Connection error") &mdash; the local vLLM judge server (<code>qwen3-vl-30b-fp8</code> on <code>localhost:8000</code>) had been shut down that same morning and never restarted; nothing was listening on the port. Recovered the exact original launch invocation from the server's own log file and relaunched it on a free GPU:</div>
          <div class="repro-code">CUDA_VISIBLE_DEVICES=&lt;free gpu&gt; vllm serve /scratch/local/ssd/junlin/models/Qwen3-VL-30B-A3B-Instruct-FP8 \
  --trust-remote-code --max-model-len 8192 --enforce-eager \
  --served-model-name qwen3-vl-30b-fp8 --gpu-memory-utilization 0.85 \
  --mm-processor-cache-gb 0.0 --max-num-seqs 4 --port 8000</div>
          <p>Re-ran judge scoring for both models against the restarted server: 0 errors, 1,000/1,000 each.</p>
        </div>
      </div>

      <div class="repro-phase">
        <div class="repro-phase-num">08</div>
        <div>
          <h3>Build this report</h3>
          <p>Adapted from the project's existing 4-model template (<code>artifacts/gen_comparison_report.py</code>) to keep the same visual language and scoring conventions as sibling reports, trimmed to 10 shared examples/task instead of 20 (this phase originally built the 2-model frombase/from100k version; see phase 09 for the from140k addition):</p>
          <div class="repro-code">python artifacts/gen_9task_sft_comparison.py
  # -&gt; artifacts/9task_sft_comparison.html</div>
          <div class="repro-why"><b>One fix needed on the way in:</b> the task list can't be taken from <code>TASK_CRITERIA.keys()</code> anymore &mdash; it now has 10 entries because <code>shape_outline_then_move</code> was added to it after the historical 900-sample runs. <code>TASK_ORDER</code> here is explicitly the 9 active task names, matching phase 01.</div>
        </div>
      </div>

      <div class="repro-phase">
        <div class="repro-phase-num">09</div>
        <div>
          <h3>Add the from140k model (follow-up session)</h3>
          <p>Same 4,500-row training set and identical SFT recipe as phases 03-05, but starting from a new, bigger next-frame checkpoint instead of the original 100k one. That checkpoint was itself built first, in the same session:</p>
          <ul>
            <li>Generated 40,000 fresh next-frame (3-in &rarr; 1-out) rows for the 4 tasks that had been held out as OOD in the 100k-row next-frame pretraining set (<code>shape_color_then_move</code>, <code>maze</code>, <code>2d_geometric_transformation</code>, <code>animal_size_sorting</code>; <code>glass_refraction</code> excluded, still no train jsonl) &mdash; raw-video generation first ran sequentially per <code>vbvr_next_frame_generate.py</code>'s existing per-task chunking, then was switched mid-run to a task-level-parallel driver (<code>vbvr_next_frame_generate_parallel.py</code>) after a live memory measurement showed a single 500-sample chunk peaks at only ~6GB RSS, making 4 concurrent tasks trivially safe on a 196GB-free shared node.</li>
            <li>Continuation-SFT'd the existing 100k-row next-frame checkpoint on those 40,000 rows (same <code>run_vbvr_gen_sft.sh</code>, same hyperparameters, only <code>INTERNVLU_CKPT</code>/<code>META_PATH</code>/<code>OUTPUT_DIR</code> changed) &mdash; giving a checkpoint that has now seen 100k + 40k = 140,000 next-frame rows total, with the former OOD tasks now treated as ordinary training data rather than held out.</li>
            <li>Restarted that training run once mid-flight after moving from a node where 3 of 4 GPUs had become contended by other tenants (step time had drifted to ~18s/step) to a node with 4 fully idle A40s (step time then ~11s/step, matching the original 100k run's per-step rate) &mdash; <code>save_strategy=no</code> meant the ~2h10m of progress on the contended node was lost and the run restarted from step 0, a deliberate trade given the contended node was actively getting worse, not just occasionally slower.</li>
          </ul>
          <p>From there, the same phase-03-through-07 pipeline ran unchanged against this new checkpoint:</p>
          <div class="repro-code">OUTPUT_DIR=.../internvlu-vbvr-9task4500-from140k \
INTERNVLU_CKPT=.../internvlu-vbvr-gen-noce-140k-merged \
bash shell/internvlu/sft/run_vbvr_gen_sft.sh
python inference/run_target_pred_eval.py --gpu &lt;n&gt; --ckpt .../internvlu-vbvr-9task4500-from140k-merged --tag from140k_9task4500
python validation/score_target_pred_eval.py --run-dir results/vbvr_target_pred_eval/from140k_9task4500
python judge_eval/score_all_with_judge.py --run-dir results/vbvr_target_pred_eval/from140k_9task4500 --workers 8
python artifacts/gen_9task_sft_comparison.py   # this file, extended: MODELS + a 3rd CSS color/legend/grid-column slot</div>
          <div class="repro-incident"><b>Judge server needed a restart again:</b> it had been deliberately shut down between sessions to free its GPU. Relaunched with the exact invocation recorded in phase 07 before running judge scoring &mdash; 0 errors, 1,000/1,000.</div>
        </div>
      </div>

      <div class="repro-phase">
        <div class="repro-phase-num">10</div>
        <div>
          <h3>Add 10 extra examples for two tasks</h3>
          <p>On request, <code>maze</code> and <code>2d_geometric_transformation</code> each got 10 additional shared examples (20 total, up from 10) without touching any other task's samples or perturbing the original 10 for these two tasks: <code>EXTRA_SAMPLES</code> draws the extra picks from an <b>independent</b> per-task RNG stream (<code>random.Random(f"&#123;SEED&#125;:extra:&#123;task&#125;")</code>) seeded and consumed strictly after the original single shared-stream <code>shared_picks</code> computation, excluding ids already picked. Re-running <code>gen_9task_sft_comparison.py</code> after this change reproduces every previously-published sample row byte-for-byte; only new rows are appended.</p>
        </div>
      </div>

      <div class="repro-phase">
        <div class="repro-phase-num">11</div>
        <div>
          <h3>Add a second judge: Qwen3.8-27B, via <code>transformers</code> instead of vLLM</h3>
          <p>Qwen3.8-27B is a brand-new (released 2026-08-14) hybrid-linear-attention native-VL model. vLLM 0.26.0 already ships native support for its architecture (<code>Qwen3_5ForConditionalGeneration</code>), but neither checkpoint actually serves on this cluster:</p>
          <ul>
            <li>The community INT4 quant (<code>cyankiwi/Qwen3.8-27B-AWQ-INT4</code>, single-GPU) crashes vLLM's Marlin repack CUDA kernel with an empty error message during weight loading. Ruled out a checkpoint/vLLM module-naming mismatch as the cause by directly testing vLLM's <code>hf_to_vllm_mapper</code> against the checkpoint's compressed-tensors ignore-list entries -- the mapping is correct -- so this looks like a bug specific to that third-party quant, not something fixable from the caller side.</li>
            <li>The official bf16 checkpoint, tensor-parallel=2 across two A40s, gets past NCCL init and weight loading but then hits <code>CUDA error: CUDA driver version is insufficient for CUDA runtime version</code> inside vLLM's <code>vllm_flash_attn</code> extension -- a separately-compiled CUDA-13-targeted binary, bundled unconditionally regardless of which <code>torch+cuXXX</code> variant pip resolves for the main package. <code>--disable-custom-all-reduce</code> routes around an identical CUDA-13 wall in vLLM's custom-all-reduce extension, but <code>--attention-backend TRITON_ATTN</code> does <b>not</b> route around the flash-attn one: that extension gets imported and probed unconditionally at startup regardless of the configured backend.</li>
          </ul>
          <p>Also separately root-caused, while investigating whether another node might have a newer driver: the entire cluster (checked 7 of 17 reachable nodes, both A40 and Quadro RTX 6000 boxes) runs the identical Ansible-managed driver, 530.30.02, which supports CUDA up to ~12.8 but not 13 -- so no node on this cluster would serve the bf16 checkpoint via vLLM today.</p>
          <p>Plain <code>transformers.generate()</code> sidesteps all of this: <code>attn_implementation="sdpa"</code> uses PyTorch's built-in attention kernel (part of torch itself, not a separately-compiled extension), and the model's own Gated-DeltaNet linear-attention layers have a documented pure-PyTorch fallback. Confirmed working end-to-end (10-run VQA benchmark, 5.6 tok/s average -- much slower than a real vLLM server would give, since this is naive multi-GPU pipeline parallelism with no batching, but it runs) before building <code>evaluators/llm_judge_transformers.py</code>, a drop-in for <code>FullVLMJudge</code> that reuses <code>TASK_CRITERIA</code>/<code>build_full_judge_instruction</code> unchanged and only swaps the OpenAI-client call for a local <code>model.generate()</code> call.</p>
          <div class="repro-incident"><b><code>device_map="auto"</code> silently offloads to disk with zero error:</b> accelerate's automatic memory balancer proved too conservative on this box's currently-free GPUs and repeatedly put a few layers (once, just <code>lm_head</code>) on <code>'disk'</code> rather than erroring -- catastrophic for latency (a first attempt at the VQA benchmark came back at 0.46 tok/s, ~600x slower, purely from repeated disk reads) and gives no warning beyond an easy-to-miss log line. Fix: build the <code>device_map</code> dict by hand (a fixed 40/24 decoder-layer split across the two GPUs used, <code>lm_head</code>/<code>embed_tokens</code>/<code>visual</code> pinned to the fuller one) and assert <code>hf_device_map</code> contains no <code>'cpu'</code>/<code>'disk'</code> entries before proceeding -- both <code>llm_judge_transformers.py</code> and the benchmark script do this now.</div>
          <div class="repro-incident"><b>The model's own "low" reasoning-effort setting is not actually short:</b> Qwen3.8-27B's chat template defaults to <code>reasoning_effort="xhigh"</code> and exposes <code>enable_thinking</code>/<code>reasoning_effort</code> template kwargs. Measured <code>reasoning_effort="low"</code> on a 4-criterion task: 238s, and it never even reached the JSON answer within <code>max_tokens=400</code> -- still produces a long, unstructured visual analysis. <code>enable_thinking=False</code> (template injects an empty <code>&lt;think&gt;&lt;/think&gt;</code> block, skipping the reasoning step outright) measured at 46s on the identical sample and reliably produces clean, valid JSON with sensible per-criterion reasoning -- used as the default from here on.</div>
          <p>Scored the identical <code>shared_picks</code> subset this report already displays (110 ids &times; 3 models = 330 judge calls), not the full 900/task pool: a real, measured single call is 40&ndash;70s (task-criteria-count dependent) even with <code>enable_thinking=False</code>, making the full 2,700-call corpus a ~35&ndash;40 hour job -- infeasible in one sitting. User-confirmed scope choice, given both the measured per-call cost and that the ask ("add the new model's score under each visual example") is itself scoped to exactly this subset.</p>
          <div class="repro-code">python judge_eval/score_shared_with_qwen38.py
  # -&gt; results/vbvr_target_pred_eval/&lt;model&gt;_9task4500/judge_scored_qwen38.json (per model, 110 ids each)
python artifacts/gen_9task_sft_comparison.py   # this file, extended: subset-only 3-way table + section,
                                                 # qwen3.8 line under every visual example, phase 11</div>
        </div>
      </div>

    </div>
    <p class="repro-note">Per-task judge-score ratios between any pair of models aren't baked into this page but are a one-line re-derivation: load two models' <code>judge_scored.json</code>, average <code>judge_score</code> by <code>task_name</code>, divide.</p>
  </div>
</section>

<footer class="meta">
  <div class="wrap">
    <p><b>Models:</b> all three InternVL-U, single-image generation, 512px output; only the starting checkpoint and training-data history differ. <b>Judge:</b> <code>qwen3-vl-30b-fp8</code>, temperature 0, images downscaled to 512px on the long edge, one call per sample (900 &times; 3 = 2,700 total judge calls across this report). <b>Sampling:</b> {N_SHARED_PER_TASK} shared test instances per task (20 for <code>maze</code> and <code>2d_geometric_transformation</code>, each with 10 extra appended in a follow-up update via an independent RNG stream), Python <code>random.Random({SEED})</code>, identical ids across all three models, not cherry-picked. <code>glass_refraction</code> is excluded: no generations exist for it (its DataFactory generator repo is access-restricted). <b>Second judge (added 2026-08-31):</b> Qwen3.8-27B, bf16, run locally via <code>transformers</code> (<code>attn_implementation="sdpa"</code>, <code>enable_thinking=False</code>), temperature 0, same rubric/instruction/weights and same 512px images as the first judge, but scored on only the {subset_n_total}-id shared subset shown on this page (not the full 900/task pool) &mdash; see phase 11.</p>
  </div>
</footer>
"""
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write(html)
    print(f"wrote {OUT_PATH} ({len(html)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
