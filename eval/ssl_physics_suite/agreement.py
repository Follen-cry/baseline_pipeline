#!/usr/bin/env python
"""Judge sanity check: agreement of the main judge with (a) a second run of itself and (b) a
different judge, on the 5% subset in sanity_uids.txt (all 4 models).

For every benchmark we compare the per-unit native score the benchmark defines:
  phyeditbench: each dimension score (1-10); anti: each dimension (1-10); picabench: each QA's
  yes/no answer; risebench: Reasoning/ApprConsistency/VisualPlausibility (1-5);
  imgedit/uge/magicbrush: the averaged 1-5 score.
Reports exact agreement, within-1 agreement, Cohen's kappa (yes/no) / Spearman rho (ordinal),
and model-ranking agreement (Kendall tau of the 4 models' subset means).
Writes results/judge_agreement.csv (env: internvlu)
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
MAIN = "qwen3vl30b_fp8"
OTHERS = {"qwen3vl30b_fp8_rerun": "same judge, 2nd run", "qwen3vl8b": "different judge (Qwen3-VL-8B)"}
MODELS = ["base", "T0", "T2", "T3"]


def units(judge, bench, keep):
    out = {}
    for m in MODELS:
        p = os.path.join(RES, "judge_raw", judge, bench, f"{m}.jsonl")
        if not os.path.exists(p):
            continue
        for l in open(p):
            r = json.loads(l)
            if r["uid"] not in keep:
                continue
            if bench == "phyeditbench":
                out[(m, r["key"])] = r.get("score")
            elif bench == "phyedit_anti":
                for d in ("Instruction_Following", "Physical_Plausibility", "Consistency", "Image_Quality"):
                    out[(m, r["key"] + "|" + d)] = r.get(d)
            elif bench == "picabench":
                out[(m, r["key"])] = r.get("model_answer")
            elif bench == "risebench":
                for d, v in (r.get("scores") or {}).items():
                    if d in ("Reasoning", "ApprConsistency", "VisualPlausibility"):
                        out[(m, r["key"] + "|" + d)] = v
            else:
                out[(m, r["key"])] = r.get("score")
    return out


def main():
    keep = set(open(os.path.join(HERE, "sanity_uids.txt")).read().split())
    rows = []
    for other, desc in OTHERS.items():
        for bench in ("phyeditbench", "phyedit_anti", "picabench", "risebench", "imgedit_basic", "imgedit_uge", "magicbrush"):
            a, b = units(MAIN, bench, keep), units(other, bench, keep)
            ks = [k for k in a if k in b and a[k] is not None and b[k] is not None]
            if not ks:
                continue
            x, y = [a[k] for k in ks], [b[k] for k in ks]
            row = dict(comparison=desc, other_judge=other, bench=bench, n_units=len(ks))
            if bench == "picabench":
                row["exact_agree"] = float(np.mean([i == j for i, j in zip(x, y)]))
                xb, yb = np.array([i == "Yes" for i in x]), np.array([j == "Yes" for j in y])
                po = np.mean(xb == yb)
                pe = xb.mean() * yb.mean() + (1 - xb.mean()) * (1 - yb.mean())
                row["kappa"] = float((po - pe) / (1 - pe)) if pe < 1 else np.nan
            else:
                x, y = np.array(x, float), np.array(y, float)
                row["exact_agree"] = float(np.mean(x == y))
                row["within1_agree"] = float(np.mean(np.abs(x - y) <= 1))
                row["spearman"] = float(stats.spearmanr(x, y).statistic) if len(set(x)) > 1 and len(set(y)) > 1 else np.nan
                row["mean_main"], row["mean_other"] = float(x.mean()), float(y.mean())
            # model ranking on the subset
            def mmeans(d):
                return [np.mean([(v == "Yes") if isinstance(v, str) else v for (m, _), v in d.items()
                                 if m == mod and v is not None and (m, _) in set(ks)]) for mod in MODELS]
            ma, mb = mmeans({k: a[k] for k in ks}), mmeans({k: b[k] for k in ks})
            row["model_rank_kendall_tau"] = float(stats.kendalltau(ma, mb).statistic)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "judge_agreement.csv"), index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
