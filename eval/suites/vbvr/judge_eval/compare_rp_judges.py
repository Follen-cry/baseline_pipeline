#!/usr/bin/env python3
"""Compare the three rotation_puzzle scoring systems on the same 500 samples:

  old rule   scored.json            (image_evaluator.py, the pre-rewrite scorer)
  new rule   scored_new.json        (scorers/rotation_puzzle.py, deterministic CV)
  rubric     judge_scored.json      (llm_judge_full.py, weighted wrong/partial/correct)
  yes/no     judge_scored_yesno.json (judge_yesno.py, 11 binary questions)

Reports, per scoring system: per-variant mean, the spread between variants, how
many distinct score values it can actually produce, and its per-sample rank
correlation with the deterministic rule scorer -- which is the only one of the
four with a documented held-out agreement check (10/10, rotation_puzzle_notes.md),
so it is used here as the reference, not as ground truth.

    python compare_rp_judges.py
"""
import json
import os
from collections import Counter

RUN = "/scratch/network/ssd2/junlin/ssl_mllm/results/vbvr_target_pred_eval"
VARIANTS = ["base", "s0", "s1", "s2", "s3"]
TASK = "rotation_puzzle"


def _spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):                       # average ties
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return _pearson(rank(xs), rank(ys))


def _pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def load():
    """-> {variant: {system: {id: score}}}"""
    out = {}
    for v in VARIANTS:
        d = os.path.join(RUN, f"{v}_4task500sft")
        sys_scores = {}
        for label, fn, kind in [("old rule", "scored.json", "records"),
                                ("new rule", "scored_new.json", "records"),
                                ("rubric", "judge_scored.json", "judge"),
                                ("yes/no", "judge_scored_yesno.json", "yesno")]:
            path = os.path.join(d, fn)
            if not os.path.exists(path):
                continue
            blob = json.load(open(path))
            if kind == "records":
                sys_scores[label] = {r["id"]: r["score"] for r in blob["records"]
                                     if r["task_name"] == TASK and r["score"] is not None}
            elif kind == "judge":
                sys_scores[label] = {k: r["judge_score"] for k, r in blob["results"].items()
                                     if r["task_name"] == TASK and r.get("judge_score") is not None}
            else:
                sys_scores[label] = {k: r["score"] for k, r in blob["results"].items()
                                     if r.get("score") is not None}
        out[v] = sys_scores
    return out


def main():
    data = load()
    systems = [s for s in ["old rule", "new rule", "rubric", "yes/no"]
               if all(s in data[v] for v in VARIANTS)]

    # ids every system scored, in every variant
    common = {v: set.intersection(*[set(data[v][s]) for s in systems]) for v in VARIANTS}

    print(f"rotation_puzzle -- {sum(len(c) for c in common.values())} samples "
          f"({', '.join(f'{v}:{len(common[v])}' for v in VARIANTS)})\n")

    hdr = f"{'system':10s} " + " ".join(f"{v:>7s}" for v in VARIANTS) + "   spread  levels"
    print(hdr)
    print("-" * len(hdr))
    for s in systems:
        means = [sum(data[v][s][i] for i in common[v]) / len(common[v]) for v in VARIANTS]
        allv = [data[v][s][i] for v in VARIANTS for i in common[v]]
        levels = len({round(x, 4) for x in allv})
        print(f"{s:10s} " + " ".join(f"{m:7.3f}" for m in means)
              + f"   {max(means)-min(means):6.3f}  {levels:6d}")

    ref = "new rule"
    print(f"\nper-sample agreement with '{ref}' (pooled over all {sum(len(c) for c in common.values())} samples):")
    rx = [data[v][ref][i] for v in VARIANTS for i in sorted(common[v])]
    for s in systems:
        if s == ref:
            continue
        sy = [data[v][s][i] for v in VARIANTS for i in sorted(common[v])]
        mae = sum(abs(a - b) for a, b in zip(rx, sy)) / len(rx)
        print(f"  {s:10s} pearson={_pearson(rx, sy):5.2f}  spearman={_spearman(rx, sy):5.2f}  MAE={mae:.3f}")

    print("\nscore distributions:")
    for s in systems:
        allv = [round(data[v][s][i], 3) for v in VARIANTS for i in common[v]]
        c = Counter(allv)
        top = ", ".join(f"{k}:{n}" for k, n in sorted(c.items())[:14])
        print(f"  {s:10s} {top}{' ...' if len(c) > 14 else ''}")

    # Which questions actually discriminate -- the failure mode that killed the
    # rubric judge was criteria that were constants.
    print("\nyes/no per-question 'no' counts (a question that is never 'no' is dead weight):")
    no_counts, n_tot = Counter(), 0
    for v in VARIANTS:
        blob = json.load(open(os.path.join(RUN, f"{v}_4task500sft", "judge_scored_yesno.json")))
        for r in blob["results"].values():
            if not r.get("answers"):
                continue
            n_tot += 1
            for k, val in r["answers"].items():
                if not val:
                    no_counts[k] += 1
    for k in json.load(open(os.path.join(RUN, "base_4task500sft", "judge_scored_yesno.json")))["results"][
            next(iter(json.load(open(os.path.join(RUN, "base_4task500sft", "judge_scored_yesno.json")))["results"]))]["answers"]:
        print(f"  {k:20s} no={no_counts[k]:4d} / {n_tot}")


if __name__ == "__main__":
    main()
