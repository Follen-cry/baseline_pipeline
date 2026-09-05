"""Score every (sample, variant) pair for a task with a new scorer and report
the distribution alongside the old rule score and the VLM judge score."""
import importlib, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import samples


def scan(task, module_name, limit=None):
    mod = importlib.import_module(module_name)
    ids = samples.ids_for(task)
    if limit:
        ids = ids[:limit]
    rows = []
    for v in samples.VARIANTS:
        js = samples.judge_scores(v)
        os_ = samples.old_scores(v)
        for rid in ids:
            inp, cand, gt = samples.triple(rid, v)
            try:
                sub = mod.score(inp, cand, gt)
                ov = mod.overall(sub)
            except Exception as e:
                sub, ov = {}, float('nan')
                print('ERR', rid, v, e)
            j = js.get(rid, {})
            rows.append({'id': rid, 'variant': v, 'new': ov, 'sub': sub,
                         'old': os_.get(rid, {}).get('score'),
                         'judge': j.get('judge_score'),
                         'judge_criteria': j.get('judge_criteria')})
    return rows


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('task'); ap.add_argument('module')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--out')
    a = ap.parse_args()
    rows = scan(a.task, a.module, a.limit)
    if a.out:
        json.dump(rows, open(a.out, 'w'), default=float)
    for v in samples.VARIANTS:
        r = [x for x in rows if x['variant'] == v]
        new = np.array([x['new'] for x in r], float)
        old = np.array([x['old'] for x in r if x['old'] is not None], float)
        jud = np.array([x['judge'] for x in r if x['judge'] is not None], float)
        print(f"{v:20s} new={new.mean():.3f} (sd {new.std():.3f}, min {new.min():.2f}, "
              f"max {new.max():.2f}, n<0.5 {int((new<0.5).sum())})  "
              f"old={old.mean() if len(old) else float('nan'):.3f}  "
              f"judge={jud.mean() if len(jud) else float('nan'):.3f} (n={len(jud)})")
