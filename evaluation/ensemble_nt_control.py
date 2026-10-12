"""Ensemble control: mix the average of the seven Base distributions (E) with the NT distribution of each target model and compare with E alone.
Run from the repository root: python3 evaluation/ensemble_nt_control.py"""
import re, numpy as np
src = open('evaluation/base_base_control.py').read()
exec(src[:src.index('rows = []   #')])          # loaders, metrics, DATA, item_metrics
from scipy.stats import wilcoxon
W = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6]
print("JS (lower better), mean over target models. E = mean of all seven Base models; row 0.0 = ensemble alone")
print(f"{'dataset':10s} {'Base+NT .55':>11s} " + " ".join(f"{'E+NT '+str(w):>9s}" for w in W))
res = {}
for name in DATA:
    d = DATA[name]; models = list(d['per'])
    ids = sorted(set.intersection(*[d['per'][t]['ids'] for t in models]))
    E = {k: np.mean([d['per'][t]['B'][k] for t in models], axis=0) for k in ids}
    ref = np.mean([item_metrics(name, ids, lambda k, i=i: .55*d['per'][i]['F'][k] + .45*d['per'][i]['B'][k])[:, 0].mean() for i in models])
    row = []
    for w in W:
        v = [item_metrics(name, ids, lambda k, i=i: w*d['per'][i]['F'][k] + (1-w)*E[k]) for i in models]
        row.append(np.mean([a[:, 0].mean() for a in v])); res[(name, w)] = np.mean([a[:, 0] for a in v], axis=0)
    print(f"{name:10s} {ref:11.4f} " + " ".join(f"{x:9.4f}" for x in row))
print("\npaired item-level test: best E+NT weight vs ensemble alone (positive = adding NT helps)")
rng = np.random.default_rng(0)
for name in DATA:
    base = res[(name, 0.0)]
    best = min(W[1:], key=lambda w: res[(name, w)].mean())
    diff = base - res[(name, best)]
    d = DATA[name]; ids = sorted(set.intersection(*[d['per'][t]['ids'] for t in d['per']]))
    cl = {}
    for k, q in enumerate(ids):
        m = re.search(r"_W(\d+)$", q) if name == "OpinionQA" else None
        key = f"W{m.group(1)}" if m else (d['Q'][q].get('country', q) if name == "ISSP" else q)
        cl.setdefault(key, []).append(k)
    keys = list(cl)
    boot = [diff[np.concatenate([cl[keys[j]] for j in rng.choice(len(keys), len(keys))])].mean() for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  {name:10s} best w={best}  gain {diff.mean():+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  NT helps on {(diff>0).mean():.0%} of items")
