"""Cross-model control: Base_i + NT_j versus Base_i + Base_j at weight 0.55 over the 42 ordered pairs (i != j).
Run from the repository root after base_base_control.py is available: python3 evaluation/cross_nt_control.py"""
import re, numpy as np
src = open('evaluation/base_base_control.py').read()
exec(src[:src.index('rows = []   #')])
A = 0.55
rng = np.random.default_rng(0)
def clusters(name, ids):
    d = DATA[name]; cl = {}
    for k, q in enumerate(ids):
        m = re.search(r"_W(\d+)$", q) if name == "OpinionQA" else None
        key = f"W{m.group(1)}" if m else (d['Q'][q].get('country', q) if name == "ISSP" else q)
        cl.setdefault(key, []).append(k)
    return list(cl.values())
def ci(diff, cl):
    b = [diff[np.concatenate([cl[j] for j in rng.choice(len(cl), len(cl))])].mean() for _ in range(2000)]
    return np.percentile(b, [2.5, 97.5])
print("Target Base_i mixed at 0.55 with partner j:  Base_i+NT_j  vs  Base_i+Base_j   (i != j, 42 ordered pairs)")
for name in DATA:
    P = DATA[name]['per']; models = list(P)
    ids = sorted(set.intersection(*[P[t]['ids'] for t in models])); cl = clusters(name, ids)
    out = {}
    for mi, mn in ((0, 'JS'), (1, 'Rep')):
        X, Y, own, won = [], [], [], 0
        for i in models:
            own.append(item_metrics(name, ids, lambda q: A*P[i]['F'][q] + (1-A)*P[i]['B'][q])[:, mi])
            for j in models:
                if j == i: continue
                x = item_metrics(name, ids, lambda q: A*P[j]['F'][q] + (1-A)*P[i]['B'][q])[:, mi]
                y = item_metrics(name, ids, lambda q: A*P[j]['B'][q] + (1-A)*P[i]['B'][q])[:, mi]
                X.append(x); Y.append(y)
                won += (x.mean() < y.mean()) if mn == 'JS' else (x.mean() > y.mean())
        X, Y, O = np.mean(X, 0), np.mean(Y, 0), np.mean(own, 0)
        diff = (Y - X) if mn == 'JS' else (X - Y)
        lo, hi = ci(diff, cl)
        print(f"{name:10s} {mn:3s}  B+NT_j {X.mean():.4f}  B+B_j {Y.mean():.4f}  own B+NT_i {O.mean():.4f}  diff(+ favors NT) {diff.mean():+.4f} [{lo:+.4f},{hi:+.4f}]  won {won}/42")
