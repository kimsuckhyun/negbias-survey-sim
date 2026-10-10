#!/usr/bin/env python3
"""base_base_control.py -- is the Base+NT gain just an ensembling effect?

Control requested for the appendix: mix Base_i with another model's Base_j (instead of NT_i) and compare with
Base_i + NT_i against the human distribution.

Run in <project root>. Reads stored panel outputs only (no GPU, no model loading).
Loaders, excluded items, positions and metrics are the same as rep_wd.py.

Conditions for target model i, partner j != i, weight w on the partner (w = 0.55, the paper's alpha):
    NT   : w * NT_i + (1-w) * Base_i                      (the paper's mixture)
    BB   : w * Base_j + (1-w) * Base_i                    (control, one partner)
    BB6  : w * mean_{j != i} Base_j + (1-w) * Base_i      (control, average of the six other Base models)
Every condition for a pair (i, j) is evaluated on the same items (those available for both i and j).

Sections printed:
  (1) fixed weight 0.55: Base, NT, Base+NT, Base+Base (mean / best / worst partner), Base+mean(6 Base)
  (2) per-model wins of Base+NT over the Base+Base partners
  (3) tuned control: the best weight chosen on OpinionQA for Base+Base, applied unchanged to the other datasets
  (4) oracle: best weight (and best partner) per model, i.e. the strongest possible control
  (5) paired item-level test, Base+NT vs Base+Base (Wilcoxon, win rate)
Results are also written to eval/base_base_control.csv.

Usage:  python3 base_base_control.py 2>&1 | tee eval/base_base_control.txt
"""
import json, os, re, sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

M = ["gemma4-12b", "gemma4-31b", "qwen36-27b", "phi4-14b", "mistral-24b", "gptoss-20b", "exaone-32b"]
DS = {"OpinionQA": ("eval/converted", "eval/panel_h100"),
      "GOQA": ("eval/converted_goqa", "eval/panel_goqa"),
      "GSS": ("eval/converted_gss", "eval/panel_gss"),
      "ISSP": ("eval/converted_issp", "eval/panel_issp")}
EXCL = {"OpinionQA": {"FUD33B_W34", "RACESURV15b_W43"},
        "GSS": {"GSS_discaff", "GSS_discaffm", "GSS_discaffw", "GSS_fehire", "GSS_marhomo"}, "GOQA": set(), "ISSP": set()}
W_PAPER = 0.55
W_GRID = [round(a, 2) for a in np.arange(0.0, 1.0001, 0.05)]


def load_q(conv, name):
    f = f"{conv}/eval_questions.json"
    if not os.path.exists(f):
        return []
    return [q for q in json.load(open(f)) if q["q_id"] not in EXCL[name]]


def load(path, qs):
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    pc = "p" if "p" in d.columns else "option_ratio"
    out = {}
    for q in qs:
        v = d[d.q_id == q["q_id"]].sort_values("option_num")[pc].to_numpy(float)
        if len(v) == len(q["options"]) and v.sum() > 0:
            out[q["q_id"]] = v / v.sum()
    return out


def positions(q):
    k = len(q["options"])
    o = q.get("ordinal")
    if isinstance(o, (list, tuple)) and len(o) == k:
        try:
            x = np.array([float(v) for v in o], dtype=float)
            if np.all(np.isfinite(x)) and len(set(x.tolist())) == k:
                return x
        except (TypeError, ValueError):
            pass
    return np.arange(1, k + 1, dtype=float)


def wd(p, h, x):
    order = np.argsort(x)
    xs, ps, hs = x[order], p[order], h[order]
    return float((np.abs(np.cumsum(ps) - np.cumsum(hs))[:-1] * np.diff(xs)).sum())


def rep(p, h, x):
    span = float(x.max() - x.min())
    return 1.0 - wd(p, h, x) / span if span > 0 else np.nan


def js(p, q):
    e = 1e-12
    p = p + e; p = p / p.sum(); q = q + e; q = q / q.sum(); m = (p + q) / 2
    kl = lambda a, b: float((a * np.log(a / b)).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


DATA = {}
for name, (conv, pdir) in DS.items():
    qs = load_q(conv, name)
    if not qs:
        print(f"{name}: question file missing, skipped")
        continue
    Q = {q["q_id"]: q for q in qs}
    H = load(f"{conv}/human_dist.csv", qs)
    X = {q["q_id"]: positions(q) for q in qs}
    per = {}
    for t in M:
        B = load(f"{pdir}/{t}_base_c10.csv", qs)
        F = load(f"{pdir}/{t}_ft_c10.csv", qs)
        if B is None or F is None:
            continue
        per[t] = dict(B=B, F=F, ids={i for i in Q if i in H and i in B and i in F})
    DATA[name] = dict(Q=Q, H=H, X=X, per=per)
if not DATA:
    sys.exit("no dataset found: run this from <project root>")


def item_metrics(name, ids, get):
    """per-item arrays (JS, Rep, signed NRR bias) for a distribution getter over ids."""
    d = DATA[name]
    out = []
    for i in ids:
        p, h, q = get(i), d["H"][i], d["Q"][i]
        out.append((js(p, h), rep(p, h, d["X"][i]), sum(p[k] for k in q["neg_idx"]) - sum(h[k] for k in q["neg_idx"])))
    return np.array(out)


def summarize(a):
    return dict(JS=a[:, 0].mean(), Rep=a[:, 1].mean(), bias=a[:, 2].mean(), absbias=np.abs(a[:, 2]).mean())


def cond_get(name, i, kind, w, j=None, others=None):
    d = DATA[name]["per"]
    if kind == "base":
        return lambda k: d[i]["B"][k]
    if kind == "nt":
        return lambda k: d[i]["F"][k]
    if kind == "NT":
        return lambda k: w * d[i]["F"][k] + (1 - w) * d[i]["B"][k]
    if kind == "BB":
        return lambda k: w * d[j]["B"][k] + (1 - w) * d[i]["B"][k]
    if kind == "BB6":
        return lambda k: w * np.mean([d[o]["B"][k] for o in others], axis=0) + (1 - w) * d[i]["B"][k]


rows = []   # dataset, model, partner, cond, w, JS, Rep, bias, absbias, n
for name in DATA:
    models = list(DATA[name]["per"])
    allids = set.intersection(*[DATA[name]["per"][t]["ids"] for t in models])
    for i in models:
        for j in models:
            if j == i:
                continue
            ids = sorted(DATA[name]["per"][i]["ids"] & DATA[name]["per"][j]["ids"])
            for w in W_GRID:
                for kind in ("NT", "BB"):
                    a = item_metrics(name, ids, cond_get(name, i, kind, w, j=j))
                    rows.append(dict(dataset=name, model=i, partner=j, cond=kind, w=w, n=len(ids), **summarize(a)))
            for kind in ("base", "nt"):
                a = item_metrics(name, ids, cond_get(name, i, kind, 0))
                rows.append(dict(dataset=name, model=i, partner=j, cond=kind, w=np.nan, n=len(ids), **summarize(a)))
        others = [o for o in models if o != i]
        ids = sorted(set.intersection(*[DATA[name]["per"][t]["ids"] for t in models]))
        for w in W_GRID:
            a = item_metrics(name, ids, cond_get(name, i, "BB6", w, others=others))
            rows.append(dict(dataset=name, model=i, partner="mean6", cond="BB6", w=w, n=len(ids), **summarize(a)))
R = pd.DataFrame(rows)
os.makedirs("eval", exist_ok=True)
R.to_csv("eval/base_base_control.csv", index=False)


def at(df, w):
    return df[np.isclose(df.w, w)]


# ───────────────────────── (1) fixed weight ─────────────────────────
print(f"=== (1) mean over target models, w = {W_PAPER} (JS lower better, Rep higher better, |bias| lower better) ===")
T1 = {}
for name in DATA:
    D = R[R.dataset == name]
    base = D[D.cond == "base"].groupby("model")[["JS", "Rep", "bias", "absbias"]].mean()
    nt = D[D.cond == "nt"].groupby("model")[["JS", "Rep", "bias", "absbias"]].mean()
    NTm = at(D[D.cond == "NT"], W_PAPER).groupby("model")[["JS", "Rep", "bias", "absbias"]].mean()
    bb = at(D[D.cond == "BB"], W_PAPER)
    bb_mean = bb.groupby("model")[["JS", "Rep", "bias", "absbias"]].mean()
    bb_best = bb.groupby("model")["JS"].min()
    bb_worst = bb.groupby("model")["JS"].max()
    bb6 = at(D[D.cond == "BB6"], W_PAPER).groupby("model")[["JS", "Rep", "bias", "absbias"]].mean()
    T1[name] = dict(base=base, nt=nt, NT=NTm, bb=bb_mean, bb6=bb6, bb_best=bb_best, bb_worst=bb_worst)
    print(f"\n  {name}  ({len(base)} models)")
    print(f"    {'condition':34s} {'JS':>8s} {'Rep':>8s} {'bias':>8s} {'|bias|':>8s}")
    for lab, df in [("Base", base), ("NT", nt), ("Base + NT (paper)", NTm), ("Base + Base_j (mean of 6 partners)", bb_mean),
                    ("Base + mean of 6 other Base", bb6)]:
        print(f"    {lab:34s} {df.JS.mean():8.4f} {df.Rep.mean():8.4f} {df.bias.mean():+8.4f} {df.absbias.mean():8.4f}")
    print(f"    {'Base + Base_j, best partner JS':34s} {bb_best.mean():8.4f}")
    print(f"    {'Base + Base_j, worst partner JS':34s} {bb_worst.mean():8.4f}")

# ───────────────────────── (2) wins ─────────────────────────
print(f"\n=== (2) models where Base+NT beats the control, w = {W_PAPER} (JS | Rep) ===")
for name in DATA:
    t = T1[name]
    n = len(t["NT"])
    print(f"  {name:10s} vs mean-BB: JS {(t['NT'].JS < t['bb'].JS).sum()}/{n} | Rep {(t['NT'].Rep > t['bb'].Rep).sum()}/{n}"
          f"   vs mean-of-6: JS {(t['NT'].JS < t['bb6'].JS).sum()}/{n} | Rep {(t['NT'].Rep > t['bb6'].Rep).sum()}/{n}"
          f"   vs best-partner: JS {(t['NT'].JS < t['bb_best']).sum()}/{n}")
    pairs = at(R[(R.dataset == name) & (R.cond == "BB")], W_PAPER).merge(
        at(R[(R.dataset == name) & (R.cond == "NT")], W_PAPER)[["model", "partner", "JS", "Rep"]],
        on=["model", "partner"], suffixes=("_bb", "_nt"))
    print(f"  {'':10s} pairwise (i,j): Base+NT better in JS {(pairs.JS_nt < pairs.JS_bb).sum()}/{len(pairs)}, "
          f"Rep {(pairs.Rep_nt > pairs.Rep_bb).sum()}/{len(pairs)}")

# ───────────────────────── (3) tuned control ─────────────────────────
print("\n=== (3) tuned control: Base+Base weight chosen on OpinionQA (min mean JS over all pairs), applied unchanged ===")
sel = {}
for kind in ("BB", "BB6", "NT"):
    D = R[(R.dataset == "OpinionQA") & (R.cond == kind)]
    g = D.groupby("w").JS.mean()
    sel[kind] = float(g.idxmin())
print(f"  weights selected on OpinionQA: Base+NT {sel['NT']:.2f} | Base+Base {sel['BB']:.2f} | Base+mean6 {sel['BB6']:.2f}")
print(f"  {'dataset':10s} {'metric':6s} {'Base':>8s} {'Base+NT':>9s} {'Base+Base':>10s} {'Base+mean6':>11s}")
for name in DATA:
    D = R[R.dataset == name]
    f = lambda kind: at(D[D.cond == kind], sel[kind]).groupby("model")[["JS", "Rep"]].mean().mean()
    b = D[D.cond == "base"].groupby("model")[["JS", "Rep"]].mean().mean()
    for m in ("JS", "Rep"):
        print(f"  {name:10s} {m:6s} {b[m]:8.4f} {f('NT')[m]:9.4f} {f('BB')[m]:10.4f} {f('BB6')[m]:11.4f}")

# ───────────────────────── (4) oracle ─────────────────────────
print("\n=== (4) oracle control: per model, best weight (and best partner) for JS on the evaluation data itself ===")
print(f"  {'dataset':10s} {'Base+NT oracle w':>17s} {'Base+Base best w, mean partner':>32s} {'Base+Base best w & partner':>28s}")
for name in DATA:
    D = R[R.dataset == name]
    nt_o = D[D.cond == "NT"].groupby(["model", "w"]).JS.mean().groupby("model").min().mean()
    bb_m = D[D.cond == "BB"].groupby(["model", "w"]).JS.mean().groupby("model").min().mean()
    bb_b = D[D.cond == "BB"].groupby("model").JS.min().mean()
    print(f"  {name:10s} {nt_o:17.4f} {bb_m:32.4f} {bb_b:28.4f}")

# ───────────────────────── (5) item-level test ─────────────────────────
print(f"\n=== (5) paired item-level test at w = {W_PAPER}: Base+NT vs Base+Base (per item, averaged over target models and partners) ===")
for name in DATA:
    models = list(DATA[name]["per"])
    ids = sorted(set.intersection(*[DATA[name]["per"][t]["ids"] for t in models]))
    nt_js, bb_js, nt_rp, bb_rp = [], [], [], []
    for i in models:
        a_nt = item_metrics(name, ids, cond_get(name, i, "NT", W_PAPER))
        a_bb = np.mean([item_metrics(name, ids, cond_get(name, i, "BB", W_PAPER, j=j)) for j in models if j != i], axis=0)
        nt_js.append(a_nt[:, 0]); bb_js.append(a_bb[:, 0]); nt_rp.append(a_nt[:, 1]); bb_rp.append(a_bb[:, 1])
    nj, bj, nr, br = (np.mean(x, axis=0) for x in (nt_js, bb_js, nt_rp, bb_rp))
    pj = wilcoxon(nj, bj).pvalue; pr = wilcoxon(nr, br).pvalue
    print(f"  {name:10s} items {len(ids):3d} | JS: NT better on {(nj < bj).mean():.1%} (p={pj:.1e}) | "
          f"Rep: NT better on {(nr > br).mean():.1%} (p={pr:.1e})")
# ───────────────────────── (6) cluster bootstrap ─────────────────────────
print(f"\n=== (6) cluster bootstrap, w = {W_PAPER}: Base+NT minus control, positive favours Base+NT (95% CI, 2000 resamples) ===")
rng = np.random.default_rng(0)
for name in DATA:
    d = DATA[name]
    models = list(d["per"])
    ids = sorted(set.intersection(*[d["per"][t]["ids"] for t in models]))
    diff = {c: {"JS": [], "Rep": []} for c in ("BB", "BB6")}
    for i in models:
        a_nt = item_metrics(name, ids, cond_get(name, i, "NT", W_PAPER))
        a_bb = np.mean([item_metrics(name, ids, cond_get(name, i, "BB", W_PAPER, j=j)) for j in models if j != i], axis=0)
        a_b6 = item_metrics(name, ids, cond_get(name, i, "BB6", W_PAPER, others=[o for o in models if o != i]))
        for c, a in (("BB", a_bb), ("BB6", a_b6)):
            diff[c]["JS"].append(a[:, 0] - a_nt[:, 0])
            diff[c]["Rep"].append(a_nt[:, 1] - a[:, 1])
    cl = {}
    for k, qid in enumerate(ids):
        if name == "OpinionQA":
            m_ = re.search(r"_W(\d+)$", qid); key = f"W{m_.group(1)}" if m_ else qid
        elif name == "ISSP":
            key = d["Q"][qid].get("country", qid)
        else:
            key = qid
        cl.setdefault(key, []).append(k)
    keys = list(cl)
    for c, lab in (("BB", "vs Base+Base_j"), ("BB6", "vs Base+mean6")):
        for m in ("JS", "Rep"):
            v = np.mean(diff[c][m], axis=0)
            point = v.mean()
            boot = [np.mean(np.concatenate([v[cl[keys[k]]] for k in rng.choice(len(keys), len(keys), replace=True)]))
                    for _ in range(2000)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            flag = "*" if lo > 0 else ("-" if hi < 0 else " ")
            print(f"  {name:10s} {lab:15s} {m:4s} clusters {len(keys):3d}  diff {point:+.4f}  CI [{lo:+.4f}, {hi:+.4f}] {flag}")
print("  (* = CI above 0: Base+NT better, - = CI below 0: control better)")
print("\nsaved eval/base_base_control.csv")
