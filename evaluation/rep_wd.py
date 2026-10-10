#!/usr/bin/env python3
"""rep_wd.py -- OpinionQA representativeness (Santurkar et al., 2023) for every variant in the paper.

Run in <project root>. Reads stored panel outputs only (no GPU, no model loading).

Metric (per item q, options mapped to ordinal positions x_1 < ... < x_N, refusals already removed):
    Rep(q) = 1 - WD(p_q, h_q) / (x_N - x_1)
Dataset score = unweighted mean over items; model-set score = mean over the seven models.
Higher is better (1 = identical to humans). JS is printed alongside for comparison (lower is better).

Sections printed:
  (0) ordinal-position diagnostics per dataset
  (1) main table: base / NT / mixture / two instructions / three scalar baselines, Rep and JS side by side
  (2) alpha sweep under Rep: improved pairs and mean Rep; alpha selected on OpinionQA; transfer to the others
  (3) scalar baselines re-selected on OpinionQA under Rep (fair comparison) vs the mixture
  (4) oracle instruction mixture under Rep (0.05 grid) vs the mixture
  (5) cluster bootstrap of mixture minus better component under Rep
  (6) per-model OpinionQA table (LaTeX rows)
Per-model results are also written to eval/rep_wd_results.csv.

Usage:  python3 rep_wd.py 2>&1 | tee eval/rep_wd.txt
"""
import json, os, re, sys
import numpy as np
import pandas as pd

M = ["gemma4-12b", "gemma4-31b", "qwen36-27b", "phi4-14b", "mistral-24b", "gptoss-20b", "exaone-32b"]
NAMES = {"gemma4-12b": "Gemma-4-12B", "gemma4-31b": "Gemma-4-31B", "qwen36-27b": "Qwen3.6-27B", "phi4-14b": "Phi-4-14B",
         "mistral-24b": "Mistral-24B", "gptoss-20b": "gpt-oss-20B", "exaone-32b": "EXAONE-32B"}
DS = {"OpinionQA": ("eval/converted", "eval/panel_h100", "eval/panel_prompt"),
      "GOQA": ("eval/converted_goqa", "eval/panel_goqa", "eval/panel_prompt_goqa"),
      "GSS": ("eval/converted_gss", "eval/panel_gss", "eval/panel_prompt_gss"),
      "ISSP": ("eval/converted_issp", "eval/panel_issp", "eval/panel_prompt_issp")}
EXCL = {"OpinionQA": {"FUD33B_W34", "RACESURV15b_W43"},
        "GSS": {"GSS_discaff", "GSS_discaffm", "GSS_discaffw", "GSS_fehire", "GSS_marhomo"}, "GOQA": set(), "ISSP": set()}
ALPHA_PAPER = 0.55
SCALAR_PAPER = {"shift": 0.08, "temp": 10.75, "tilt": 0.05}          # JS-selected values reported in the paper
A_GRID = [round(a, 2) for a in np.arange(0.0, 1.0001, 0.05)]
GRIDS = {"shift": [round(x, 2) for x in np.arange(0.01, 0.5001, 0.01)],
         "temp": [round(x, 2) for x in np.arange(0.5, 20.0001, 0.25)],
         "tilt": [round(x, 2) for x in np.arange(-2.0, 2.0001, 0.05)]}


# ─────────────────────────────── loading ───────────────────────────────
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
    """Ordinal positions of the options. Uses the question's own ordinal values when they are a numeric list
    aligned with the options (OpinionQA convention); otherwise falls back to 1..N in option order."""
    k = len(q["options"])
    o = q.get("ordinal")
    if isinstance(o, (list, tuple)) and len(o) == k:
        try:
            x = np.array([float(v) for v in o], dtype=float)
            if np.all(np.isfinite(x)) and len(set(x.tolist())) == k:
                return x, "field"
        except (TypeError, ValueError):
            pass
    return np.arange(1, k + 1, dtype=float), "index"


# ─────────────────────────────── metrics ───────────────────────────────
def wd(p, h, x):
    """1-Wasserstein distance between two discrete distributions on positions x."""
    order = np.argsort(x)
    xs, ps, hs = x[order], p[order], h[order]
    cdf_gap = np.abs(np.cumsum(ps) - np.cumsum(hs))[:-1]
    return float((cdf_gap * np.diff(xs)).sum())


def rep(p, h, x):
    span = float(x.max() - x.min())
    return 1.0 - wd(p, h, x) / span if span > 0 else np.nan


def js(p, q):
    e = 1e-12
    p = p + e; p = p / p.sum(); q = q + e; q = q / q.sum(); m = (p + q) / 2
    kl = lambda a, b: float((a * np.log(a / b)).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def score(k, neg):
    sc = np.linspace(1, -1, k)
    return sc if 0 in neg else sc[::-1]


def adjust(kind, p, q, v):
    neg = q["neg_idx"]
    if kind == "shift":
        r = p.copy()
        for i in neg:
            r[i] = max(r[i] + v, 1e-9)
        return r / r.sum()
    if kind == "temp":
        r = np.power(np.clip(p, 1e-9, None), 1.0 / v)
        return r / r.sum()
    r = p * np.exp(v * score(len(p), neg))                                   # tilt
    return r / r.sum()


# ─────────────────────────────── data ───────────────────────────────
DATA = {}
print("=== (0) ordinal positions per dataset ===")
for name, (conv, pdir, ppdir) in DS.items():
    qs = load_q(conv, name)
    if not qs:
        print(f"  {name:10s} question file missing, skipped")
        continue
    Q = {q["q_id"]: q for q in qs}
    X, src = {}, {"field": 0, "index": 0}
    for q in qs:
        X[q["q_id"]], s = positions(q)
        src[s] += 1
    H = load(f"{conv}/human_dist.csv", qs)
    ex = qs[0].get("ordinal")
    print(f"  {name:10s} items {len(qs):3d} | positions from 'ordinal' field: {src['field']:3d}, from option index: {src['index']:3d}"
          f" | example ordinal value: {str(ex)[:40]}")
    per = {}
    for t in M:
        B = load(f"{pdir}/{t}_base_c10.csv", qs)
        F = load(f"{pdir}/{t}_ft_c10.csv", qs)
        if B is None or F is None:
            continue
        A = load(f"{ppdir}/{t}_phonest_c10.csv", qs)
        S = load(f"{ppdir}/{t}_pstrong_c10.csv", qs)
        ids = [i for i in Q if i in H and i in B and i in F]
        per[t] = dict(B=B, F=F, A=A, S=S, ids=ids)
    DATA[name] = dict(Q=Q, X=X, H=H, per=per)

if not DATA:
    sys.exit("no dataset found: run this from <project root>")


def mean_rep(name, t, get, ids=None):
    d = DATA[name]; pt = d["per"][t]
    ids = ids or pt["ids"]
    return float(np.mean([rep(get(i), d["H"][i], d["X"][i]) for i in ids]))


def mean_js(name, t, get, ids=None):
    d = DATA[name]; pt = d["per"][t]
    ids = ids or pt["ids"]
    return float(np.mean([js(get(i), d["H"][i]) for i in ids]))


def getter(name, t, kind, v=None):
    d = DATA[name]; pt = d["per"][t]
    if kind == "base":
        return lambda i: pt["B"][i]
    if kind == "nt":
        return lambda i: pt["F"][i]
    if kind == "mix":
        return lambda i: v * pt["F"][i] + (1 - v) * pt["B"][i]
    if kind in ("anti", "inten"):
        src = pt["A"] if kind == "anti" else pt["S"]
        return (lambda i: src[i]) if src is not None else None
    return lambda i: adjust(kind, pt["B"][i], d["Q"][i], v)


def instr_ids(name, t):
    pt = DATA[name]["per"][t]
    if pt["A"] is None or pt["S"] is None:
        return None
    return [i for i in pt["ids"] if i in pt["A"] and i in pt["S"]]


rows = []

# ─────────────────────────── (1) main table ───────────────────────────
print("\n=== (1) mean over seven models: Rep (higher better) | JS (lower better) ===")
variants = [("base", None), ("nt", None), ("mix", ALPHA_PAPER), ("anti", None), ("inten", None),
            ("shift", SCALAR_PAPER["shift"]), ("temp", SCALAR_PAPER["temp"]), ("tilt", SCALAR_PAPER["tilt"])]
label = {"base": "Base", "nt": "NT", "mix": "Mixture a=0.55", "anti": "Anti-attenuation", "inten": "Intensity-emph.",
         "shift": "Shift d=0.08", "temp": "Temperature T=10.75", "tilt": "Tilt l=0.05"}
MAIN = {}
for name in DATA:
    print(f"\n  {name}")
    print(f"    {'variant':22s} {'Rep':>8s} {'JS':>8s}")
    for kind, v in variants:
        reps, jss = [], []
        for t in DATA[name]["per"]:
            g = getter(name, t, kind, v)
            if g is None:
                continue
            ids = instr_ids(name, t) if kind in ("anti", "inten") else None
            if kind in ("anti", "inten") and not ids:
                continue
            r, j = mean_rep(name, t, g, ids), mean_js(name, t, g, ids)
            reps.append(r); jss.append(j)
            rows.append(dict(dataset=name, model=t, variant=label[kind], rep=r, js=j))
        if reps:
            MAIN[(name, kind)] = (np.mean(reps), np.mean(jss))
            print(f"    {label[kind]:22s} {np.mean(reps):8.4f} {np.mean(jss):8.4f}")

print("\n  best variant per dataset")
for name in DATA:
    cand = {k: MAIN[(name, k)] for k, _ in variants if (name, k) in MAIN}
    br = max(cand, key=lambda k: cand[k][0]); bj = min(cand, key=lambda k: cand[k][1])
    print(f"    {name:10s} by Rep: {label[br]:22s} by JS: {label[bj]}")

print("\n  models for which the mixture beats each alternative, under Rep (and under JS)")
for name in DATA:
    out = []
    for kind, v in variants:
        if kind == "mix":
            continue
        wr = wj = n = 0
        for t in DATA[name]["per"]:
            g = getter(name, t, kind, v)
            if g is None:
                continue
            ids = instr_ids(name, t) if kind in ("anti", "inten") else DATA[name]["per"][t]["ids"]
            if not ids:
                continue
            gm = getter(name, t, "mix", ALPHA_PAPER)
            wr += mean_rep(name, t, gm, ids) > mean_rep(name, t, g, ids)
            wj += mean_js(name, t, gm, ids) < mean_js(name, t, g, ids)
            n += 1
        out.append(f"{label[kind].split()[0]} {wr}/{n} ({wj}/{n})")
    print(f"    {name:10s} " + " | ".join(out))

# ─────────────────────────── (2) alpha sweep ───────────────────────────
print("\n=== (2) alpha sweep under Rep ===")
curve = {}
for name in DATA:
    for t in DATA[name]["per"]:
        curve[(name, t)] = {a: mean_rep(name, t, getter(name, t, "mix", a)) for a in A_GRID}
pairs = list(curve)
print(f"  {'alpha':>6s} {'improved':>10s} {'meanRep':>9s}")
for a in A_GRID:
    if a in (0.0, 1.0):
        continue
    win = sum(curve[p][a] > max(curve[p][0.0], curve[p][1.0]) for p in pairs)
    print(f"  {a:6.2f} {win:5d}/{len(pairs):<4d} {np.mean([curve[p][a] for p in pairs]):9.4f}")
oqa = [p for p in pairs if p[0] == "OpinionQA"]
a_rep = max([a for a in A_GRID if 0 < a < 1], key=lambda a: np.mean([curve[p][a] for p in oqa]))
print(f"\n  alpha selected on OpinionQA under Rep: {a_rep:.2f}  (paper, selected under JS: {ALPHA_PAPER:.2f})")
for a in sorted({a_rep, ALPHA_PAPER}):
    for name in DATA:
        ps = [p for p in pairs if p[0] == name]
        win = sum(curve[p][a] > max(curve[p][0.0], curve[p][1.0]) for p in ps)
        print(f"    alpha={a:.2f}  {name:10s} mixture above both components for {win}/{len(ps)} models")

# ─────────────────────────── (3) scalar baselines re-selected ───────────────────────────
print("\n=== (3) scalar baselines re-selected on OpinionQA under Rep ===")
sel = {}
for kind, grid in GRIDS.items():
    best = max(grid, key=lambda v: np.mean([mean_rep("OpinionQA", t, getter("OpinionQA", t, kind, v)) for t in DATA["OpinionQA"]["per"]]))
    sel[kind] = best
    edge = " (grid edge)" if best in (grid[0], grid[-1]) else ""
    print(f"  {kind:6s} selected {best}{edge}")
print(f"\n  {'dataset':10s} {'mix a_rep':>10s} {'shift':>8s} {'temp':>8s} {'tilt':>8s}   models where mixture is higher")
for name in DATA:
    ms = [mean_rep(name, t, getter(name, t, "mix", a_rep)) for t in DATA[name]["per"]]
    cells, wins = [], []
    for kind in GRIDS:
        vs = [mean_rep(name, t, getter(name, t, kind, sel[kind])) for t in DATA[name]["per"]]
        cells.append(np.mean(vs)); wins.append(f"{kind} {sum(m > v for m, v in zip(ms, vs))}/{len(ms)}")
    print(f"  {name:10s} {np.mean(ms):10.4f} " + " ".join(f"{c:8.4f}" for c in cells) + "   " + ", ".join(wins))

# ─────────────────────────── (4) oracle instruction mixture ───────────────────────────
print("\n=== (4) oracle instruction mixture under Rep (per-model best variant and alpha, 0.05 grid) ===")
tot = won = 0
for name in DATA:
    orc, mix, w, n = [], [], 0, 0
    for t in DATA[name]["per"]:
        ids = instr_ids(name, t)
        if not ids:
            continue
        pt = DATA[name]["per"][t]
        best = max(mean_rep(name, t, (lambda V, a: (lambda i: a * V[i] + (1 - a) * pt["B"][i]))(V, a), ids)
                   for V in (pt["A"], pt["S"]) for a in A_GRID if 0 < a < 1)
        m = mean_rep(name, t, getter(name, t, "mix", ALPHA_PAPER), ids)
        orc.append(best); mix.append(m); w += m > best; n += 1
    if n:
        tot += n; won += w
        print(f"  {name:10s} mixture {np.mean(mix):.4f} | oracle instruction {np.mean(orc):.4f} | mixture higher {w}/{n}")
print(f"  total: mixture higher in {won}/{tot} pairs")

# ─────────────────────────── (5) cluster bootstrap ───────────────────────────
print("\n=== (5) cluster bootstrap: mixture minus better component, Rep (positive favours the mixture) ===")
rng = np.random.default_rng(0)
for name in DATA:
    d = DATA[name]; per_item = {}
    for t in d["per"]:
        pt = d["per"][t]; ids = pt["ids"]
        rb = mean_rep(name, t, getter(name, t, "base")); rf = mean_rep(name, t, getter(name, t, "nt"))
        better = "B" if rb >= rf else "F"
        for i in ids:
            mv = rep(ALPHA_PAPER * pt["F"][i] + (1 - ALPHA_PAPER) * pt["B"][i], d["H"][i], d["X"][i])
            per_item.setdefault(i, []).append(mv - rep(pt[better][i], d["H"][i], d["X"][i]))
    vals = {i: np.mean(v) for i, v in per_item.items()}
    cl = {}
    for i in vals:
        if name == "OpinionQA":
            m_ = re.search(r"_W(\d+)$", i); key = f"W{m_.group(1)}" if m_ else i
        elif name == "ISSP":
            key = d["Q"][i].get("country", i)
        else:
            key = i
        cl.setdefault(key, []).append(i)
    keys = list(cl); point = np.mean(list(vals.values()))
    boot = [np.mean([vals[i] for k in rng.choice(len(keys), len(keys), replace=True) for i in cl[keys[k]]]) for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  {name:10s} clusters {len(keys):3d}  difference {point:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")

# ─────────────────────────── (6) per-model OpinionQA table ───────────────────────────
print("\n=== (6) OpinionQA per model, Rep (LaTeX rows: base & NT & mixture & temperature) ===")
for t in DATA["OpinionQA"]["per"]:
    vals = [mean_rep("OpinionQA", t, getter("OpinionQA", t, k, v)) for k, v in
            [("base", None), ("nt", None), ("mix", ALPHA_PAPER), ("temp", SCALAR_PAPER["temp"])]]
    print(f"  {NAMES[t]} & " + " & ".join(f"{v:.3f}" for v in vals) + r" \\")

os.makedirs("eval", exist_ok=True)
pd.DataFrame(rows).to_csv("eval/rep_wd_results.csv", index=False)
print("\nsaved eval/rep_wd_results.csv")
