#!/usr/bin/env python3
"""item_level.py -- item-level view of the calibration on OpinionQA, and a check of whether the adapter
learned option positions rather than negative content. Stored outputs only (no GPU).

Run in <project root>:
    python3 item_level.py 2>&1 | tee eval/item_level.txt

Writes
    eval/item_level_oqa.csv            one row per item: group, pole position, human / base / NT / mixture
                                       negative share, per-item JS and Rep (means over the seven models)
    eval/item_level_tables.tex         LaTeX rows for the most improved and most worsened items
    figures/item_scatter.pdf           every item: distance to humans before (Base) and after (mixture)
    figures/item_dumbbell.pdf          selected items: human, Base and mixture negative share side by side
    figures/position_check.pdf         (a) change in front-half mass per model, (b) change in negative share
                                       by where the negative pole sits

Logic of the position check
    The synthetic training questionnaires fix option 1 as the most negative stance, and the negative
    personas answer mostly 1-2. An adapter that memorised positions would therefore move answers toward
    the FRONT of the option list. On OpinionQA the negative pole sits at the BACK for most items, so a
    position-memorising adapter would move toward positive answers there. A move toward the back, with
    negative share rising where the negative pole is at the back, follows content rather than the
    training positions. A generic shift toward later options is not excluded by this check; the
    reversed-order evaluation (make_reversed_eval.py) separates it.
"""
import json, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

M = ["gemma4-12b", "gemma4-31b", "qwen36-27b", "phi4-14b", "mistral-24b", "gptoss-20b", "exaone-32b"]
NAMES = {"gemma4-12b": "Gemma-4-12B", "gemma4-31b": "Gemma-4-31B", "qwen36-27b": "Qwen3.6-27B", "phi4-14b": "Phi-4-14B",
         "mistral-24b": "Mistral-24B", "gptoss-20b": "gpt-oss-20B", "exaone-32b": "EXAONE-32B"}
CONV, PDIR = "eval/converted", "eval/panel_h100"
EXCL = {"FUD33B_W34", "RACESURV15b_W43"}
ALPHA = 0.55
CONCERN = {'WORRYD_W26', 'WORRYF_W26', 'WORRYB_W26', 'WORRYE_W26', 'WORRYBILL_W29', 'CONCERNGRPa_W49', 'CONCERNGRPc_W49',
           'HOMEASSIST2_W49', 'CONCERNCO_W49', 'GAP21Q24_W82', 'FTRWORRYc_W41', 'FTRWORRYe_W41', 'FTRWORRYf_W41'}
NORM = {'GROWUPVIOL_W26', 'HARASS1F1d_W41', 'HARASS1NOWRKF2d_W41', 'HARASS1F1b_W41', 'RQ5_F1B_W42', 'RQ5_F1C_W42',
        'RACESURV47a_W43', 'RACESURV47b_W43', 'RACESURV47d_W43', 'RACESURV47e_W43', 'RACESURV47f_W43', 'GAP21Q23_W82',
        'NEWSPROBd_W45', 'GUNSAFE_W26', 'GAP21Q41_W82'}
GROUP_LABEL = {"unchanged": "Target-directed + mixed", "concern": "Concern-oriented", "norm": "Norm-oriented"}
SLATE, SAGE, BRICK, GREY, INK = "#4C6E91", "#5E9A82", "#B0504F", "#8A94A0", "#222222"
GCOL = {"unchanged": SLATE, "concern": BRICK, "norm": SAGE}
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
                     "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})


def load(path, qs):
    d = pd.read_csv(path); pc = "p" if "p" in d.columns else "option_ratio"; out = {}
    for q in qs:
        v = d[d.q_id == q["q_id"]].sort_values("option_num")[pc].to_numpy(float)
        if len(v) == len(q["options"]) and v.sum() > 0:
            out[q["q_id"]] = v / v.sum()
    return out


def positions(q):
    k = len(q["options"]); o = q.get("ordinal")
    if isinstance(o, (list, tuple)) and len(o) == k:
        try:
            x = np.array([float(v) for v in o])
            if np.all(np.isfinite(x)) and len(set(x.tolist())) == k:
                return x
        except (TypeError, ValueError):
            pass
    return np.arange(1, k + 1, dtype=float)


def js(p, q):
    e = 1e-12; p = p + e; p = p / p.sum(); q = q + e; q = q / q.sum(); m = (p + q) / 2
    return 0.5 * float((p * np.log(p / m)).sum()) + 0.5 * float((q * np.log(q / m)).sum())


def rep(p, h, x):
    o = np.argsort(x); xs, ps, hs = x[o], p[o], h[o]
    return 1.0 - float((np.abs(np.cumsum(ps) - np.cumsum(hs))[:-1] * np.diff(xs)).sum()) / float(xs.max() - xs.min())


def nrr(p, neg): return float(sum(p[i] for i in neg))
def front(p): return float(p[:max(1, len(p) // 2)].sum())        # mass on the first half of the option list


qs = [q for q in json.load(open(f"{CONV}/eval_questions.json")) if q["q_id"] not in EXCL]
Q = {q["q_id"]: q for q in qs}
H = load(f"{CONV}/human_dist.csv", qs)
B, F = {}, {}
for t in M:
    b, f = f"{PDIR}/{t}_base_c10.csv", f"{PDIR}/{t}_ft_c10.csv"
    if os.path.exists(b) and os.path.exists(f):
        B[t], F[t] = load(b, qs), load(f, qs)
models = list(B)
print(f"items {len(qs)} | models {len(models)}")

rows = []
for qid, q in Q.items():
    if qid not in H: continue
    ts = [t for t in models if qid in B[t] and qid in F[t]]
    if not ts: continue
    neg = q["neg_idx"]; x = positions(q); h = H[qid]
    grp = "concern" if qid in CONCERN else "norm" if qid in NORM else "unchanged"
    pole = "front" if 0 in neg else "back"
    mix = {t: ALPHA * F[t][qid] + (1 - ALPHA) * B[t][qid] for t in ts}
    r = dict(q_id=qid, group=grp, pole=pole, n_models=len(ts), question=str(q.get("question", "")).replace("\n", " "),
             nrr_human=nrr(h, neg),
             nrr_base=np.mean([nrr(B[t][qid], neg) for t in ts]),
             nrr_nt=np.mean([nrr(F[t][qid], neg) for t in ts]),
             nrr_mix=np.mean([nrr(mix[t], neg) for t in ts]),
             js_base=np.mean([js(B[t][qid], h) for t in ts]), js_mix=np.mean([js(mix[t], h) for t in ts]),
             rep_base=np.mean([rep(B[t][qid], h, x) for t in ts]), rep_mix=np.mean([rep(mix[t], h, x) for t in ts]),
             front_base=np.mean([front(B[t][qid]) for t in ts]), front_nt=np.mean([front(F[t][qid]) for t in ts]))
    r["dist_base"] = abs(r["nrr_base"] - r["nrr_human"]); r["dist_mix"] = abs(r["nrr_mix"] - r["nrr_human"])
    r["d_dist"] = r["dist_mix"] - r["dist_base"]          # negative = negative share moved closer to humans
    r["d_rep"] = r["rep_mix"] - r["rep_base"]               # positive = closer under Rep
    r["d_js"] = r["js_mix"] - r["js_base"]                  # negative = closer under JS
    rows.append(r)
D = pd.DataFrame(rows)
os.makedirs("eval", exist_ok=True); os.makedirs("figures", exist_ok=True)
D.to_csv("eval/item_level_oqa.csv", index=False)

# ── summary ─────────────────────────────────────────────────────────────────
print("\n=== items whose mixture is closer to humans than Base (model means) ===")
print(f"  {'group':24s} {'pole':6s} {'n':>3s} {'NRR closer':>11s} {'Rep higher':>11s} {'JS lower':>9s}")
for grp in ["unchanged", "concern", "norm"]:
    for pole in ["back", "front"]:
        g = D[(D.group == grp) & (D.pole == pole)]
        if len(g) == 0: continue
        print(f"  {GROUP_LABEL[grp]:24s} {pole:6s} {len(g):3d} {int((g.d_dist < 0).sum()):5d}/{len(g):<5d} "
              f"{int((g.d_rep > 0).sum()):5d}/{len(g):<5d} {int((g.d_js < 0).sum()):3d}/{len(g)}")
tot = len(D)
print(f"  {'all':24s} {'':6s} {tot:3d} {int((D.d_dist < 0).sum()):5d}/{tot:<5d} {int((D.d_rep > 0).sum()):5d}/{tot:<5d} {int((D.d_js < 0).sum()):3d}/{tot}")

print("\n=== position check: change in front-half mass, NT minus Base (training negatives sit at the front) ===")
pm = []
for t in models:
    ids = [i for i in D.q_id if i in B[t] and i in F[t]]
    fb = np.mean([front(B[t][i]) for i in ids]); ff = np.mean([front(F[t][i]) for i in ids])
    pm.append((t, fb, ff)); print(f"  {NAMES[t]:12s} {fb:.3f} -> {ff:.3f}  ({ff - fb:+.3f})")
print("  a position-memorising adapter would raise front-half mass; a negative change goes the other way")

print("\n=== change in negative share, NT minus Base, by where the negative pole sits ===")
for grp in ["unchanged", "concern", "norm", None]:
    for pole in ["back", "front"]:
        g = D[(D.pole == pole) & ((D.group == grp) if grp else True)]
        if len(g) == 0: continue
        d = g.nrr_nt - g.nrr_base
        print(f"  {(GROUP_LABEL[grp] if grp else 'all items'):24s} pole={pole:5s} n={len(g):3d}  mean {d.mean():+.3f}  rises on {int((d > 0).sum())}/{len(g)}")

# ── LaTeX rows ──────────────────────────────────────────────────────────────
def esc(s): return s.replace("\\", " ").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#").replace("$", "\\$")
with open("eval/item_level_tables.tex", "w", encoding="utf-8") as fo:
    for title, sub in [("most improved (Rep)", D.sort_values("d_rep", ascending=False).head(8)),
                       ("most worsened (Rep)", D.sort_values("d_rep").head(5))]:
        fo.write(f"% {title}: item & group & pole & human NRR & Base NRR & mixture NRR & dRep\n")
        for _, r in sub.iterrows():
            fo.write(f"\\texttt{{{esc(r.q_id)}}} & {GROUP_LABEL[r.group]} & {r.pole} & {r.nrr_human:.2f} & {r.nrr_base:.2f} & "
                     f"{r.nrr_mix:.2f} & {r.d_rep:+.3f} \\\\  % {esc(r.question[:90])}\n")
print("\nsaved eval/item_level_oqa.csv, eval/item_level_tables.tex")

# ── figure 1: every item, distance to humans before and after ───────────────
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
for ax, (bcol, mcol, lab, better) in zip(axes, [("dist_base", "dist_mix", "|negative share − human|", "below"),
                                                 ("rep_base", "rep_mix", "Rep", "above")]):
    for grp in ["unchanged", "concern", "norm"]:
        for pole, mk in [("back", "o"), ("front", "^")]:
            g = D[(D.group == grp) & (D.pole == pole)]
            ax.scatter(g[bcol], g[mcol], s=16, marker=mk, color=GCOL[grp], alpha=0.8, linewidths=0,
                       label=f"{GROUP_LABEL[grp]}, pole {pole}" if ax is axes[0] else None)
    lo = min(D[bcol].min(), D[mcol].min()); hi = max(D[bcol].max(), D[mcol].max()); pad = 0.03 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=GREY, lw=0.7, ls="--")
    ax.set_xlabel(f"Base: {lab}"); ax.set_ylabel(f"Mixture: {lab}")
    ax.set_title(f"({'a' if ax is axes[0] else 'b'}) {lab}; {better} the diagonal = closer", fontsize=8.5, loc="left")
axes[0].legend(fontsize=6.2, frameon=False, loc="upper left", handletextpad=0.2)
fig.tight_layout(); fig.savefig("figures/item_scatter.pdf", bbox_inches="tight"); plt.close(fig)

# ── figure 2: selected items, human vs Base vs mixture negative share ───────
sel = pd.concat([D[D.group == "concern"].sort_values("nrr_human"),
                 D[D.group == "norm"].sort_values("nrr_human"),
                 D[D.group == "unchanged"].sort_values("d_rep", ascending=False).head(12).sort_values("nrr_human")])
fig, ax = plt.subplots(figsize=(6.6, 0.16 * len(sel) + 0.8))
y = np.arange(len(sel))[::-1]
for yi, (_, r) in zip(y, sel.iterrows()):
    ax.plot([r.nrr_base, r.nrr_mix], [yi, yi], color=GREY, lw=0.8, zorder=1)
    ax.scatter(r.nrr_base, yi, s=14, color=GREY, zorder=2)
    ax.scatter(r.nrr_mix, yi, s=14, color=BRICK, zorder=3)
    ax.scatter(r.nrr_human, yi, s=28, marker="|", color=INK, zorder=4)
ax.set_yticks(y); ax.set_yticklabels([f"{r.q_id} ({'F' if r.pole == 'front' else 'B'})" for _, r in sel.iterrows()], fontsize=5.6)
ax.set_xlabel("negative-option share (mean over models)"); ax.set_xlim(0, 1)
ax.scatter([], [], s=14, color=GREY, label="Base"); ax.scatter([], [], s=14, color=BRICK, label="Mixture")
ax.scatter([], [], s=28, marker="|", color=INK, label="Human")
ax.legend(fontsize=6.5, frameon=False, ncol=3, loc="lower right")
n1 = (sel.group == "concern").sum(); n2 = (sel.group == "norm").sum()
for yy, txt in [(len(sel) - n1 / 2, "concern"), (len(sel) - n1 - n2 / 2, "norm"), ((len(sel) - n1 - n2) / 2, "target+mixed (top 12 by ΔRep)")]:
    ax.text(1.01, yy - 0.5, txt, fontsize=6.3, va="center", transform=ax.get_yaxis_transform())
for b in [len(sel) - n1 - 0.5, len(sel) - n1 - n2 - 0.5]:
    ax.axhline(b, color="#E6E8EA", lw=0.6)
fig.tight_layout(); fig.savefig("figures/item_dumbbell.pdf", bbox_inches="tight"); plt.close(fig)

# ── figure 3: position check ────────────────────────────────────────────────
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.5), gridspec_kw={"width_ratios": [1.15, 1]})
xs = np.arange(len(pm)); dv = [ff - fb for _, fb, ff in pm]
a1.bar(xs, dv, color=[SLATE if v < 0 else BRICK for v in dv], width=0.6)
a1.axhline(0, color=INK, lw=0.6)
a1.set_xticks(xs); a1.set_xticklabels([NAMES[t] for t, _, _ in pm], rotation=35, ha="right", fontsize=6.8)
a1.set_ylabel("Δ front-half mass (NT − Base)")
a1.set_title("(a) Position memorised from training\n     would make these positive", fontsize=8.2, loc="left")
dd = D.assign(delta=D.nrr_nt - D.nrr_base)
groups = [("back", "unchanged"), ("front", "unchanged"), ("back", "concern/norm"), ("front", "concern/norm")]
vals, labs = [], []
for pole, kind in groups:
    g = dd[(dd.pole == pole) & ((dd.group == "unchanged") if kind == "unchanged" else (dd.group != "unchanged"))]
    vals.append(g.delta.to_numpy()); labs.append(f"{kind}\npole {pole}\n(n={len(g)})")
a2.boxplot(vals, widths=0.5, showfliers=False, medianprops=dict(color=BRICK))
for i, v in enumerate(vals, 1):
    a2.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(v)), v, s=6, color=GREY, alpha=0.7, zorder=3)
a2.axhline(0, color=INK, lw=0.6); a2.set_xticks(range(1, len(labs) + 1)); a2.set_xticklabels(labs, fontsize=6.3)
a2.set_ylabel("Δ negative share (NT − Base)"); a2.set_title("(b) By position of the negative pole", fontsize=8.2, loc="left")
fig.tight_layout(); fig.savefig("figures/position_check.pdf", bbox_inches="tight"); plt.close(fig)
print("saved figures/item_scatter.pdf, figures/item_dumbbell.pdf, figures/position_check.pdf")
