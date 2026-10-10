#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_final.py — Table 1 확정 분석
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
(a) 과신 지표: 문항별 max prob 평균 / 정규화 엔트로피 (base vs ft vs human)
(b) wave 홀짝 dev/test 분할 → dev에서 α* 선택(2기준) → test 성능 보고
    - α*_bias : dev |NRR bias| 최소
    - α*_JS   : dev JS 최소
(c) family별 NRR bias 표 (n 병기, base/ft)
(d) 페어드 유의성 (test 문항, ft vs base)

실행:
  python3 analyze_final.py \
      --questions converted/eval_questions.json \
      --human converted/human_dist.csv \
      --base dist_base.csv --ft dist_ft.csv
"""
import os, json, argparse
import numpy as np
import pandas as pd

def load_dist(path, questions):
    df = pd.read_csv(os.path.expanduser(path))
    pcol = "p" if "p" in df.columns else "option_ratio"
    out = {}
    for q in questions:
        sub = df[df.q_id == q["q_id"]].sort_values("option_num")
        if len(sub) != len(q["options"]):
            continue
        v = sub[pcol].to_numpy(dtype=float)
        s = v.sum()
        out[q["q_id"]] = v / s if s > 0 else v
    return out

def js_div(p, q, eps=1e-12):
    p = np.asarray(p) + eps; p /= p.sum()
    q = np.asarray(q) + eps; q /= q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def wd1(p, q, positions):
    p = np.asarray(p, float); q = np.asarray(q, float)
    pos = np.asarray(positions, float)
    order = np.argsort(pos)
    pos, p, q = pos[order], p[order], q[order]
    span = pos[-1] - pos[0]
    if span <= 0 or len(p) <= 1:
        return 0.0
    return float(np.sum(np.abs(np.cumsum(p - q)[:-1]) * np.diff(pos)) / span)

def nrr(p, neg_idx):
    return float(sum(p[i] for i in neg_idx))

def norm_entropy(p):
    p = np.asarray(p, float) + 1e-12
    p /= p.sum()
    K = len(p)
    if K <= 1:
        return 1.0
    return float(-(p * np.log(p)).sum() / np.log(K))

def eval_set(questions, human, model):
    rows = []
    for q in questions:
        qid = q["q_id"]
        if qid not in human or qid not in model:
            continue
        h, mdl = human[qid], model[qid]
        rows.append({
            "q_id": qid, "family": q["family"], "wave": q["table_num"],
            "bias": nrr(mdl, q["neg_idx"]) - nrr(h, q["neg_idx"]),
            "JS": js_div(mdl, h), "WD": wd1(mdl, h, q["ordinal"]),
        })
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--human", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--ft", required=True)
    args = ap.parse_args()

    questions = json.load(open(os.path.expanduser(args.questions)))
    human = load_dist(args.human, questions)
    base = load_dist(args.base, questions)
    ft = load_dist(args.ft, questions)
    qs = [q for q in questions if q["q_id"] in human
          and q["q_id"] in base and q["q_id"] in ft]
    print(f"[로드] 공통 문항 {len(qs)}")

    # (a) 과신 지표
    print("\n== (a) 과신 지표 (문항 평균) ==")
    for name, dist in [("human", human), ("base", base), ("ft", ft)]:
        mx = np.mean([dist[q['q_id']].max() for q in qs])
        en = np.mean([norm_entropy(dist[q['q_id']]) for q in qs])
        print(f"  {name:6s}: mean max-prob {mx:.3f} | 정규화 엔트로피 {en:.3f}")

    # (b) dev/test 분할 + α 선택
    dev_q  = [q for q in qs if q["table_num"] % 2 == 1]
    test_q = [q for q in qs if q["table_num"] % 2 == 0]
    print(f"\n== (b) wave 홀짝 분할: dev {len(dev_q)}문항 / test {len(test_q)}문항 ==")

    alphas = [round(a, 2) for a in np.arange(0.05, 1.0, 0.05)]
    def ens(a):
        return {qid: a * ft[qid] + (1 - a) * base[qid] for qid in ft if qid in base}

    dev_stats = []
    for a in alphas:
        d = eval_set(dev_q, human, ens(a))
        dev_stats.append((a, abs(d.bias.mean()), d.JS.mean(), d.WD.mean()))
    a_bias = min(dev_stats, key=lambda x: x[1])[0]
    a_js   = min(dev_stats, key=lambda x: x[2])[0]
    print(f"  dev 기준 α*_bias = {a_bias} (|bias| 최소) / α*_JS = {a_js} (JS 최소)")

    print(f"\n  [test 성능 — 논문 보고용]")
    header = f"  {'model':14s}{'NRR_bias':>10s}{'|bias|':>9s}{'JS':>9s}{'WD':>9s}"
    print(header)
    rows_for_sig = {}
    for tag, dist in [("base", base), ("ft", ft),
                      (f"ens(a={a_bias})", ens(a_bias)),
                      (f"ens(a={a_js})", ens(a_js))]:
        t = eval_set(test_q, human, dist)
        rows_for_sig[tag] = t
        print(f"  {tag:14s}{t.bias.mean():>10.4f}{t.bias.abs().mean():>9.4f}"
              f"{t.JS.mean():>9.4f}{t.WD.mean():>9.4f}")

    # (d) 페어드 유의성 (test)
    try:
        from scipy.stats import wilcoxon
        b = rows_for_sig["base"].set_index("q_id")
        f = rows_for_sig["ft"].set_index("q_id")
        common = b.index.intersection(f.index)
        print(f"\n== (d) 페어드 유의성 (test {len(common)}문항, ft vs base) ==")
        for col in ["JS", "WD"]:
            s, p = wilcoxon(b.loc[common, col], f.loc[common, col])
            print(f"  {col}: p={p:.2e} | 개선 비율 "
                  f"{(f.loc[common, col] < b.loc[common, col]).mean():.1%}")
    except ImportError:
        print("\n  (scipy 없음 — 유의성 생략)")

    # (c) family 표 (전체 문항 기준)
    print("\n== (c) family별 NRR bias (n 병기) ==")
    fb = eval_set(qs, human, base).rename(columns={"bias": "base"})
    ff = eval_set(qs, human, ft)[["q_id", "bias"]].rename(columns={"bias": "ft"})
    fam = fb.merge(ff, on="q_id")
    g = fam.groupby("family").agg(n=("q_id", "size"),
                                  base=("base", "mean"), ft=("ft", "mean"))
    g = g.sort_values("base")
    print(g.round(3).to_string())
    print("\n  * 판단-부정 회피(base 큰 음수) vs 우려 과잉표출(base 큰 양수) 구분 확인용")

if __name__ == "__main__":
    main()
