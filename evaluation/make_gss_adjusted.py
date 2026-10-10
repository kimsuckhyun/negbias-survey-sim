#!/usr/bin/env python3
"""GSS 조정 NRR — 반전 4 / 제외 5 / 유지 23
   사용: cd <project root> && python3 scripts/make_gss_adjusted.py"""
import pandas as pd, json, numpy as np, glob, re, os

FLIP = {"GSS_fefam", "GSS_fepol", "GSS_fepresch", "GSS_spanking"}   # 차별·체벌 진술 거부 = 안전한 부정
EXCL = {"GSS_discaff", "GSS_discaffm", "GSS_discaffw",              # 확률 척도(부정 극 미정의)
        "GSS_fehire", "GSS_marhomo"}                                # 중립지 'Neither'가 NEG로 오라벨

qs = json.load(open('eval/converted_gss/eval_questions.json'))
def load(f):
    d = pd.read_csv(f); pc = 'p' if 'p' in d.columns else 'option_ratio'; out = {}
    for q in qs:
        s = d[d.q_id == q['q_id']].sort_values('option_num')[pc].to_numpy(float)
        if len(s) == len(q['options']) and s.sum() > 0: out[q['q_id']] = s/s.sum()
    return out
def js(p, q):
    e=1e-12; p=p+e; p=p/p.sum(); q=q+e; q=q/q.sum(); m=(p+q)/2
    kl=lambda a,b: float((a*np.log(a/b)).sum()); return .5*kl(p,m)+.5*kl(q,m)
H = load('eval/converted_gss/human_dist.csv')

rows = []
for f in sorted(glob.glob('eval/panel_gss/*_c10.csv')):
    m = re.match(r'.*/(.+)_(base|ft)_c10\.csv', f)
    if not m: continue
    tag, var = m.group(1), m.group(2)
    D = load(f); raw, adj, jss = [], [], []
    for q in qs:
        qid = q['q_id']
        if qid in EXCL or qid not in D or qid not in H: continue
        b = float(sum(D[qid][i] for i in q['neg_idx']) - sum(H[qid][i] for i in q['neg_idx']))
        raw.append(b); adj.append(-b if qid in FLIP else b); jss.append(js(D[qid], H[qid]))
    rows.append(dict(model=tag, variant=var, n=len(raw),
                     NRR=round(np.mean(raw), 3),
                     NRR_adj=round(np.mean(adj), 3),
                     JS=round(np.mean(jss), 4)))
t = pd.DataFrame(rows).sort_values(['model', 'variant'])
t.to_csv('eval/table_gss_adjusted.csv', index=False)
print(f"GSS 조정 NRR (유효 {len(qs)-len(EXCL)}문항 · 반전 {len(FLIP)} · 제외 {len(EXCL)})")
print(t.to_string(index=False))
for v in ['base', 'ft']:
    s = t[t.variant == v]
    if len(s): print(f"  {v:4s} 평균: NRR {s.NRR.mean():+.3f} · 조정 {s.NRR_adj.mean():+.3f} · JS {s.JS.mean():.4f}")
