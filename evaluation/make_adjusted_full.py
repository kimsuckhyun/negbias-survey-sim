#!/usr/bin/env python3
# 조정 NRR 전체표: 반전 28(우려·규범지적) 부호 뒤집기 + 라벨오류 2 제외
import pandas as pd, json, glob, re
import numpy as np

FLIP = {'WORRYD_W26','WORRYF_W26','WORRYB_W26','WORRYE_W26','WORRYBILL_W29',
'CONCERNGRPa_W49','CONCERNGRPc_W49','HOMEASSIST2_W49','CONCERNCO_W49','GAP21Q24_W82',
'FTRWORRYc_W41','FTRWORRYe_W41','FTRWORRYf_W41','GROWUPVIOL_W26','HARASS1F1d_W41',
'HARASS1NOWRKF2d_W41','HARASS1F1b_W41','RQ5_F1B_W42','RQ5_F1C_W42','RACESURV47a_W43',
'RACESURV47b_W43','RACESURV47d_W43','RACESURV47e_W43','RACESURV47f_W43','GAP21Q23_W82',
'NEWSPROBd_W45','GUNSAFE_W26','GAP21Q41_W82'}          # 28: 조정 시 부호 −로
EXCLUDE = {'FUD33B_W34','RACESURV15b_W43'}              # 2: 라벨 오류 → 제외
EVAL={'risk','unacceptab','fair','pessimis','comfortab','dissatisf','negative','confident','disagree','worse'}
WORRY={'concerned','worry','worried'}

qs = json.load(open('eval/converted/eval_questions.json'))
def load(f):
    d=pd.read_csv(f); pc='p' if 'p' in d.columns else 'option_ratio'; out={}
    for q in qs:
        s=d[d.q_id==q['q_id']].sort_values('option_num')[pc].to_numpy(float)
        if len(s)==len(q['options']) and s.sum()>0: out[q['q_id']]=s/s.sum()
    return out
H = load('eval/converted/human_dist.csv')

rows=[]
for f in sorted(glob.glob('eval/panel_h100/*_c*.csv')):
    m=re.match(r'.*/(.+)_(base|ft)_c(\d+)\.csv', f)
    if not m: continue
    tag,var,c = m.group(1), m.group(2), int(m.group(3))
    D = load(f); rec=[]
    for q in qs:
        qid=q['q_id']
        if qid in EXCLUDE or qid not in D or qid not in H: continue
        b = float(sum(D[qid][i] for i in q['neg_idx'])-sum(H[qid][i] for i in q['neg_idx']))
        adj = -b if qid in FLIP else b
        # 클러스터: 걱정문항 3개는 A→B'로 이동
        if qid in {'FTRWORRYc_W41','FTRWORRYe_W41','FTRWORRYf_W41'}: cl='B'
        elif q['family'] in EVAL: cl='A'
        elif q['family'] in WORRY: cl='B'
        else: cl='C'
        rec.append((adj, b, cl))
    d = pd.DataFrame(rec, columns=['adj','raw','cl'])
    rows.append(dict(model=tag, variant=var, cohort=c,
        A_adj=round(d[d.cl=='A'].adj.mean(),3),
        B_adj=round(d[d.cl=='B'].adj.mean(),3),
        C_adj=round(d[d.cl=='C'].adj.mean(),3),
        전체_adj=round(d.adj.mean(),3),
        전체_raw=round(d.raw.mean(),3)))
t = pd.DataFrame(rows).sort_values(['model','variant','cohort'])
t.to_csv('eval/table_adjusted_full.csv', index=False)
print("조정 NRR 전체표  (음수 = 순화/부정능력 결핍 · 반전28 · 제외2 · A'36/B'13/C'43... n=92)")
print(t.to_string(index=False))
print("\n── c10 요약 (모델 평균) ──")
for v in ['base','ft']:
    s = t[(t.variant==v)&(t.cohort==10)]
    print(f"  {v:4s}: 전체_adj {s.전체_adj.mean():+.3f}  (A' {s.A_adj.mean():+.3f} · B' {s.B_adj.mean():+.3f} · C' {s.C_adj.mean():+.3f})  | 원본 {s.전체_raw.mean():+.3f}")
