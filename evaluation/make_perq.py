#!/usr/bin/env python3
"""94문항 개별 지표: 문항 × (모델×변형) bias 전표 + JS + 상위 회피/과잉 문항"""
import pandas as pd, json, numpy as np, glob, re, os
qs = json.load(open('eval/converted/eval_questions.json'))
EVAL={'risk','unacceptab','fair','pessimis','comfortab','dissatisf','negative','confident','disagree','worse'}
WORRY={'concerned','worry','worried'}
def load(f):
    d=pd.read_csv(f); pc='p' if 'p' in d.columns else 'option_ratio'; out={}
    for q in qs:
        s=d[d.q_id==q['q_id']].sort_values('option_num')[pc].to_numpy(float)
        if len(s)==len(q['options']) and s.sum()>0: out[q['q_id']]=s/s.sum()
    return out
def js(p,q):
    e=1e-12;p=p+e;p=p/p.sum();q=q+e;q=q/q.sum();m=(p+q)/2
    kl=lambda a,b: float((a*np.log(a/b)).sum()); return .5*kl(p,m)+.5*kl(q,m)
H=load('eval/converted/human_dist.csv')
rows={q['q_id']: {'q_id':q['q_id'],'family':q['family'],
      'cluster':'A' if q['family'] in EVAL else ('B' if q['family'] in WORRY else 'C'),
      'human_neg':round(float(sum(H[q['q_id']][i] for i in q['neg_idx'])),3)}
      for q in qs if q['q_id'] in H}
for f in sorted(glob.glob('eval/panel_h100/*_c10.csv')):
    m=re.match(r'.*/(.+)_(base|ft)_c10\.csv',f)
    if not m: continue
    tag,var=m.group(1),m.group(2); D=load(f)
    for q in qs:
        qid=q['q_id']
        if qid not in D or qid not in rows: continue
        b=float(sum(D[qid][i] for i in q['neg_idx'])-sum(H[qid][i] for i in q['neg_idx']))
        rows[qid][f'{tag}_{var}']=round(b,3)
df=pd.DataFrame(rows.values())
base_cols=[c for c in df.columns if c.endswith('_base')]
df['base_mean']=df[base_cols].mean(axis=1).round(3)
df=df.sort_values('base_mean')
df.to_csv('eval/perq_c10.csv',index=False)
print(f"저장: eval/perq_c10.csv ({len(df)}문항 × {len(df.columns)}열)\n")
print("═══ base 평균 기준 회피 상위 10문항 (전 모델 평균이 가장 −) ═══")
print(df[['q_id','cluster','family','human_neg','base_mean']].head(10).to_string(index=False))
print("\n═══ 과잉 상위 10문항 (가장 +) ═══")
print(df[['q_id','cluster','family','human_neg','base_mean']].tail(10).to_string(index=False))
print("\n═══ 클러스터별 분해 (base 평균) ═══")
print(df.groupby('cluster').base_mean.agg(['count','mean','min','max']).round(3).to_string())
