import pandas as pd, json, numpy as np, glob, os, re
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
def wd(p,q,pos):
    o=np.argsort(pos);pos=np.array(pos,float)[o];p=p[o];q=q[o];sp=pos[-1]-pos[0]
    return float((np.abs(np.cumsum(p-q)[:-1])*np.diff(pos)).sum()/sp) if sp>0 else 0.0
H=load('eval/converted/human_dist.csv')
rows=[]
for f in sorted(glob.glob('eval/dists/*_*.csv')):
    m=re.match(r'.*/(.+)_(base|ft)\.csv', f)
    if not m: continue
    tag,var=m.group(1),m.group(2); D=load(f); rec=[]
    for q in qs:
        qid=q['q_id']
        if qid not in D or qid not in H: continue
        b=float(sum(D[qid][i] for i in q['neg_idx'])-sum(H[qid][i] for i in q['neg_idx']))
        cl='A' if q['family'] in EVAL else ('B' if q['family'] in WORRY else 'C')
        rec.append((b,js(D[qid],H[qid]),wd(D[qid],H[qid],q['ordinal']),cl))
    df=pd.DataFrame(rec,columns=['b','j','w','cl'])
    rows.append(dict(model=tag,variant=var,n=len(df),
        A37=round(df[df.cl=='A'].b.mean(),3), B10=round(df[df.cl=='B'].b.mean(),3),
        bias=round(df.b.mean(),3), abs_bias=round(df.b.abs().mean(),3),
        JS=round(df.j.mean(),3), WD=round(df.w.mean(),3)))
t=pd.DataFrame(rows).sort_values(['variant','A37'])
t.to_csv('eval/table2_multimodel.csv', index=False)
print(t.to_string(index=False))

# ── 앙상블 확장: 모델별 α 그리드 → 최적 α와 성능 ──
print("\n════ 앙상블 (α·FT + (1−α)·base) — 모델별 최적 α ════")
import itertools
ens_rows=[]
tags = sorted(set(re.match(r'.*/(.+)_(base|ft)\.csv', f).group(1) for f in glob.glob('eval/dists/*_*.csv')))
for tag in tags:
    fb, ff = f'eval/dists/{tag}_base.csv', f'eval/dists/{tag}_ft.csv'
    if not (os.path.exists(fb) and os.path.exists(ff)): continue
    B, F = load(fb), load(ff)
    best=None
    for a in [0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8]:
        rec=[]
        for q in qs:
            qid=q['q_id']
            if qid not in B or qid not in F: continue
            e=a*F[qid]+(1-a)*B[qid]
            b=float(sum(e[i] for i in q['neg_idx'])-sum(H[qid][i] for i in q['neg_idx']))
            cl='A' if q['family'] in EVAL else ('B' if q['family'] in WORRY else 'C')
            rec.append((b,js(e,H[qid]),wd(e,H[qid],q['ordinal']),cl))
        df=pd.DataFrame(rec,columns=['b','j','w','cl'])
        row=dict(model=tag, alpha=a, A37=round(df[df.cl=='A'].b.mean(),3),
                 bias=round(df.b.mean(),3), abs_bias=round(df.b.abs().mean(),3),
                 JS=round(df.j.mean(),3), WD=round(df.w.mean(),3))
        if best is None or row['abs_bias']<best['abs_bias']: best=row
    ens_rows.append(best)
e=pd.DataFrame(ens_rows).sort_values('model')
e.to_csv('eval/table2_ensemble.csv', index=False)
print(e.to_string(index=False))
print("\n→ alpha 열이 0.2~0.5 구간에 몰리면 base-heavy 보편성(RQ3) 성립")
