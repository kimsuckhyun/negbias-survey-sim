import pandas as pd, json, numpy as np, glob, re
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
for f in sorted(glob.glob('eval/panel_h100/*_c*.csv')):
    m=re.match(r'.*/(.+)_(base|ft)_c(\d+)\.csv', f)
    if not m: continue
    tag,var,c=m.group(1),m.group(2),int(m.group(3)); D=load(f)
    d0=pd.read_csv(f)
    nresp=int(d0[d0.option_num==1].n_responses.median()) if 'n_responses' in d0.columns else -1
    rec=[]
    for q in qs:
        qid=q['q_id']
        if qid not in D or qid not in H: continue
        b=float(sum(D[qid][i] for i in q['neg_idx'])-sum(H[qid][i] for i in q['neg_idx']))
        cl='A' if q['family'] in EVAL else ('B' if q['family'] in WORRY else 'C')
        rec.append((b,js(D[qid],H[qid]),wd(D[qid],H[qid],q['ordinal']),cl))
    df=pd.DataFrame(rec,columns=['b','j','w','cl'])
    rows.append(dict(model=tag,variant=var,cohort=c,n_resp=nresp,
        NRR=round(df.b.mean(),3), abs_b=round(df.b.abs().mean(),3),
        A37=round(df[df.cl=='A'].b.mean(),3), B10=round(df[df.cl=='B'].b.mean(),3),
        C47=round(df[df.cl=='C'].b.mean(),3),
        JS=round(df.j.mean(),3), WD=round(df.w.mean(),3)))
t=pd.DataFrame(rows).sort_values(['model','variant','cohort'])
t.to_csv('eval/table3_panel_ablation.csv', index=False)
print(t.to_string(index=False))
