#!/usr/bin/env python3
import json, glob, os
from collections import defaultdict, Counter
import pandas as pd
NEG_K = ['not at all confident','not too confident','not very confident','hardly any',
 'strongly oppose','somewhat oppose','oppose','very unfavorable','somewhat unfavorable',
 'very worried','somewhat worried','worry a lot','worry a little','worried','afraid',
 'very concerned','somewhat concerned','concerned','very big problem','moderately big problem',
 'major problem','serious problem','unacceptable','dissatisf','disagree','pessimis',
 'mostly negative','very negative','somewhat negative','negative','a lot of risk','great deal of risk',
 'some risk','less safe','never justif','morally wrong','always wrong','almost always wrong','wrong',
 'very bad','somewhat bad','bad','worse','harmful','disapprove','distrust','too little','not likely',
 'very unlikely','somewhat unlikely','unlikely','very dissatisfied','not too happy','unhappy']
POS_K = ['very confident','somewhat confident','a great deal','strongly favor','somewhat favor','favor',
 'very favorable','somewhat favorable','not worried','not at all worried','not too worried',
 'not concerned','not a problem','acceptable','satisf','agree','optimis','mostly positive',
 'very positive','somewhat positive','positive','no risk','not much risk','safer','always justif',
 'not wrong at all','not wrong','very good','somewhat good','good','better','beneficial','approve',
 'trust','very likely','somewhat likely','likely','very happy','pretty happy']
NEU_K = ['neither','moderate','in between','middle','about right','about the same','no effect',
 'stayed the same','depends','mixed','only some','sometimes']
DK_K = ["don't know","dont know",'refused','no answer','not applicable','never heard',
 'unsure','not sure',"can't choose",'no opinion','undecided',"haven't heard"]
def label(opt):
    t = str(opt).lower().strip()
    for k in DK_K:
        if k in t: return 'DK', None
    for k in NEG_K:
        if k in t: return 'NEG', k.replace(' ', '_')
    for k in POS_K:
        if k in t: return 'POS', None
    for k in NEU_K:
        if k in t: return 'NEU', None
    return 'UNK', None
f = sorted(glob.glob('data_subpop/**/*eval*.jsonl', recursive=True))[0]
print("입력:", f)
by_q = defaultdict(dict); meta = {}
for line in open(f):
    r = json.loads(line)
    if str(r.get('attribute','')).upper() != 'SEX': continue
    by_q[r['qkey']][r['group']] = list(map(float, r['responses']))
    meta[r['qkey']] = r
W = {'Male': 0.487, 'Female': 0.513}
qs, rows, stats = [], [], Counter()
audit = open('eval/converted_gss_audit.txt', 'w')
for qk, groups in sorted(by_q.items()):
    stats['total'] += 1
    if not all(g in groups for g in W): stats['no_both_sex'] += 1; continue
    r = meta[qk]; opts = list(r['options'])
    P = [W['Male']*m + W['Female']*fm for m, fm in zip(groups['Male'], groups['Female'])]
    labs = [label(o) for o in opts]
    keep = [i for i,(l,_) in enumerate(labs) if l != 'DK']
    if len(keep) < 2 or len(keep) > 10: stats['badK'] += 1; continue
    L = [labs[i][0] for i in keep]
    if 'UNK' in L: stats['unk'] += 1; continue
    if 'NEG' not in L or 'POS' not in L: stats['no_poles'] += 1; continue
    O = [str(opts[i]) for i in keep]; Pk = [P[i] for i in keep]; s = sum(Pk)
    if s <= 0: continue
    Pk = [x/s for x in Pk]
    neg_idx = [i for i,l in enumerate(L) if l=='NEG']
    fam = labs[[i for i in range(len(labs)) if labs[i][0]=='NEG'][0]][1]
    ordv = [r['ordinal'][i] for i in keep] if r.get('ordinal') else list(range(1,len(O)+1))
    qs.append(dict(q_id=f"GSS_{qk}", question=r['question'], options=O,
                   neg_idx=neg_idx, family=fam, ordinal=ordv))
    for i,(o,pp) in enumerate(zip(O,Pk)):
        rows.append(dict(q_id=f"GSS_{qk}", option_num=i+1, p=round(pp,6)))
    hn = sum(Pk[i] for i in neg_idx)
    print(f"\n[GSS_{qk}] family={fam}", file=audit)
    print(f"  Q: {r['question']}", file=audit)
    for i,o in enumerate(O):
        tag = " <- NEG" if i in neg_idx else ""
        print(f"    {i+1}. {o}  ({Pk[i]:.1%}){tag}", file=audit)
    print(f"  인간 NEG선택률: {hn:.1%}", file=audit)
    stats['kept'] += 1
audit.close()
os.makedirs('eval/converted_gss', exist_ok=True)
json.dump(qs, open('eval/converted_gss/eval_questions.json','w'))
pd.DataFrame(rows).to_csv('eval/converted_gss/human_dist.csv', index=False)
print(f"GSS {stats['total']}문항 | 유지 {stats['kept']} | 탈락: 극성무 {stats['no_poles']} · UNK {stats['unk']} · 옵션수 {stats['badK']} · 성별결측 {stats['no_both_sex']}")
