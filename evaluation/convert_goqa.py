#!/usr/bin/env python3
"""GlobalOpinionQA(미국) → 우리 스키마 변환 + valence 라벨 + 감사파일"""
import re, ast, json, os
from collections import Counter
from datasets import load_dataset
import pandas as pd

NEG_K = ['not at all confident','not too confident','not very confident','no confidence',
 'strongly oppose','somewhat oppose','very unfavorable','somewhat unfavorable','unfavorable',
 'very worried','somewhat worried','worry a lot','worry a little','worried','very concerned',
 'somewhat concerned','concerned','very big problem','moderately big problem','major problem',
 'minor problem','small problem','serious problem','unacceptable','dissatisf','disagree',
 'pessimis','mostly negative','very negative','somewhat negative','negative','a lot of risk',
 'great deal of risk','some risk','less safe','never justifiable','morally wrong','wrong',
 'very bad','somewhat bad','bad thing','bad','worse','harmful','disapprove','distrust','oppose']
POS_K = ['very confident','somewhat confident','a lot of confidence','strongly favor',
 'somewhat favor','very favorable','somewhat favorable','favorable','not worried',
 'not at all worried','not too worried','not concerned','not a problem','acceptable','satisf',
 'agree','optimis','mostly positive','very positive','somewhat positive','positive',
 'no risk at all','not much risk','more safe','safer','always justifiable','justifiable',
 'morally acceptable','very good','somewhat good','good thing','good','better','beneficial',
 'approve','trust','favor']
NEU_K = ['neither','moderate','somewhere in between','middle','about the same','no effect',
 'stayed about the same','depends','mixed','equal']
DK_K = ["don't know","dont know",'refused','no answer','not applicable','never heard',
 'unsure','not sure',"can't choose",'cant choose','no opinion','undecided','decline']

def label(opt):
    t = opt.lower().strip()
    for k in DK_K:
        if k in t: return 'DK', None
    for k in NEG_K:
        if k in t: return 'NEG', k.split()[0] if ' ' not in k else k.replace(' ','_')
    for k in POS_K:
        if k in t: return 'POS', None
    for k in NEU_K:
        if k in t: return 'NEU', None
    return 'UNK', None

ds = load_dataset("Anthropic/llm_global_opinions")['train']
EVALF = {'unacceptab','dissatisf','disagree','unfavorab','pessimis','negative','worse',
         'risk','confident','bad','oppose','wrong','safe','justif','harmful','disapprove','distrust',
         'very_bad','somewhat_bad','bad_thing','a_lot_of_risk','great_deal_of_risk','some_risk',
         'less_safe','never_justifiable','morally_wrong','not_at_all_confident','not_too_confident',
         'not_very_confident','no_confidence','strongly_oppose','somewhat_oppose','very_unfavorable',
         'somewhat_unfavorable','mostly_negative','very_negative','somewhat_negative'}
qs, rows, stats = [], [], Counter()
audit = open('eval/converted_goqa_audit.txt','w')
kept = 0
for idx, r in enumerate(ds):
    stats['total'] += 1
    m = re.search(r"\{.*\}", r['selections'], re.S)
    if not m: continue
    try:
        sel = ast.literal_eval(m.group(0))
        opts = ast.literal_eval(r['options'])
    except Exception:
        stats['parse_fail'] += 1; continue
    if 'United States' not in sel: continue
    p = list(map(float, sel['United States']))
    if len(p) != len(opts) or sum(p) < 0.9: continue
    stats['us'] += 1
    labs = [label(str(o)) for o in opts]
    keep_i = [i for i, (l, _) in enumerate(labs) if l != 'DK']
    if len(keep_i) < 2 or len(keep_i) > 10: stats['badK'] += 1; continue
    L = [labs[i][0] for i in keep_i]
    if 'UNK' in L or 'NEG' not in L or 'POS' not in L:
        stats['no_valence'] += 1; continue
    O = [str(opts[i]) for i in keep_i]
    P = [p[i] for i in keep_i]; s = sum(P)
    if s <= 0: continue
    P = [x/s for x in P]
    fam = next(f for (l, f), i in zip(labs, range(len(labs))) if l == 'NEG' and i in keep_i)
    qid = f"GOQA{idx:04d}"
    neg_idx = [i for i, l in enumerate(L) if l == 'NEG']
    cl = 'B' if any(k in fam for k in ('worr','concern')) else ('A' if any(fam.startswith(e) or e in fam for e in EVALF) else 'C')
    stats[f'cl_{cl}'] += 1
    qs.append(dict(q_id=qid, question=r['question'], options=O, neg_idx=neg_idx,
                   family=fam, ordinal=list(range(1, len(O)+1))))
    for i, (o, pp) in enumerate(zip(O, P)):
        rows.append(dict(q_id=qid, option_num=i+1, p=round(pp, 6)))
    hn = sum(P[i] for i in neg_idx)
    print(f"\n[{qid}] cl={cl} family={fam}", file=audit)
    print(f"  Q: {r['question']}", file=audit)
    for i, o in enumerate(O):
        tag = " <- NEG" if i in neg_idx else ""
        print(f"    {i+1}. {o}  ({P[i]:.1%}){tag}", file=audit)
    print(f"  인간 NEG선택률: {hn:.1%}", file=audit)
    kept += 1
audit.close()
os.makedirs('eval/converted_goqa', exist_ok=True)
json.dump(qs, open('eval/converted_goqa/eval_questions.json','w'))
pd.DataFrame(rows).to_csv('eval/converted_goqa/human_dist.csv', index=False)
print(f"전체 {stats['total']} | 미국 응답 보유 {stats['us']} | 유지 {kept}")
print(f"클러스터: A(평가) {stats['cl_A']} · B(우려) {stats['cl_B']} · C(혼합) {stats['cl_C']}")
print(f"탈락: 극성불명 {stats['no_valence']} · 옵션수 {stats['badK']} · 파싱 {stats['parse_fail']}")
print("감사파일: eval/converted_goqa_audit.txt")
