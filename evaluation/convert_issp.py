#!/usr/bin/env python3
"""SimBench-ISSP → 우리 스키마 (국가별 페르소나 · valence 라벨 · 감사)"""
import pandas as pd, re, json, os, ast
from collections import Counter

NEG_K = ['completely dissatisfied','very dissatisfied','fairly dissatisfied','dissatisf',
 'strongly disagree','disagree strongly','disagree','very negative','somewhat negative','negative',
 'very strong conflicts','strong conflicts','definitely not accept','probably not accept',
 'decreased it a lot','decreased it a little','definitely should not','probably should not',
 'to a very great extent','to a great extent','very bad','extremely bad','bad',
 'not at all satisfied','very worried','extremely worried','somewhat worried','worried',
 'very dangerous','extremely dangerous','dangerous','very serious','serious problem',
 'very unlikely','strongly oppose','oppose','very unfair','unfair','worse','much worse',
 'not very likely','never justif','always wrong','very harmful','harmful']
POS_K = ['completely satisfied','very satisfied','fairly satisfied','satisf',
 'strongly agree','agree strongly','agree','very positive','somewhat positive','positive',
 'no conflicts','not very strong conflict','definitely accept','probably accept',
 'increased it a lot','increased it a little','definitely should have','probably should have',
 'not at all','not worried','not at all worried','not very worried','not dangerous',
 'very good','extremely good','good','better','much better','very fair','fair',
 'very likely','strongly favor','favor','not harmful','beneficial','not wrong']
NEU_K = ['neither','neutral','in between','about the same','no effect','depends','mixed',
 'to some extent','to a small extent','neither increased','neither positive','neither satisfied']
# 부분 문자열 오매칭 방지: 아래는 NEG 검사보다 먼저 POS로 확정
POS_FIRST = ['not very strong conflict','there are no conflicts','no conflicts',
 'not very worried','not at all worried','not worried','not dangerous','not at all dangerous',
 'not very likely','not at all satisfied' if False else 'not applicable',
 'not very serious','not serious','not harmful','not at all harmful','not bad','not unfair']
DK_K = ["can't choose","cant choose",'refused','no answer','not applicable','never heard',
 'unsure','not sure','no opinion','undecided',"don't know",'dont know']
# 제외할 척도 (부정 극 미정의)
SKIP_PAT = [r'^\d+$', r'^\d+ ', r'once a week', r'once in the past', r'several times',
            r'^never$', r'essential', r'important']

def label(o):
    t = str(o).lower().strip()
    for k in DK_K:
        if k in t: return 'DK', None
    for p in SKIP_PAT:
        if re.search(p, t): return 'SKIP', None
    for k in POS_FIRST:
        if k in t: return 'POS', None
    for k in NEU_K:
        if k in t: return 'NEU', None
    for k in NEG_K:
        if k in t: return 'NEG', k.replace(' ','_')
    for k in POS_K:
        if k in t: return 'POS', None
    return 'UNK', None

d = pd.read_csv('data_simbench/SimBenchPop.csv')
s = d[d.dataset_name == 'ISSP']
qs, rows, stats = [], [], Counter()
audit = open('eval/converted_issp_audit.txt', 'w')
for idx, r in s.iterrows():
    stats['total'] += 1
    country = re.sub(r'.*You are from ', '', r.group_prompt_template).rstrip('.').strip()
    body = r.input_template.split('Options:')[0].strip()
    opts = re.findall(r'\(([A-Z]+)\): (.+)', r.input_template)
    try:
        H = ast.literal_eval(r.human_answer)
    except Exception:
        stats['parse'] += 1; continue
    labs = [(k, v, *label(v)) for k, v in opts]
    if any(l == 'SKIP' for _, _, l, _ in labs): stats['skip_scale'] += 1; continue
    keep = [(k, v, l, f) for k, v, l, f in labs if l != 'DK']
    if not (2 <= len(keep) <= 10): stats['badK'] += 1; continue
    L = [l for _, _, l, _ in keep]
    if 'UNK' in L: stats['unk'] += 1; continue
    if 'NEG' not in L or 'POS' not in L: stats['no_poles'] += 1; continue
    P = [float(H.get(k, 0)) for k, _, _, _ in keep]
    tot = sum(P)
    if tot <= 0: continue
    P = [p/tot for p in P]
    O = [v for _, v, _, _ in keep]
    neg_idx = [i for i, l in enumerate(L) if l == 'NEG']
    fam = next(f for _, _, l, f in keep if l == 'NEG')
    qid = f"ISSP{idx:04d}"
    qs.append(dict(q_id=qid, question=body, options=O, neg_idx=neg_idx, family=fam,
                   ordinal=list(range(1, len(O)+1)), country=country,
                   persona_prefix=r.group_prompt_template))
    for i, p in enumerate(P):
        rows.append(dict(q_id=qid, option_num=i+1, p=round(p, 6)))
    hn = sum(P[i] for i in neg_idx)
    print(f"\n[{qid}] {country} · family={fam}", file=audit)
    print(f"  Q: {body[:300]}", file=audit)
    for i, o in enumerate(O):
        print(f"    {i+1}. {o}  ({P[i]:.1%}){'  <- NEG' if i in neg_idx else ''}", file=audit)
    print(f"  인간 NEG선택률: {hn:.1%}", file=audit)
    stats['kept'] += 1
audit.close()
os.makedirs('eval/converted_issp', exist_ok=True)
json.dump(qs, open('eval/converted_issp/eval_questions.json','w'))
pd.DataFrame(rows).to_csv('eval/converted_issp/human_dist.csv', index=False)
print(f"ISSP {stats['total']} | 유지 {stats['kept']} | 탈락: 척도부적합 {stats['skip_scale']} · 극성무 {stats['no_poles']} · UNK {stats['unk']} · 옵션수 {stats['badK']}")
print("국가 수:", len({q['country'] for q in qs}))
print("감사: eval/converted_issp_audit.txt")
