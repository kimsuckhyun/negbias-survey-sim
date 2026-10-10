#!/bin/bash
# run_eval_all.sh — 완료된 어댑터 자동 발견 → base/FT 판독 → Table 2 출력
# 사용 (v8 종료 후, GPU 빈 상태에서):
#   conda activate acl31b && cd "${ACL_ROOT:-.}"
#   nohup bash scripts/run_eval_all.sh > eval_all.log 2>&1 &
set -u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate acl31b
export NCCL_NVLS_ENABLE=0
cd "${ACL_ROOT:-.}"
Q=eval/converted/eval_questions.json
D=eval/dists
mkdir -p $D

# TAG|MODEL|EXTRA (학습 때와 동일 로드 옵션)
CFG=(
  "gemma4-31b|google/gemma-4-31b-it|--load_4bit"
  "qwen36-27b|Qwen/Qwen3.6-27B|--load_4bit"
  "phi4-14b|microsoft/phi-4|--load_4bit"
  "gemma4-12b|google/gemma-4-12b-it|--load_4bit"
  "glm47-flash|zai-org/GLM-4.7-Flash|"
  "gptoss-20b|openai/gpt-oss-20b|--attn eager"
  "mistral-24b|mistralai/Mistral-Small-24B-Instruct-2501|--load_4bit"
  "exaone-32b|LGAI-EXAONE/EXAONE-4.0.1-32B|--load_4bit"
)
for entry in "${CFG[@]}"; do
  IFS='|' read -r TAG MODEL EXTRA <<< "$entry"
  ADP=outputs/${TAG}-neg-1to10-ep1
  if ! ls $ADP/adapter_model*.safetensors >/dev/null 2>&1; then
    echo "[eval-skip] $TAG (어댑터 없음)"; continue; fi
  for VAR in base ft; do
    OUTCSV=$D/${TAG}_${VAR}.csv
    [ -f $OUTCSV ] && { echo "[resume] $TAG $VAR"; continue; }
    A=$([ $VAR = ft ] && echo $ADP || echo none)
    echo "════ [eval] $TAG $VAR ════ $(date)"
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_multi_score.py $EXTRA \
        --model_name "$MODEL" --adapter $A --questions $Q --out $OUTCSV \
        > logs/eval_${TAG}_${VAR}.log 2>&1 \
      && echo "[eval-done] $TAG $VAR" || echo "[eval-fail] $TAG $VAR"
  done
done

# ── Table 2: 클러스터별 bias + JS/WD 집계 ──
python3 - << 'PYEOF'
import pandas as pd, json, numpy as np, glob, os, re
qs = json.load(open('eval/converted/eval_questions.json'))
EVAL={'risk','unacceptab','fair','pessimis','comfortab','dissatisf','negative','confident','disagree','worse'}
WORRY={'concerned','worry','worried'}
def load(f):
    d=pd.read_csv(f); out={}
    for q in qs:
        s=d[d.q_id==q['q_id']].sort_values('option_num')['p'].to_numpy(float)
        if len(s)==len(q['options']) and s.sum()>0: out[q['q_id']]=s/s.sum()
    return out
def js(p,q):
    e=1e-12;p=p+e;p=p/p.sum();q=q+e;q=q/q.sum();m=(p+q)/2
    kl=lambda a,b: float((a*np.log(a/b)).sum()); return .5*kl(p,m)+.5*kl(q,m)
def wd(p,q,pos):
    o=np.argsort(pos);pos=np.array(pos,float)[o];p=p[o];q=q[o];sp=pos[-1]-pos[0]
    return float((np.abs(np.cumsum(p-q)[:-1])*np.diff(pos)).sum()/sp) if sp>0 else 0.0
H=load('eval/converted/human_dist.csv') if os.path.exists('eval/converted/human_dist.csv') else None
if H is None:
    hd=pd.read_csv('eval/converted/human_dist.csv'); raise SystemExit('human_dist 없음')
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
print("\n════ Table 2 — 모델별 평가적 부정 회피 (A37) · 우려 과잉 (B10) ════")
print(t.to_string(index=False))
PYEOF
echo "════ eval 전체 완료 $(date) ════"
