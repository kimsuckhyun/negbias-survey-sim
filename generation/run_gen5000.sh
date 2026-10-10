#!/bin/bash
# ============================================================
# Auto-Survey EN — one-shot runner for 3-machine parallel gen
#   per machine: domains 100 x polarity 5 x repeats 16 = 8000 attempts
#   quality filter ~65-75% -> ~5200-6000 samples/machine -> ~15-18K on 3 machines
# Usage:
#   SEED=101 bash run_gen5000.sh        (machine1)
#   SEED=102 bash run_gen5000.sh        (machine2)
#   SEED=103 bash run_gen5000.sh        (machine3)
# Override knobs: N_DOMAINS / N_PERSONA / N_PER_POL / PORT / ACL_BASE
# ============================================================
set -e

SEED="${SEED:-42}"
N_DOMAINS="${N_DOMAINS:-100}"
N_PERSONA="${N_PERSONA:-30}"      # personas per polarity (x5)
N_PER_POL="${N_PER_POL:-16}"      # repeats per (survey x polarity)
PORT="${PORT:-8080}"
SERVER="http://127.0.0.1:${PORT}"
export ACL_BASE="${ACL_BASE:-$HOME/ACL_EN}"

CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$ACL_BASE/logs"

echo "═══════════════════════════════════════════════════════════"
echo "  시작: $(date)   host=$(hostname)"
echo "  SEED=$SEED  domains=$N_DOMAINS  personas/pol=$N_PERSONA  repeats=$N_PER_POL"
echo "  목표: 시도 $((N_DOMAINS * 5 * N_PER_POL))개 → 통과 ~$((N_DOMAINS * 5 * N_PER_POL * 70 / 100))개"
echo "  BASE=$ACL_BASE"
echo "═══════════════════════════════════════════════════════════"

# ── conda python 강제 (ssh 비대화형 셸은 conda 미활성) ──
for C in "$HOME/miniconda3" "$HOME/anaconda3"; do
    if [ -x "$C/bin/python" ]; then export PATH="$C/bin:$PATH"; break; fi
done
echo "[env] python = $(command -v python)"

# ── 0) python 의존성 ──
python -c "import requests" 2>/dev/null || python -m pip install -q requests

# ── 1) llama-server 확보 (떠 있으면 재사용, 없으면 기동) ──
if curl -s --max-time 3 "$SERVER/v1/models" > /dev/null 2>&1; then
    echo "[server] 이미 $PORT 에서 동작 중 → 재사용"
else
    BIN="$HOME/llama.cpp/build/bin/llama-server"
    [ -x "$BIN" ] || { echo "ERROR: $BIN 없음"; exit 1; }
    MODEL_PATH=$(find "$HOME" -maxdepth 3 -iname "*gemma-4*31B*Q4_K_M*.gguf" 2>/dev/null | head -1)
    [ -n "$MODEL_PATH" ] || { echo "ERROR: gemma-4 31B Q4_K_M gguf를 \$HOME 아래에서 못 찾음"; exit 1; }
    echo "[server] 기동: $MODEL_PATH"
    nohup env CUDA_VISIBLE_DEVICES=0 "$BIN" \
        -m "$MODEL_PATH" -c 131072 -ngl 99 -fa on --no-mmap \
        --cache-type-k q8_0 --cache-type-v q8_0 -b 2048 -ub 2048 --threads 20 \
        --jinja --reasoning off --host 0.0.0.0 --port "$PORT" \
        > "$HOME/llama_server.log" 2>&1 &
    SRV_PID=$!
    sleep 8
    # 인자 에러 등으로 즉사했으면 미니멀 플래그로 1회 재시도 (구버전 빌드 대비)
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
        echo "[server] 전체 플래그 실패 → 미니멀 플래그로 재시도"
        nohup env CUDA_VISIBLE_DEVICES=0 "$BIN" \
            -m "$MODEL_PATH" -c 131072 -ngl 99 --no-mmap \
            --jinja --host 0.0.0.0 --port "$PORT" \
            > "$HOME/llama_server.log" 2>&1 &
    fi
fi

# ── 로딩 대기 + 모델 ID 조회 (최대 10분) ──
# llama-server는 모델 로딩 전에 포트부터 열므로, 연결 성공이 아니라
# /v1/models 에서 모델 ID가 실제로 읽힐 때까지 기다려야 함
echo -n "[server] 로딩 대기"
MODEL_ID=""
for i in $(seq 1 120); do
    MODEL_ID=$(curl -s --max-time 3 "$SERVER/v1/models" 2>/dev/null | python -c "
import sys, json
try:
    j = json.load(sys.stdin)
    d = j.get('data') or j.get('models') or []
    print(d[0].get('id') or d[0].get('name') or d[0].get('model') or '')
except Exception:
    print('')
" 2>/dev/null) || MODEL_ID=""
    [ -n "$MODEL_ID" ] && break
    echo -n "."
    sleep 5
done
echo ""
if [ -z "$MODEL_ID" ]; then
    echo "ERROR: 서버가 10분 내에 준비 안 됨. ~/llama_server.log 마지막 30줄:"
    tail -30 "$HOME/llama_server.log"
    exit 1
fi
echo "[server] MODEL_ID = $MODEL_ID"

cd "$CODE_DIR"
COMMON=(--server "$SERVER" --model "$MODEL_ID")

# ── Stage 1: 도메인 ──
echo ""
echo "[$(date +%H:%M:%S)] Stage 1/4: 도메인 $N_DOMAINS 개 (다분야)"
python auto_survey_pipeline.py gen-domains --n "$N_DOMAINS" "${COMMON[@]}"

# ── Stage 2: 페르소나 ──
echo ""
echo "[$(date +%H:%M:%S)] Stage 2/4: 페르소나 polarity별 $N_PERSONA 명 (seed=$SEED)"
python auto_survey_pipeline.py gen-personas --n-per-polarity "$N_PERSONA" --seed "$SEED" "${COMMON[@]}"

# ── Stage 3: 설문지 ──
echo ""
echo "[$(date +%H:%M:%S)] Stage 3/4: 설문지 생성 (도메인 수만큼, ~16-21시간)"
python auto_survey_pipeline.py gen-surveys "${COMMON[@]}"

# ── Stage 4: 응답 (5종 polarity 전부) ──
echo ""
echo "[$(date +%H:%M:%S)] Stage 4/4: 응답 생성 (5종 x 반복 $N_PER_POL, ~5-9일)"
if [ -n "${POLARITIES:-}" ]; then
    python auto_survey_pipeline.py gen-responses --polarities "$POLARITIES" \
        --n-per-polarity "$N_PER_POL" --seed "$SEED" "${COMMON[@]}"
else
    python auto_survey_pipeline.py gen-responses \
        --n-per-polarity "$N_PER_POL" --seed "$SEED" "${COMMON[@]}"
fi

# ── 분석 + 분포 ──
echo ""
LATEST=$(ls -t "$ACL_BASE/output"/train_data_*.jsonl 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    python auto_survey_pipeline.py analyze --jsonl "$LATEST"
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  완료: $(date)   host=$(hostname)"
    echo "  데이터: $LATEST ($(wc -l < "$LATEST") samples)"
    python - "$LATEST" << 'PYEOF'
import json, sys, statistics
from collections import Counter
samples = [json.loads(l) for l in open(sys.argv[1])]
c = Counter(s['polarity'] for s in samples)
exp = {'strongly_negative':1.4,'weakly_negative':2.3,'neutral':3.0,
       'weakly_positive':3.7,'strongly_positive':4.6}
print("  ─── polarity별 분포 / ORDINAL 평균 ───")
for k in exp:
    sub = [s['metrics']['mean'] for s in samples if s['polarity']==k]
    m = statistics.mean(sub) if sub else 0
    flag = "OK" if sub and abs(m-exp[k])<0.5 else ("WARN" if sub else "-")
    print(f"    {k:18s}: {c.get(k,0):>5d}개  mean={m:.2f} (기대 {exp[k]})  [{flag}]")
PYEOF
    echo "═══════════════════════════════════════════════════════════"
else
    echo "WARN: train_data jsonl이 없음 — Stage 4 로그 확인 필요"
fi
