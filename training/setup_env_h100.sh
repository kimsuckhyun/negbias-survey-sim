#!/bin/bash
# setup_env_h100.sh — scp-gpu (H100×8) 학습 환경 구성
# 사용: bash setup_env_h100.sh
set -e

# ── 0) HF 캐시를 대용량 디스크로 (31B 모델 60GB+ 다운로드 대비) ──
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
mkdir -p $HF_HOME
grep -q "HF_HOME" ~/.bashrc || echo 'export HF_HOME=${HF_HOME:-$HOME/hf_cache}' >> ~/.bashrc

# ── 1) conda 환경 (EC2에서 검증된 스택 이식) ──
conda create -n acl31b python=3.11 -y
source activate acl31b 2>/dev/null || conda activate acl31b

# ── 2) PyTorch (CUDA 13.0 드라이버 대응) ──
pip install torch==2.13.* --index-url https://download.pytorch.org/whl/cu130

# ── 3) 학습 스택 (EC2 검증 버전 고정) ──
pip install "transformers>=4.57" "trl==1.8.*" "peft==0.19.*" \
    "bitsandbytes==0.49.*" accelerate datasets pillow \
    sentencepiece protobuf pandas scipy

# ── 4) flash-attn (H100 필수급 — 실패해도 학습 스크립트에 우회 플래그 있음) ──
pip install flash-attn --no-build-isolation || \
    echo "[warn] flash-attn 빌드 실패 — 학습 시 --attn eager 로 진행 가능"

# ── 5) 게이트 모델 로그인 (gemma는 HF에서 라이선스 동의 + 토큰 필요) ──
echo ""
echo ">> huggingface-cli login 을 실행해 토큰을 등록하세요 (gemma 계열 필수)"
echo ">> https://huggingface.co/google/gemma-4-31b-it 페이지에서 라이선스 동의 선행"

# ── 6) 검증 ──
python - << 'EOF'
import torch, transformers, peft, trl
print("=" * 50)
print("torch      :", torch.__version__, "| CUDA:", torch.version.cuda)
print("GPU count  :", torch.cuda.device_count(), "(기대: 8)")
print("GPU 0      :", torch.cuda.get_device_name(0))
print("bf16 지원  :", torch.cuda.is_bf16_supported())
print("transformers:", transformers.__version__, "| peft:", peft.__version__, "| trl:", trl.__version__)
try:
    import flash_attn
    print("flash-attn :", flash_attn.__version__)
except ImportError:
    print("flash-attn : 미설치 (--attn eager 로 진행)")
print("=" * 50)
EOF
echo "환경 구성 완료 — conda activate acl31b 후 사용"
