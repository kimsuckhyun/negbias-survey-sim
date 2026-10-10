#!/bin/bash
# run_all_models.sh — 모델 라인업 순차 학습 (스모크 → 본학습)
# 사용: conda activate acl31b && nohup bash run_all_models.sh > run_all.log 2>&1 &
set -u
cd "${ACL_ROOT:-.}"
DATA=${ACL_ROOT:-.}/data/train_neg_1to10_cohort.jsonl
OUT=${ACL_ROOT:-.}/outputs
mkdir -p $OUT logs

# 모델 라인업: "HF_ID|저장태그"  (필요 시 추가/삭제)
MODELS=(
  "google/gemma-4-12b-it|gemma4-12b"
  "google/gemma-4-31b-it|gemma4-31b"
  "Qwen/Qwen3.6-27B|qwen36-27b"
  "microsoft/phi-4|phi4-14b"
  "mistralai/Mistral-Small-24B-Instruct-2501|mistral-24b"
  "openai/gpt-oss-20b|gptoss-20b"
  "LGAI-EXAONE/EXAONE-4.0.1-32B|exaone-32b"
)

for entry in "${MODELS[@]}"; do
  MODEL="${entry%%|*}"; TAG="${entry##*|}"
  echo "════════ [$TAG] $MODEL ════════"

  # 1) 스모크 (1 GPU · 10 steps · seq 4K) — 로딩/템플릿/LoRA 타깃 검증
  echo "[smoke] $TAG 시작"
  CUDA_VISIBLE_DEVICES=0 python scripts/train_lora_h100.py --attn sdpa \
      --model_name "$MODEL" --data $DATA \
      --out $OUT/${TAG}-smoke --max_steps 10 --max_seq 4096 \
      > logs/${TAG}_smoke.log 2>&1
  if [ $? -ne 0 ]; then
      echo "[smoke] $TAG 실패 — logs/${TAG}_smoke.log 확인, 본학습 건너뜀"
      continue
  fi
  echo "[smoke] $TAG 통과"

  # 2) 본학습 (8 GPU · 1 epoch)
  echo "[train] $TAG 시작 $(date)"
  torchrun --nproc_per_node 8 scripts/train_lora_h100.py --attn sdpa \
      --model_name "$MODEL" --data $DATA \
      --out $OUT/${TAG}-neg-1to10-ep1 \
      > logs/${TAG}_train.log 2>&1
  echo "[train] $TAG 종료 $(date) (exit $?)"
done
echo "════════ 전체 완료 $(date) ════════"
