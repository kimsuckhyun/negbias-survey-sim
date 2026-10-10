# -*- coding: utf-8 -*-
"""
================================================================================
Auto-Survey Phase 1: 학습 데이터 생성 파이프라인 (Detailed Edition)
================================================================================

목표:
  Teacher LLM (gemma-4-31B-it) 으로 도메인-페르소나 다양성을 가진
  소비자 설문 학습 데이터를 자동 생성. Student 모델 QLoRA fine-tuning 입력.

4단계 파이프라인:
  Stage 1 (gen-domains):   다양한 소비자 조사 도메인 풀 100~150개 생성
  Stage 2 (gen-personas):  인구통계 + 라이프스타일 풍부한 페르소나 풀 생성
  Stage 3 (gen-surveys):   도메인 → 설문지 (도메인별 파일 1개)
  Stage 4 (gen-responses): (도메인 × 설문 × 페르소나) → 응답 → 학습 데이터

핵심 개선:
  ✓ 페르소나 = polarity + 인구통계 + 라이프스타일 (다층 구조)
  ✓ 도메인-페르소나 호환성 검증 (가격대, 연령 적합성)
  ✓ 응답 일관성 다층 검증 (척도/행동/가격민감도)
  ✓ 견고한 재시도 / 부분 실패 / 이어하기
  ✓ 비용 추적 (토큰 사용량)
  ✓ 진행 상황 풍부한 로깅
  ✓ 학습 데이터 통계 자동 분석
  ✓ data/ output/ logs/ 폴더 자동 분리

서버 폴더 구조 (기본값):
  ~/ACL/
    ├── code/    ← 이 스크립트 위치
    ├── data/    ← domains.json, personas.json, surveys/*.json
    ├── output/  ← train_data_*.jsonl  (Student 모델 학습 입력)
    └── logs/    ← gen_log_*.jsonl, stats_*.json, analysis_*.json

  환경변수로 변경 가능:
    export ACL_BASE=/path/to/your/work
    또는 명령마다 --data-dir, --output-root-dir, --logs-root-dir 지정.

실행 예시 (기본 경로 사용):
  # 1. 도메인 100개 → ~/ACL/data/domains.json
  python auto_survey_pipeline.py gen-domains --n 100

  # 2. 페르소나 50명 → ~/ACL/data/personas.json
  python auto_survey_pipeline.py gen-personas --n-per-polarity 10

  # 3. 설문지 → ~/ACL/data/surveys/*.json
  python auto_survey_pipeline.py gen-surveys

  # 4. 응답 → ~/ACL/output/train_data_*.jsonl
  #         → ~/ACL/logs/{gen_log,stats}_*.jsonl
  python auto_survey_pipeline.py gen-responses --n-per-polarity 3

  # 5. 분석 → ~/ACL/logs/analysis_*.json
  python auto_survey_pipeline.py analyze \\
      --jsonl ~/ACL/output/train_data_XXX.jsonl
================================================================================
"""

import argparse
import glob
import json
import os
import re
import random
import time
import sys
import traceback
from datetime import datetime
from collections import defaultdict, Counter
from typing import Optional

import requests


# ════════════════════════════════════════════════════════════════════════════
# 0. 전역 설정
# ════════════════════════════════════════════════════════════════════════════

LLAMA_SERVER = "http://127.0.0.1:8080"
MODEL_NAME   = "gemma-4-31b-it"

# 기본 작업 디렉토리 (사용자 홈 기준 ~/ACL)
#   ACL_BASE/
#     ├── code/    ← 스크립트 위치 (참고용)
#     ├── data/    ← domains.json, personas.json, surveys/ (입력/중간 산출)
#     ├── output/  ← train_data_*.jsonl, analyze 결과 (최종 산출)
#     └── logs/    ← gen_log_*.jsonl, stats_*.json (로그/통계)
ACL_BASE   = os.path.expanduser(os.environ.get("ACL_BASE",   "~/ACL"))
ACL_DATA   = os.path.expanduser(os.environ.get("ACL_DATA",   os.path.join(ACL_BASE, "data")))
ACL_OUTPUT = os.path.expanduser(os.environ.get("ACL_OUTPUT", os.path.join(ACL_BASE, "output")))
ACL_LOGS   = os.path.expanduser(os.environ.get("ACL_LOGS",   os.path.join(ACL_BASE, "logs")))

# 표준 파일/폴더명
DOMAINS_FILE   = "domains.json"
PERSONAS_FILE  = "personas.json"
SURVEYS_SUBDIR = "surveys"

POLARITY_LIST = ["strongly_negative", "weakly_negative", "neutral",
                 "weakly_positive", "strongly_positive"]

# Expected mean of ORDINAL (Likert) items, assuming a 5-point scale where
# 1 = most negative stance and 5 = most positive stance (polarity-aligned).
POLARITY_EXPECTED_MEAN = {
    "strongly_negative": 1.4,
    "weakly_negative":   2.3,
    "neutral":           3.0,
    "weakly_positive":   3.7,
    "strongly_positive": 4.6,
}

# Expected std of ORDINAL items (too little variance looks unnatural/monotonic).
POLARITY_EXPECTED_STD = {
    "strongly_negative": 0.6,
    "weakly_negative":   0.8,
    "neutral":           1.0,
    "weakly_positive":   0.8,
    "strongly_positive": 0.6,
}

# Demographic variables (US adult population)
AGE_GROUPS    = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
GENDERS       = ["Male", "Female", "Non-binary"]
INCOME_LEVELS = ["Under $25k", "$25k-$50k", "$50k-$75k",
                 "$75k-$100k", "$100k-$150k", "$150k+"]
HOUSEHOLDS    = ["Living alone", "Couple, no children", "Family with young children",
                 "Family with teenagers", "Multigenerational household", "Other"]
REGIONS       = ["Northeast", "Midwest", "South", "West"]
LIFESTYLES    = [
    "Health & wellness focused", "Budget-conscious", "Trend-sensitive",
    "Traditional/conservative", "Environmentally & ethically conscious",
    "Convenience-seeking", "Premium-oriented", "DIY/hands-on",
    "Digital native", "Prefers in-person/offline",
]


# ════════════════════════════════════════════════════════════════════════════
# 1. 비용 추적
# ════════════════════════════════════════════════════════════════════════════

class CostTracker:
    """LLM 호출 비용/시간 추적"""
    def __init__(self):
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.elapsed_total = 0.0
        self.errors = 0

    def record(self, tokens_in: int, tokens_out: int, elapsed: float):
        self.calls += 1
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.elapsed_total += elapsed

    def record_error(self):
        self.errors += 1

    def summary(self) -> dict:
        avg_t = self.elapsed_total / self.calls if self.calls else 0
        return {
            "n_calls":        self.calls,
            "n_errors":       self.errors,
            "tokens_in":      self.tokens_in,
            "tokens_out":     self.tokens_out,
            "tokens_total":   self.tokens_in + self.tokens_out,
            "elapsed_total":  round(self.elapsed_total, 1),
            "avg_call_sec":   round(avg_t, 2),
        }


COST = CostTracker()


# ════════════════════════════════════════════════════════════════════════════
# 2. LLM 호출 (재시도 / 비용 추적)
# ════════════════════════════════════════════════════════════════════════════

def call_llm(messages: list,
             temperature: float = 0.7,
             max_tokens: int = 4096,
             server: Optional[str] = None,
             model: Optional[str] = None,
             retry: int = 3,
             retry_delay: float = 2.0) -> str:
    """
    LLM 호출 + 자동 재시도 + 비용 추적.
    실패 시 마지막 예외를 그대로 raise.
    """
    s = server or LLAMA_SERVER
    m = model  or MODEL_NAME
    last_err = None

    for attempt in range(retry):
        t0 = time.time()
        try:
            resp = requests.post(
                f"{s}/v1/chat/completions",
                json={
                    "model":       m,
                    "messages":    messages,
                    "max_tokens":  max_tokens,
                    "temperature": temperature,
                    "top_p":       0.95,   # gemma-4 권장
                    "top_k":       64,     # gemma-4 권장
                },
                timeout=600,
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = time.time() - t0

            # 응답 파싱
            choice = data["choices"][0]
            raw    = choice.get("message", {})
            content = (raw.get("content", "")
                       or raw.get("reasoning_content", "")
                       or choice.get("text", ""))

            # ─── thinking 토큰 / channel 태그 제거 ───
            # gemma-4 모델 카드 형식 (비대칭):
            #   <|channel>thought\n[reasoning]<channel|>[final answer]
            # disabled thinking 모드에서도 빈 thought 블록은 항상 출력됨:
            #   <|channel>thought\n<channel|>[final answer]
            # 다양한 변형 (|, |> 위치 다른 경우)까지 포착해야 함

            # 1) <think>...</think> (EXAONE/Qwen 스타일)
            while "<think>" in content:
                st = content.find("<think>")
                et = content.find("</think>", st)
                content = content[:st] + (content[et + 8:] if et != -1 else "")

            # 2) gemma-4 channel/thought 블록 통째로 제거
            #    여는 태그: <|channel> 또는 <|channel|>
            #    "thought\n" 이후 ~~~ 닫는 태그까지
            #    닫는 태그: <channel|> 또는 <|channel|> 또는 <|channel>
            content = re.sub(
                r"<\|?channel\|?>\s*thought.*?<\|?channel\|?>",
                "",
                content,
                flags=re.DOTALL | re.IGNORECASE,
            )

            # 3) <|think|> ... <|/think|> 형식 (다른 변형)
            content = re.sub(
                r"<\|think\|?>.*?<\|/?think\|?>",
                "",
                content,
                flags=re.DOTALL | re.IGNORECASE,
            )

            # 4) 잔여 채널/특수 토큰 제거 (단독으로 남은 것)
            #    <|channel>, <channel|>, <|channel|>, <|think|>, <|...|>, <|...> 등
            content = re.sub(r"<\|?(?:channel|think|thought|message|start|end)\|?>",
                             "", content, flags=re.IGNORECASE)

            # 5) 일반 <|...|> 형태 제어 토큰 (이름이 무엇이든)
            content = re.sub(r"<\|[^>\n]{0,40}\|?>", "", content)

            # 6) thought 라는 단어가 라인 시작 + 텍스트 잔여물로 남아있으면 제거
            #    예: "thought\nLet me think about...\n\n{...}" → JSON만 남기기
            #    JSON 시작('{')이 있으면 그 이전까지의 자유 텍스트 무시
            #    (단, 이건 너무 공격적이므로 extract_json_object에서 처리)

            content = content.strip()

            # 비용 기록
            usage = data.get("usage", {})
            COST.record(
                tokens_in=usage.get("prompt_tokens", 0),
                tokens_out=usage.get("completion_tokens", 0),
                elapsed=elapsed,
            )
            return content.strip()

        except Exception as e:
            last_err = e
            COST.record_error()
            if attempt < retry - 1:
                time.sleep(retry_delay * (attempt + 1))  # exponential backoff
            else:
                raise last_err

    raise RuntimeError(f"LLM call failed after {retry} retries: {last_err}")


# ════════════════════════════════════════════════════════════════════════════
# 3. JSON 추출 / 파일명 정규화
# ════════════════════════════════════════════════════════════════════════════

def extract_json_object(text: str) -> Optional[dict]:
    """
    LLM 응답 문자열에서 JSON 객체 추출.
    여러 형식 대응:
      - 그냥 {...}
      - ```json ... ``` 코드블록 안
      - <|channel|>thought ... <|channel|> 같은 thinking 토큰 잔여물
      - 응답 앞뒤에 설명 문장 섞임
      - 마지막 } 뒤에 후행 텍스트 (자르기)
    """
    if not text or not isinstance(text, str):
        return None

    # 코드 블록 마커, thinking/channel 토큰 정리 (call_llm에서 한번 했지만 안전장치)
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"```", "", cleaned)
    # gemma-4 channel/thought 블록
    cleaned = re.sub(r"<\|?channel\|?>\s*thought.*?<\|?channel\|?>",
                     "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    # <think>...</think>
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    # 잔여 특수 토큰
    cleaned = re.sub(r"<\|?(?:channel|think|thought|message|start|end)\|?>",
                     "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<\|[^>\n]{0,40}\|?>", "", cleaned)
    cleaned = cleaned.strip()

    # 첫 번째 '{' 찾기
    s = cleaned.find("{")
    if s == -1:
        return None

    # 균형 잡힌 닫는 괄호 찾기 (string 내부의 { } 무시)
    depth = 0
    in_str = False
    escape = False
    for i in range(s, len(cleaned)):
        c = cleaned[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[s:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # JSON 파싱 실패 시 흔한 오류 자동 수정 시도
                    fixed = _try_fix_json(candidate)
                    if fixed is not None:
                        return fixed
                    # 다음 '}' 시도 (혹시 중첩이 잘못 계산됐다면)
                    continue

    # 닫는 괄호 못 찾으면: 잘렸을 수 있음 → 마지막 } 까지 한번 더 시도
    last_close = cleaned.rfind("}")
    if last_close > s:
        candidate = cleaned[s:last_close + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            fixed = _try_fix_json(candidate)
            if fixed is not None:
                return fixed

    # 최종 fallback: max_tokens 초과로 잘린 설문지 응답 복구
    # questions 배열의 마지막 완전한 문항까지만 살림
    return _try_recover_truncated_survey(cleaned, s)


def _try_recover_truncated_survey(cleaned: str, start: int) -> Optional[dict]:
    """
    설문지 JSON이 questions 배열 중간에서 잘렸을 때, 마지막 완전한 문항까지만
    살려서 재구성. 'questions': [ ... 까지는 정상이지만 그 안의 마지막 문항이
    잘린 경우 흔히 발생.

    전략: questions 배열에서 균형 잡힌 { ... } 객체를 하나씩 추출.
    """
    # surveyTitle, surveyDescription 추출
    title_m = re.search(r'"surveyTitle"\s*:\s*"([^"]+)"', cleaned)
    desc_m  = re.search(r'"surveyDescription"\s*:\s*"([^"]+)"', cleaned)
    title   = title_m.group(1) if title_m else "(recovered survey)"
    desc    = desc_m.group(1)  if desc_m  else ""

    # questions 배열 시작 위치 찾기
    q_start_m = re.search(r'"questions"\s*:\s*\[', cleaned)
    if not q_start_m:
        return None
    q_start = q_start_m.end()  # '[' 바로 다음

    # 배열 안의 균형 잡힌 { ... } 객체들을 하나씩 파싱
    questions = []
    i = q_start
    n = len(cleaned)
    while i < n:
        # 다음 '{' 찾기
        while i < n and cleaned[i] not in '{':
            if cleaned[i] == ']':
                i = n  # 배열 끝
                break
            i += 1
        if i >= n:
            break

        # { 부터 균형 잡힌 } 까지 추출
        depth = 0
        in_str = False
        escape = False
        obj_start = i
        ended = False
        for j in range(i, n):
            c = cleaned[j]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    obj_str = cleaned[obj_start:j+1]
                    try:
                        q_obj = json.loads(obj_str)
                        questions.append(q_obj)
                    except json.JSONDecodeError:
                        # 후행 콤마 등 시도
                        try:
                            fixed = re.sub(r",(\s*[}\]])", r"\1", obj_str)
                            q_obj = json.loads(fixed)
                            questions.append(q_obj)
                        except json.JSONDecodeError:
                            pass
                    i = j + 1
                    ended = True
                    break
        if not ended:
            # 마지막 문항이 잘려서 } 못 찾음
            break

    if len(questions) < 5:  # 최소 5문항도 못 살리면 포기
        return None

    return {
        "surveyTitle":       title,
        "surveyDescription": desc,
        "questions":         questions,
    }


def _try_fix_json(text: str) -> Optional[dict]:
    """JSON 파싱 실패 시 흔한 오류 자동 수정"""
    # 1) 후행 콤마 제거 ("key": "value",})
    fixed = re.sub(r",(\s*[}\]])", r"\1", text)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # 2) 한국어 따옴표(", ") → 일반 따옴표
    fixed = fixed.replace("\u201c", '"').replace("\u201d", '"')
    fixed = fixed.replace("\u2018", "'").replace("\u2019", "'")
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # 3) ★ gemma-4 흔한 실수: "optionText" 키 누락
    #    {"optionOrder": 2, "20대"}  →  {"optionOrder": 2, "optionText": "20대"}
    #    숫자/콤마 뒤 곧바로 따옴표 시작하는 string 값을 찾아서 키 추가
    fixed = re.sub(
        r'("optionOrder"\s*:\s*\d+\s*,\s*)(?!"[a-zA-Z_])"',
        r'\1"optionText": "',
        fixed,
    )
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # 4) ★ 더 일반적: 숫자/콤마 뒤에 키 없이 string 값만 있는 경우
    #    {key: 1, "value"} 형태 → 키가 빠진 string 값에 임시 키 부여는 위험하므로
    #    여기서는 optionText만 한정해서 처리
    return None


def extract_json_array(text: str) -> Optional[list]:
    """LLM 응답에서 첫 번째 JSON 배열 추출 (channel/thinking 토큰 정리 포함)"""
    if not text or not isinstance(text, str):
        return None

    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"```", "", cleaned)
    # gemma-4 channel/thought 블록
    cleaned = re.sub(r"<\|?channel\|?>\s*thought.*?<\|?channel\|?>",
                     "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<\|?(?:channel|think|thought|message|start|end)\|?>",
                     "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<\|[^>\n]{0,40}\|?>", "", cleaned).strip()

    s = cleaned.find("[")
    e = cleaned.rfind("]")
    if s == -1 or e == -1 or e < s:
        return None
    try:
        return json.loads(cleaned[s:e + 1])
    except json.JSONDecodeError:
        # 후행 콤마 등 자동 수정 시도
        candidate = cleaned[s:e + 1]
        fixed = re.sub(r",(\s*[}\]])", r"\1", candidate)
        fixed = fixed.replace("\u201c", '"').replace("\u201d", '"')
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None


def safe_filename(name: str, max_len: int = 80) -> str:
    """파일명에 사용 불가능한 문자 정규화"""
    name = re.sub(r"[\\/:*?\"<>|\s]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:max_len] or "unknown"


# ════════════════════════════════════════════════════════════════════════════
# Stage 1: 도메인 풀 생성
# ════════════════════════════════════════════════════════════════════════════

DOMAIN_GEN_SYSTEM = """You are an expert survey research planner. \
Produce a diverse pool of realistic survey domains that real research \
organizations might field, spanning MANY topic areas (not just consumer goods).

[Coverage rules - spread domains across these survey_types]
- consumer_product   (~30%): products/services people buy
    e.g. "ready-to-eat ramen kits", "premium wireless earbuds", "budget airlines"
- social_political   (~20%): public-opinion / policy attitudes
    e.g. "remote work mandates", "urban congestion pricing", "AI regulation"
- values_lifestyle   (~15%): personal values, beliefs, lifestyle attitudes
    e.g. "work-life balance priorities", "attitudes toward marriage", "religiosity"
- public_service     (~15%): evaluation of public/community services
    e.g. "local public transit", "city park quality", "DMV experience"
- workplace          (~10%): job / organization attitudes
    e.g. "employee engagement", "satisfaction with management", "hybrid work"
- media_content      (~10%): media, platforms, content
    e.g. "a streaming service", "short-form video apps", "podcast platforms"

[For each domain provide]
- domain        : a concrete, specific topic (avoid vague categories like "food")
- survey_type   : one of consumer_product | social_political | values_lifestyle |
                  public_service | workplace | media_content
- scale_label   : what the ORDINAL (Likert) items will primarily measure, one of:
                  "satisfaction" | "agreement" | "approval" | "favorability" | "frequency"
- price_range   : ONLY meaningful for consumer_product
                  ("budget" | "mid-low" | "mid" | "premium" | "luxury");
                  use "N/A" for all other survey_types
- description   : 1 sentence describing the topic

[Output: a JSON array ONLY, no prose]
[
  {"domain": "premium wireless earbuds", "survey_type": "consumer_product", "scale_label": "satisfaction", "price_range": "premium", "description": "High-end true-wireless earbuds with ANC."},
  {"domain": "urban congestion pricing", "survey_type": "social_political", "scale_label": "approval", "price_range": "N/A", "description": "Charging fees to drive into city centers to reduce traffic."},
  {"domain": "work-life balance priorities", "survey_type": "values_lifestyle", "scale_label": "agreement", "price_range": "N/A", "description": "How people weigh career against personal life."}
]"""


def gen_domains(n: int, server: str, model: str,
                temperature: float = 0.9,
                batch_size: int = 25,
                max_extra_batches: int = 5,
                save_path: Optional[str] = None) -> list[dict]:
    """
    도메인 풀 생성 (배치 단위 + 부족분 자동 보충 + 실시간 저장).

    save_path가 주어지면 매 배치마다 누적 결과를 JSON으로 저장.
    중간에 끊겨도 지금까지 만든 도메인은 디스크에 안전하게 보존됨.
    """
    domains: list[dict] = []
    seen: set = set()
    base_batches = (n + batch_size - 1) // batch_size
    max_total_batches = base_batches + max_extra_batches
    b_idx = 0

    # 출력 폴더 미리 만들기
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    while len(domains) < n and b_idx < max_total_batches:
        target = min(batch_size, n - len(domains))
        if target <= 0:
            break

        avoid_str = ", ".join(sorted(seen)[:30]) if seen else "(없음)"
        user_msg = (
            f"Generate {target} domains.\n"
            f"Do NOT duplicate any of these existing domains: {avoid_str}\n\n"
            f"Spread them across survey_types per the coverage rules. "
            f"Output a JSON array ONLY, no prose."
        )

        try:
            raw = call_llm(
                [{"role": "system", "content": DOMAIN_GEN_SYSTEM},
                 {"role": "user",   "content": user_msg}],
                temperature=temperature,
                max_tokens=4096,
                server=server,
                model=model,
            )
        except Exception as e:
            print(f"  [경고] 배치 {b_idx + 1} 호출 실패: {e}")
            b_idx += 1
            continue

        batch = extract_json_array(raw)
        if not batch:
            print(f"  [경고] 배치 {b_idx + 1}: JSON 배열 파싱 실패")
            b_idx += 1
            continue

        added = 0
        for d in batch:
            if not isinstance(d, dict) or "domain" not in d:
                continue
            name = d["domain"].strip()
            if name in seen:
                continue
            seen.add(name)
            st = d.get("survey_type", "consumer_product").strip()
            domains.append({
                "domain":      name,
                "survey_type": st,
                "scale_label": d.get("scale_label", "satisfaction").strip(),
                "category":    st,  # alias kept for backward-compat (analyze, etc.)
                "price_range": d.get("price_range", "N/A").strip(),
                "description": d.get("description", "").strip(),
            })
            added += 1

        progress = f"신규 {added} / 누적 {len(domains)}/{n}"
        if b_idx >= base_batches:
            progress += " [보충 배치]"
        print(f"  배치 {b_idx + 1}: {progress}")

        # ★ 실시간 저장 (배치 1개 끝날 때마다)
        if save_path and added > 0:
            _atomic_write_json(save_path, domains)

        b_idx += 1

    if len(domains) < n:
        print(f"  ⚠ 목표 {n}개 미달, 최종 {len(domains)}개로 진행")

    # 마지막에도 한 번 저장 (안전장치)
    if save_path:
        _atomic_write_json(save_path, domains[:n])

    return domains[:n]


def _atomic_write_json(path: str, data) -> None:
    """원자적 쓰기 - 임시파일 → 이름 변경. 쓰기 도중 끊겨도 기존 파일 보존."""
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception as e:
        # 저장 실패해도 메인 작업은 계속
        print(f"  [경고] {path} 저장 실패: {e}")
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def cmd_gen_domains(args):
    # 기본 경로: $ACL_DATA/domains.json
    output = args.output or os.path.join(args.data_dir, DOMAINS_FILE)

    print(f"\n{'═' * 70}")
    print(f"  [Stage 1] 도메인 풀 생성 (목표 n={args.n})")
    print(f"  Teacher: {args.model} @ {args.server}")
    print(f"  출력 경로: {output}  (배치마다 실시간 저장)")
    print(f"{'═' * 70}\n")

    t0 = time.time()
    domains = gen_domains(args.n, args.server, args.model,
                          temperature=args.temperature,
                          save_path=output)   # ★ 실시간 저장

    # 카테고리/가격대 분포
    cat_dist = Counter(d["category"] for d in domains)
    pr_dist  = Counter(d["price_range"] for d in domains)

    print(f"\n{'─' * 70}")
    print(f"  완료! {len(domains)}개 / 소요 {time.time() - t0:.0f}초")
    print(f"  저장: {output}")
    print(f"\n  카테고리 분포 (top 10):")
    for c, n in cat_dist.most_common(10):
        print(f"    {c:20s}: {n}")
    print(f"\n  가격대 분포:")
    for p, n in pr_dist.most_common():
        print(f"    {p:10s}: {n}")
    print(f"\n  비용: {COST.summary()}")


# ════════════════════════════════════════════════════════════════════════════
# Stage 2: 페르소나 풀 생성 (polarity × 인구통계 × 라이프스타일)
# ════════════════════════════════════════════════════════════════════════════

PERSONA_GEN_SYSTEM = """You are an expert survey-respondent persona designer. \
Create concrete, diverse US adult personas that match a given polarity \
(overall attitude tendency toward the survey topic).

[Persona fields]
- name (a common English first + last name)
- age_group, gender, income_level, household, region
- occupation (specific)
- lifestyle_tags (2-3)
- background (2-3 sentences: daily life, habits, values)
- attitude (1-2 sentences on what this person cares about / how they form opinions)
- pain_points (2-3 typical gripes or sources of (dis)satisfaction matching the polarity)

[Polarity = overall attitude tendency]
Polarity is general and applies to ANY survey topic (a product's quality,
a policy, a value statement, a service, etc.), not just consumer satisfaction:
- "strongly_negative": strong skepticism / opposition / past bad experiences, demanding standards
- "weakly_negative": mild skepticism, cautious, value-for-money minded
- "neutral": average respondent, middle-of-the-road
- "weakly_positive": generally favorable, moderate trust, willing to recommend/support
- "strongly_positive": enthusiastic advocate, loyal, vocal supporter

[Output: a JSON array ONLY]
[
  {
    "name": "Michael Reed",
    "polarity": "strongly_negative",
    "age_group": "35-44",
    "gender": "Male",
    "income_level": "$75k-$100k",
    "household": "Family with young children",
    "region": "Northeast",
    "occupation": "IT project manager",
    "lifestyle_tags": ["Budget-conscious", "Health & wellness focused"],
    "background": "Dual-income parent of a 5-year-old who scrutinizes value and safety. Deeply skeptical of marketing claims and distrusts social-media hype. Has been burned several times by overhyped products.",
    "attitude": "Checks labels, sources, and independent reviews; only trusts things that are proven.",
    "pain_points": ["Quality falls short of the advertising", "Poor value for the price", "Hidden downsides surface later"]
  }
]"""


def gen_personas(n_per_polarity: int, server: str, model: str,
                 temperature: float = 0.85,
                 save_path: Optional[str] = None) -> list[dict]:
    """
    polarity별 n_per_polarity개씩 다양한 페르소나 생성.

    save_path가 주어지면 매 배치마다 누적 결과를 JSON으로 저장.
    중간에 끊겨도 지금까지 만든 페르소나는 디스크에 안전하게 보존됨.
    """
    all_personas: list[dict] = []

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    def _save_progress():
        """글로벌 ID 매기고 저장"""
        if not save_path:
            return
        snapshot = []
        for i, p in enumerate(all_personas):
            p_copy = dict(p)
            p_copy["persona_id"] = f"P{i+1:04d}"
            snapshot.append(p_copy)
        _atomic_write_json(save_path, snapshot)

    for polarity in POLARITY_LIST:
        print(f"\n  [{polarity}] {n_per_polarity}개 생성 중...")
        polarity_personas: list[dict] = []
        seen_names: set = set()

        # 한 번에 5개씩 여러 번 호출 (다양성 확보)
        batch_size = 5
        n_batches = (n_per_polarity + batch_size - 1) // batch_size

        for b_idx in range(n_batches):
            target = min(batch_size, n_per_polarity - len(polarity_personas))
            if target <= 0:
                break

            # 다양성 시드: 인구통계 변수 조합 hint 제공
            hint_age   = random.sample(AGE_GROUPS, min(target, len(AGE_GROUPS)))
            hint_life  = random.sample(LIFESTYLES, min(3, len(LIFESTYLES)))

            user_msg = (
                f"Write {target} US adult personas with polarity = '{polarity}'.\n\n"
                f"[Diversity guide]\n"
                f"- Try to span these age groups: {hint_age}\n"
                f"- Candidate lifestyle tags: {hint_life}\n"
                f"- Vary occupation, income, and household type\n"
                f"- Avoid reusing these names: {sorted(seen_names)[:10] if seen_names else 'none'}\n\n"
                f"Output a JSON array ONLY."
            )

            try:
                raw = call_llm(
                    [{"role": "system", "content": PERSONA_GEN_SYSTEM},
                     {"role": "user",   "content": user_msg}],
                    temperature=temperature,
                    max_tokens=3072,
                    server=server,
                    model=model,
                )
            except Exception as e:
                print(f"    배치 {b_idx + 1} 실패: {e}")
                continue

            batch = extract_json_array(raw)
            if not batch:
                print(f"    배치 {b_idx + 1} JSON 파싱 실패")
                continue

            new_count = 0
            for p in batch:
                if not isinstance(p, dict):
                    continue
                if "name" not in p or "polarity" not in p:
                    continue
                p["polarity"] = polarity
                if p["name"] in seen_names:
                    continue
                seen_names.add(p["name"])
                polarity_personas.append(p)
                # 임시로 all_personas에 추가 (실시간 저장용)
                all_personas.append(p)
                new_count += 1

            print(f"    배치 {b_idx + 1}/{n_batches}: 누적 {len(polarity_personas)}/{n_per_polarity}")

            # ★ 실시간 저장 (배치 1개 끝날 때마다)
            if new_count > 0:
                _save_progress()

        # 이 polarity에서 초과분이 있으면 잘라내기 (그러려면 all_personas에서도 빼야 함)
        excess = len(polarity_personas) - n_per_polarity
        if excess > 0:
            for _ in range(excess):
                # 마지막에 추가된 polarity_personas 항목들이 all_personas의 마지막 항목들과 일치
                all_personas.pop()
                polarity_personas.pop()

        print(f"  → [{polarity}] 최종 {len(polarity_personas)}개")

    # 글로벌 ID 부여 (최종)
    for i, p in enumerate(all_personas):
        p["persona_id"] = f"P{i+1:04d}"

    # 마지막 한 번 더 저장 (안전장치)
    if save_path:
        _atomic_write_json(save_path, all_personas)

    return all_personas


def cmd_gen_personas(args):
    output = args.output or os.path.join(args.data_dir, PERSONAS_FILE)

    print(f"\n{'═' * 70}")
    print(f"  [Stage 2] 페르소나 풀 생성")
    print(f"  polarity별 {args.n_per_polarity}개 × {len(POLARITY_LIST)}종 = "
          f"{args.n_per_polarity * len(POLARITY_LIST)}개")
    print(f"  출력 경로: {output}  (배치마다 실시간 저장)")
    print(f"{'═' * 70}")

    t0 = time.time()
    personas = gen_personas(args.n_per_polarity, args.server, args.model,
                            temperature=args.temperature,
                            save_path=output)   # ★ 실시간 저장

    # 분포 분석
    by_polarity   = Counter(p["polarity"] for p in personas)
    by_age        = Counter(p.get("age_group", "?") for p in personas)
    by_gender     = Counter(p.get("gender", "?")    for p in personas)
    by_household  = Counter(p.get("household", "?") for p in personas)

    print(f"\n{'─' * 70}")
    print(f"  완료! {len(personas)}명 / 소요 {time.time() - t0:.0f}초")
    print(f"  저장: {output}\n")
    print(f"  polarity 분포:    {dict(by_polarity)}")
    print(f"  연령 분포:        {dict(by_age.most_common())}")
    print(f"  성별 분포:        {dict(by_gender)}")
    print(f"  가구형태 분포:    {dict(by_household.most_common())}")
    print(f"\n  비용: {COST.summary()}")


# ════════════════════════════════════════════════════════════════════════════
# Stage 3: 도메인 → 설문지 생성
# ════════════════════════════════════════════════════════════════════════════

SURVEY_GEN_SYSTEM = """You are an expert survey designer. Given a survey \
domain, write a professional, fieldable questionnaire.

[Questionnaire structure - 25 to 30 items total]
1. SQ (Screening, 1-2 items): awareness / experience screening (CATEGORICAL)
2. Awareness / exposure: 3-4 items (CATEGORICAL)
3. Core attitude items: 8-12 items (ORDINAL, 5-point Likert)
   - Adapt to the domain's scale_label:
     * satisfaction  -> 1=very dissatisfied ... 5=very satisfied
     * agreement     -> 1=strongly disagree ... 5=strongly agree (write items as positive statements)
     * approval      -> 1=strongly disapprove ... 5=strongly approve
     * favorability  -> 1=very unfavorable ... 5=very favorable
     * frequency     -> map to intensity, but still 5 points
4. Behavior / usage: 4-5 items (CATEGORICAL)
5. Loyalty / recommendation / future intent: 2-3 items (ORDINAL)
6. Improvements / complaints: 1-2 items (CATEGORICAL_MULTI)
7. Demographics: 2-3 items (CATEGORICAL) - age, gender, income

[ORDINAL scale rules - CRITICAL]
- Every ORDINAL item MUST be polarity-aligned: option 1 = the MOST NEGATIVE
  stance, option 5 = the MOST POSITIVE stance, regardless of scale_label.
- Exactly 5 options (optionOrder 1..5).
- Do not reverse-word items (avoid items where agreeing means a negative stance).

[Option rules]
- Option counts: ORDINAL=5, CATEGORICAL=3-7, CATEGORICAL_MULTI=4-8
- Options must be mutually exclusive
- An "Other" option only occasionally, and only in CATEGORICAL_MULTI
- Keep option text clear and short

[Output: exactly ONE JSON object, no surrounding prose]
{
  "surveyTitle": "...",
  "surveyDescription": "(1-2 sentences)",
  "questions": [
    {
      "questionOrder": 1,
      "questionText": "...",
      "questionType": "CATEGORICAL" | "CATEGORICAL_MULTI" | "ORDINAL",
      "category": "awareness" | "attitude" | "behavior" | "loyalty" | "improvement" | "demographics",
      "options": [
        {"optionOrder": 1, "optionText": "..."},
        ...
      ]
    }
  ]
}

[ABSOLUTE JSON rules]
- Every option MUST have BOTH keys: "optionOrder" and "optionText"
- Wrong: {"optionOrder": 2, "18-24"}        <- missing optionText
- Right: {"optionOrder": 2, "optionText": "18-24"}
- No trailing comma after the last item
- All keys and string values use double quotes (") only"""


def validate_survey(survey: dict, min_q: int = 10, max_q: int = 35) -> tuple[bool, str]:
    """설문지 구조 검증 (잘림 복구된 경우도 10문항 이상이면 OK)"""
    if not isinstance(survey, dict):
        return False, "설문지가 dict 아님"
    if "questions" not in survey:
        return False, "questions 키 없음"
    qs = survey["questions"]
    if not isinstance(qs, list):
        return False, "questions가 list 아님"
    if not (min_q <= len(qs) <= max_q):
        return False, f"문항 수 {len(qs)} 이 [{min_q}, {max_q}] 범위 밖"

    valid_types = {"CATEGORICAL", "CATEGORICAL_MULTI", "ORDINAL"}
    seen_orders = set()

    for i, q in enumerate(qs):
        if not isinstance(q, dict):
            return False, f"문항{i+1} dict 아님"
        if "questionText" not in q:
            return False, f"문항{i+1} questionText 없음"
        if not q.get("questionText", "").strip():
            return False, f"문항{i+1} questionText 비어있음"

        opts = q.get("options")
        if not isinstance(opts, list) or len(opts) < 2:
            return False, f"문항{i+1} 보기 < 2"
        if len(opts) > 15:
            return False, f"문항{i+1} 보기 > 15 (너무 많음)"

        q_type = q.get("questionType", "CATEGORICAL")
        if q_type not in valid_types:
            return False, f"문항{i+1} type {q_type} 잘못됨"

        # ORDINAL은 5개여야 (예외 허용 범위 4~7)
        if q_type == "ORDINAL" and not (4 <= len(opts) <= 7):
            return False, f"문항{i+1} ORDINAL 보기 수 {len(opts)} (4~7 권장)"

        for j, o in enumerate(opts):
            if not isinstance(o, dict) or "optionText" not in o:
                return False, f"문항{i+1} 보기{j+1} 형식 오류"
            if not o.get("optionText", "").strip():
                return False, f"문항{i+1} 보기{j+1} 텍스트 비어있음"

        order = q.get("questionOrder", i + 1)
        if order in seen_orders:
            return False, f"questionOrder 중복: {order}"
        seen_orders.add(order)

    return True, f"OK ({len(qs)} 문항)"


def normalize_survey(survey: dict) -> dict:
    """설문지 구조 정규화 (필드 누락 보완 등)"""
    cleaned_qs = []
    for i, q in enumerate(survey["questions"], 1):
        cleaned_opts = []
        for j, o in enumerate(q["options"], 1):
            cleaned_opts.append({
                "optionOrder": int(o.get("optionOrder", j)),
                "optionText":  str(o["optionText"]).strip(),
            })
        cleaned_qs.append({
            "questionOrder": int(q.get("questionOrder", i)),
            "questionText":  str(q["questionText"]).strip(),
            "questionType":  q.get("questionType", "CATEGORICAL"),
            "category":      q.get("category", ""),
            "options":       cleaned_opts,
        })
    return {
        "surveyTitle":       survey.get("surveyTitle", "").strip(),
        "surveyDescription": survey.get("surveyDescription", "").strip(),
        "questions":         cleaned_qs,
    }


def gen_survey_for_domain(domain_info: dict, server: str, model: str,
                          temperature: float = 0.7,
                          retry: int = 3,
                          debug_dir: Optional[str] = None,
                          domain_idx: int = 0) -> tuple[Optional[dict], str]:
    """도메인 → 설문지 (재시도 + raw 응답 저장)"""
    user_msg = (
        f"[Domain]\n"
        f"name: {domain_info['domain']}\n"
        f"survey_type: {domain_info.get('survey_type', domain_info.get('category', ''))}\n"
        f"scale_label: {domain_info.get('scale_label', 'satisfaction')}\n"
        f"price_range: {domain_info.get('price_range', 'N/A')}\n"
        f"description: {domain_info.get('description', '')}\n\n"
        f"Write a 25-30 item questionnaire for this domain as JSON.\n"
        f"Match the ORDINAL items to scale_label, and keep every ORDINAL item "
        f"polarity-aligned (1=most negative, 5=most positive).\n\n"
        f"IMPORTANT: output exactly ONE JSON object. No code fences (```), "
        f"no explanation, no preamble. Start with '{{' and end with '}}'."
    )

    last_raw = ""
    last_reason = ""

    for attempt in range(retry):
        try:
            raw = call_llm(
                [{"role": "system", "content": SURVEY_GEN_SYSTEM},
                 {"role": "user",   "content": user_msg}],
                temperature=temperature + (attempt * 0.1),  # 재시도마다 온도↑
                max_tokens=60000,  # 잘림 방지로 넉넉히 (서버 -c 131072라 여유)
                server=server,
                model=model,
            )
            last_raw = raw
        except Exception as e:
            last_reason = f"LLM 호출 실패: {e}"
            continue

        survey = extract_json_object(raw)
        if not survey:
            last_reason = "JSON 파싱 실패"
            # 디버깅: raw 응답을 파일로 저장 (첫 번째 실패만)
            if debug_dir and attempt == 0:
                try:
                    os.makedirs(debug_dir, exist_ok=True)
                    debug_path = os.path.join(
                        debug_dir,
                        f"raw_{domain_idx:04d}_{safe_filename(domain_info['domain'])}.txt"
                    )
                    with open(debug_path, "w", encoding="utf-8") as f:
                        f.write(f"=== Domain ===\n{domain_info['domain']}\n\n")
                        f.write(f"=== Attempt {attempt + 1} (temp={temperature + attempt * 0.1:.2f}) ===\n")
                        f.write(f"=== Raw response ({len(raw)} chars) ===\n\n")
                        f.write(raw)
                except Exception:
                    pass
            continue

        ok, reason = validate_survey(survey)
        if ok:
            return normalize_survey(survey), reason
        last_reason = reason

    return None, f"{retry}회 재시도 실패 (마지막 사유: {last_reason})"


def cmd_gen_surveys(args):
    """도메인별 설문지 생성"""
    # 기본 경로 자동 매핑
    domains_path = args.domains    or os.path.join(args.data_dir, DOMAINS_FILE)
    output_dir   = args.output_dir or os.path.join(args.data_dir, SURVEYS_SUBDIR)

    with open(domains_path, encoding="utf-8") as f:
        domains = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'═' * 70}")
    print(f"  [Stage 3] 설문지 생성 ({len(domains)} 도메인)")
    print(f"  도메인 입력: {domains_path}")
    print(f"  출력 폴더:   {output_dir}")
    print(f"  이어하기: {'ON (이미 존재 = 스킵)' if not args.overwrite else 'OFF (덮어쓰기)'}")
    print(f"{'═' * 70}\n")

    ok_n, fail_n, skip_n = 0, 0, 0
    fail_log = []

    t0 = time.time()
    for d_idx, domain in enumerate(domains, 1):
        fname = f"{d_idx:04d}_{safe_filename(domain['domain'])}.json"
        out_path = os.path.join(output_dir, fname)

        # 이어하기
        if os.path.exists(out_path) and not args.overwrite:
            print(f"  [{d_idx:>4}/{len(domains)}] {domain['domain'][:35]:<35s} → 스킵 (이미 존재)")
            skip_n += 1
            continue

        survey, reason = gen_survey_for_domain(
            domain, args.server, args.model,
            temperature=args.temperature, retry=args.retry,
            debug_dir=os.path.join(output_dir, "_debug_raw"),
            domain_idx=d_idx,
        )

        if not survey:
            print(f"  [{d_idx:>4}/{len(domains)}] {domain['domain'][:35]:<35s} ✗ {reason}")
            fail_n += 1
            fail_log.append({"domain": domain["domain"], "reason": reason})
            continue

        record = {
            "domain":            domain["domain"],
            "survey_type":       domain.get("survey_type", domain.get("category", "")),
            "scale_label":       domain.get("scale_label", "satisfaction"),
            "category":          domain.get("category", ""),
            "price_range":       domain.get("price_range", ""),
            "description":       domain.get("description", ""),
            "surveyTitle":       survey["surveyTitle"],
            "surveyDescription": survey["surveyDescription"],
            "n_questions":       len(survey["questions"]),
            "questions":         survey["questions"],
            "generated_at":      datetime.now().isoformat(),
            "model":             args.model,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        # 문항 카테고리 요약
        cat_summary = Counter(q.get("category", "?") for q in survey["questions"])
        cat_str = " ".join(f"{k}:{v}" for k, v in cat_summary.most_common())
        print(f"  [{d_idx:>4}/{len(domains)}] {domain['domain'][:35]:<35s} "
              f"✓ {len(survey['questions'])}문항 ({cat_str})")
        ok_n += 1

    elapsed = time.time() - t0
    print(f"\n{'─' * 70}")
    print(f"  완료! 성공 {ok_n} / 실패 {fail_n} / 스킵 {skip_n}")
    print(f"  소요 {elapsed:.0f}초 / 평균 {elapsed/max(ok_n,1):.1f}초/도메인")

    if fail_log:
        fail_path = os.path.join(output_dir, "_failed_domains.json")
        with open(fail_path, "w", encoding="utf-8") as f:
            json.dump(fail_log, f, ensure_ascii=False, indent=2)
        print(f"  실패 로그: {fail_path}")

    print(f"\n  비용: {COST.summary()}")


# ════════════════════════════════════════════════════════════════════════════
# Stage 4: 응답 생성
# ════════════════════════════════════════════════════════════════════════════

RESPONSE_GEN_SYSTEM = """You simulate a survey respondent answering as a given persona.

[Response principles]
1. Fully inhabit the persona: reflect their background, values, and attitudes.
2. Keep polarity consistent throughout, but do NOT pick the same scale point
   for every item; allow natural variation.
   e.g. a 'strongly_negative' persona mostly picks 1-2, but a 3 may appear occasionally.
3. Demographic items (age/gender/income) MUST match the persona exactly.
4. If the persona is a poor fit for the topic (e.g. not in the relevant group),
   let that mismatch show in the answers.

[Output rules - MUST follow]
- Output exactly ONE JSON object. No prose, comments, or code fences.
- Keys are question-number strings ("1", "2", ...).
- Values MUST be in the form "optionNumber. optionText".
  e.g. "2. Somewhat dissatisfied", "3. Occasionally"
- Answer EVERY question. None may be missing.
- For multi-select items, pick a single representative option (for training simplicity)."""


def build_response_prompt(survey: dict, persona: dict) -> list[dict]:
    """페르소나 + 설문 → user 메시지"""
    # 문항 텍스트 구성
    q_parts = []
    for q in survey["questions"]:
        opt_lines = "\n".join(
            f"  {o['optionOrder']}. {o['optionText']}"
            for o in q["options"]
        )
        q_parts.append(
            f"[Q{q['questionOrder']}] ({q.get('questionType', 'CATEGORICAL')})\n"
            f"Question: {q['questionText']}\n"
            f"Options:\n{opt_lines}"
        )

    # Persona card
    persona_card = (
        f"=== Persona ===\n"
        f"Name: {persona.get('name', 'Anonymous')}\n"
        f"Polarity: {persona['polarity']}\n"
        f"Age: {persona.get('age_group', '?')}\n"
        f"Gender: {persona.get('gender', '?')}\n"
        f"Income: {persona.get('income_level', '?')}\n"
        f"Household: {persona.get('household', '?')}\n"
        f"Region: {persona.get('region', '?')}\n"
        f"Occupation: {persona.get('occupation', '?')}\n"
        f"Lifestyle: {', '.join(persona.get('lifestyle_tags', []))}\n"
        f"\nBackground:\n{persona.get('background', '')}\n"
        f"\nAttitude:\n{persona.get('attitude', persona.get('consumption_attitude', ''))}\n"
        f"\nKey pain/satisfaction points:\n"
        + "\n".join(f"- {pp}" for pp in persona.get('pain_points', []))
    )

    example = '{"1": "2. Occasionally", "2": "1. Very dissatisfied", ...}'

    user = (
        f"{persona_card}\n\n"
        f"Answer the following survey as this persona. Output a JSON object ONLY.\n"
        f"Value format: \"number. text\"\n\n"
        f"Example: {example}\n\n"
        + "\n\n".join(q_parts)
    )

    return [
        {"role": "system", "content": RESPONSE_GEN_SYSTEM},
        {"role": "user",   "content": user},
    ]


# 응답 value 파싱 패턴
_NUMBER_DOT = re.compile(r"^\s*(\d+)\s*[.)\]:]\s*(.+?)\s*$")


def parse_response_value(value, options: list[dict]) -> Optional[tuple[int, str]]:
    """
    LLM이 반환한 value 문자열 → (code, text)
    여러 형식 지원:
      "2. 불만족하는 편이다"
      "2) 불만족하는 편이다"
      "2: 불만족하는 편이다"
      "2"
      "불만족하는 편이다"
    """
    if not isinstance(value, str):
        return None
    v = value.strip().strip("\"").strip("'")
    if not v:
        return None

    opt_map = {o["optionOrder"]: o["optionText"] for o in options}

    # 1) "번호. 텍스트" 형식
    m = _NUMBER_DOT.match(v)
    if m:
        code = int(m.group(1))
        if code in opt_map:
            return code, opt_map[code]

    # 2) 숫자만
    if v.isdigit():
        code = int(v)
        if code in opt_map:
            return code, opt_map[code]

    # 3) 텍스트 정확 매칭
    for code, text in opt_map.items():
        if v == text:
            return code, text

    # 4) 부분 매칭 (정확 매칭 안 될 때만)
    for code, text in opt_map.items():
        if v in text or text in v:
            return code, text

    return None


def polarity_fallback_code(polarity: str, options: list[dict],
                            q_type: str, q_category: str = "") -> int:
    """LLM 응답 누락 시 fallback (polarity + 문항 유형 기반)"""
    codes = sorted(o["optionOrder"] for o in options)
    n = len(codes)
    if n == 0:
        return 1

    # Demographic items: random (persona matching should be handled by the LLM)
    if q_category == "demographics":
        return random.choice(codes)

    if q_type == "ORDINAL":
        # polarity -> index mapping (assumes a 5-point, polarity-aligned scale)
        polarity_to_idx = {
            "strongly_negative": 0,
            "weakly_negative":   max(0, n // 4),
            "neutral":           n // 2,
            "weakly_positive":   min(n - 1, (3 * n) // 4),
            "strongly_positive": n - 1,
        }
        idx = polarity_to_idx.get(polarity, n // 2)
        # 약간의 노이즈 (인접 옵션 선택 가능성)
        if random.random() < 0.2 and 0 < idx < n - 1:
            idx += random.choice([-1, 1])
        return codes[idx]
    else:
        # CATEGORICAL/MULTI: 보통은 첫 옵션
        return codes[0]


def gen_responses_for_survey(survey: dict, persona: dict,
                              server: str, model: str,
                              temperature: float = 0.7,
                              retry: int = 2) -> tuple[list[dict], list, str]:
    """설문 + 페르소나 → 응답"""
    polarity = persona["polarity"]
    messages = build_response_prompt(survey, persona)
    raw = ""
    obj = None

    for attempt in range(retry):
        try:
            raw = call_llm(messages,
                           temperature=temperature + (attempt * 0.05),
                           max_tokens=4096,
                           server=server, model=model)
            obj = extract_json_object(raw)
            if obj:
                break
        except Exception as e:
            print(f"      [응답 생성 시도 {attempt + 1}] 오류: {e}")

    obj = obj or {}

    # 문항별 응답 매핑
    answers = []
    for q in survey["questions"]:
        q_order = q["questionOrder"]
        opt_map = {o["optionOrder"]: o["optionText"] for o in q["options"]}

        # LLM 응답에서 추출
        llm_value = obj.get(str(q_order)) or obj.get(q_order)
        parsed = parse_response_value(llm_value, q["options"]) if llm_value else None

        if parsed:
            code, text = parsed
            source = "llm"
        else:
            code = polarity_fallback_code(
                polarity, q["options"],
                q.get("questionType", "CATEGORICAL"),
                q.get("category", ""),
            )
            text = opt_map[code]
            source = "fallback"

        answers.append({
            "questionOrder":  q_order,
            "questionText":   q["questionText"],
            "questionType":   q.get("questionType", "CATEGORICAL"),
            "questionCategory": q.get("category", ""),
            "selected_code":  code,
            "selected_text":  text,
            "response_text":  f"{code}. {text}",
            "source":         source,
            "llm_value":      llm_value,
        })

    return answers, messages, raw


# ════════════════════════════════════════════════════════════════════════════
# 품질 필터 (다층 검증)
# ════════════════════════════════════════════════════════════════════════════

def compute_polarity_metrics(answers: list[dict], polarity: str) -> dict:
    """척도 응답에서 polarity 정합성 지표 계산"""
    ord_codes = [a["selected_code"] for a in answers
                 if a.get("questionType") == "ORDINAL"
                 and a.get("questionCategory") != "demographics"]
    if not ord_codes:
        return {"n_ordinal": 0, "mean": None, "std": None,
                "consistency": 1.0}  # 척도 없으면 패스

    n = len(ord_codes)
    mean = sum(ord_codes) / n
    if n > 1:
        var = sum((c - mean) ** 2 for c in ord_codes) / (n - 1)
        std = var ** 0.5
    else:
        std = 0.0

    expected_mean = POLARITY_EXPECTED_MEAN[polarity]
    expected_std  = POLARITY_EXPECTED_STD[polarity]

    # 평균 차이 점수 (0~1)
    diff = abs(mean - expected_mean)
    if diff <= 0.7:
        mean_score = 1.0
    elif diff <= 1.5:
        mean_score = 0.6
    elif diff <= 2.5:
        mean_score = 0.3
    else:
        mean_score = 0.0

    # 표준편차 점수 (너무 단조롭거나 너무 흩어져도 감점)
    if expected_std * 0.5 <= std <= expected_std * 2.0:
        std_score = 1.0
    elif std < expected_std * 0.3:
        # 분산 거의 없음 (단조)
        std_score = 0.4
    else:
        std_score = 0.6

    consistency = 0.7 * mean_score + 0.3 * std_score

    return {
        "n_ordinal":     n,
        "mean":          round(mean, 2),
        "std":           round(std, 2),
        "expected_mean": expected_mean,
        "expected_std":  expected_std,
        "mean_score":    round(mean_score, 2),
        "std_score":     round(std_score, 2),
        "consistency":   round(consistency, 2),
    }


def quality_filter(survey: dict, answers: list[dict], persona: dict,
                   min_consistency: float = 0.5,
                   max_fallback_ratio: float = 0.3,
                   min_unique_ratio: float = 0.12) -> tuple[bool, str, dict]:
    """
    학습 데이터 품질 다층 검증.

    검증 항목:
      1. 응답 개수 일치
      2. fallback 비율 (LLM 실패 비율)
      3. polarity 일관성 (척도 평균/분산)
      4. 응답 다양성 (모든 응답이 같은 코드면 의심)
    """
    metrics = {}

    # 1. 응답 개수
    if len(answers) != len(survey["questions"]):
        return False, "응답 개수 불일치", metrics

    # 2. fallback 비율
    fb_n = sum(1 for a in answers if a["source"] == "fallback")
    fb_ratio = fb_n / len(answers)
    metrics["fallback_ratio"] = round(fb_ratio, 2)
    if fb_ratio > max_fallback_ratio:
        return False, f"fallback 비율 {fb_ratio:.0%} 초과", metrics

    # 3. polarity 일관성
    pol_metrics = compute_polarity_metrics(answers, persona["polarity"])
    metrics.update(pol_metrics)
    if pol_metrics["consistency"] < min_consistency:
        return False, f"polarity 일관성 {pol_metrics['consistency']:.2f} 미달", metrics

    # 4. 응답 다양성 (척도 답변이 모두 같은 값이면 의심)
    ord_codes = [a["selected_code"] for a in answers
                 if a.get("questionType") == "ORDINAL"
                 and a.get("questionCategory") != "demographics"]
    if ord_codes:
        unique_ratio = len(set(ord_codes)) / len(ord_codes)
        metrics["ordinal_unique_ratio"] = round(unique_ratio, 2)
        if unique_ratio < min_unique_ratio:
            return False, f"척도 응답 다양성 {unique_ratio:.0%} 미달 (단조)", metrics

    return True, "OK", metrics


# ════════════════════════════════════════════════════════════════════════════
# QLoRA 학습 레코드
# ════════════════════════════════════════════════════════════════════════════

QLORA_SYSTEM = """You are an expert survey-research model. Given a domain and a \
respondent persona, construct a suitable 25-30 item questionnaire and generate \
persona-consistent answers. Output in JSON."""


def to_qlora_record(domain_info: dict, persona: dict, survey: dict,
                    answers: list[dict]) -> dict:
    """학습용 input/output 메시지 생성"""
    user_content = (
        f"[Domain]\n"
        f"name: {domain_info['domain']}\n"
        f"survey_type: {domain_info.get('survey_type', domain_info.get('category', ''))}\n"
        f"scale_label: {domain_info.get('scale_label', 'satisfaction')}\n"
        f"price_range: {domain_info.get('price_range', 'N/A')}\n"
        f"description: {domain_info.get('description', '')}\n\n"
        f"[Persona]\n"
        f"name: {persona.get('name', '')}\n"
        f"polarity: {persona['polarity']}\n"
        f"age: {persona.get('age_group', '')} / gender: {persona.get('gender', '')}\n"
        f"income: {persona.get('income_level', '')} / household: {persona.get('household', '')}\n"
        f"occupation: {persona.get('occupation', '')}\n"
        f"lifestyle: {', '.join(persona.get('lifestyle_tags', []))}\n"
        f"background: {persona.get('background', '')}\n"
        f"attitude: {persona.get('attitude', persona.get('consumption_attitude', ''))}\n\n"
        f"Build a 25-30 item questionnaire for this domain and generate this "
        f"persona's answers. Output a JSON object only."
    )

    output_obj = {
        "survey": {
            "title":       survey.get("surveyTitle", domain_info["domain"]),
            "description": survey.get("surveyDescription", ""),
            "questions":   survey["questions"],
        },
        "responses": {
            str(a["questionOrder"]): a["response_text"]
            for a in answers
        },
    }
    assistant_content = json.dumps(output_obj, ensure_ascii=False, indent=2)

    messages = [
        {"role": "system",    "content": QLORA_SYSTEM},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    text = "\n".join(
        f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>"
        for m in messages
    )
    return {"messages": messages, "text": text}


def cmd_gen_responses(args):
    """저장된 설문지 + 페르소나 → 응답 생성"""
    # 기본 경로 자동 매핑
    surveys_dir   = args.surveys_dir or os.path.join(args.data_dir, SURVEYS_SUBDIR)
    personas_path = args.personas    or os.path.join(args.data_dir, PERSONAS_FILE)
    output_dir    = args.output_dir  or args.output_root_dir   # 학습 데이터 (.jsonl)
    logs_dir      = args.logs_dir    or args.logs_root_dir     # 로그 + 통계

    survey_files = sorted(glob.glob(os.path.join(surveys_dir, "0*.json")))
    if not survey_files:
        print(f"[오류] 설문지 파일 없음: {surveys_dir}")
        return

    with open(personas_path, encoding="utf-8") as f:
        personas = json.load(f)

    # polarity 필터
    requested_polarities = (
        [p.strip() for p in args.polarities.split(",")]
        if args.polarities else POLARITY_LIST
    )
    requested_polarities = [p for p in requested_polarities if p in POLARITY_LIST]

    # polarity별 페르소나 그룹화
    personas_by_pol = defaultdict(list)
    for p in personas:
        if p["polarity"] in requested_polarities:
            personas_by_pol[p["polarity"]].append(p)

    print(f"\n{'═' * 70}")
    print(f"  [Stage 4] 응답 생성")
    print(f"  설문지 폴더: {surveys_dir}  ({len(survey_files)}개)")
    print(f"  페르소나:    {personas_path}")
    print(f"    - 총 {sum(len(v) for v in personas_by_pol.values())}명")
    for pol in requested_polarities:
        print(f"    - {pol}: {len(personas_by_pol[pol])}명")
    print(f"  설문지 × polarity별 반복: {args.n_per_polarity}회")
    n_target = (len(survey_files) * len(requested_polarities) * args.n_per_polarity)
    print(f"  목표 sample: ~{n_target}")
    print(f"\n  학습 데이터 출력: {output_dir}")
    print(f"  로그/통계 출력:   {logs_dir}")
    print(f"{'═' * 70}\n")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logs_dir,   exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_jsonl = os.path.join(output_dir, f"train_data_{ts}.jsonl")  # 학습 데이터
    out_log   = os.path.join(logs_dir,   f"gen_log_{ts}.jsonl")     # 생성 로그
    out_stats = os.path.join(logs_dir,   f"stats_{ts}.json")        # 통계

    stats = defaultdict(int)
    written = 0
    t0 = time.time()

    with open(out_jsonl, "w", encoding="utf-8", buffering=1) as f_out, \
         open(out_log,   "w", encoding="utf-8", buffering=1) as f_log:

        for s_idx, survey_path in enumerate(survey_files, 1):
            with open(survey_path, encoding="utf-8") as f:
                survey_data = json.load(f)

            domain_info = {
                "domain":      survey_data["domain"],
                "survey_type": survey_data.get("survey_type", survey_data.get("category", "")),
                "scale_label": survey_data.get("scale_label", "satisfaction"),
                "category":    survey_data.get("category", ""),
                "price_range": survey_data.get("price_range", ""),
                "description": survey_data.get("description", ""),
            }
            survey = {
                "surveyTitle":       survey_data.get("surveyTitle", domain_info["domain"]),
                "surveyDescription": survey_data.get("surveyDescription", ""),
                "questions":         survey_data["questions"],
            }

            print(f"\n[{s_idx}/{len(survey_files)}] {domain_info['domain']} "
                  f"({len(survey['questions'])}문항)")

            for polarity in requested_polarities:
                pool = personas_by_pol[polarity]
                if not pool:
                    print(f"  ⚠ {polarity} 페르소나 없음, 스킵")
                    continue

                for rep in range(args.n_per_polarity):
                    # 페르소나 무작위 선택 (rep 마다 다른 페르소나)
                    persona = random.choice(pool)

                    try:
                        answers, _, raw = gen_responses_for_survey(
                            survey, persona,
                            args.server, args.model,
                            temperature=args.temperature,
                            retry=2,
                        )
                    except Exception as e:
                        print(f"  ✗ {polarity} #{rep+1} ({persona.get('name')}): {type(e).__name__}: {e}")
                        stats[f"err_{polarity}"] += 1
                        f_log.write(json.dumps({
                            "domain":   domain_info["domain"],
                            "polarity": polarity,
                            "persona":  persona.get("name"),
                            "rep":      rep,
                            "ok":       False,
                            "error":    str(e),
                        }, ensure_ascii=False) + "\n")
                        f_log.flush()
                        continue

                    ok, reason, metrics = quality_filter(survey, answers, persona)
                    f_log.write(json.dumps({
                        "domain":     domain_info["domain"],
                        "polarity":   polarity,
                        "persona":    persona.get("name"),
                        "persona_id": persona.get("persona_id"),
                        "rep":        rep,
                        "ok":         ok,
                        "reason":     reason,
                        "metrics":    metrics,
                    }, ensure_ascii=False) + "\n")
                    f_log.flush()

                    if not ok:
                        print(f"  ⚠ {polarity} #{rep+1} ({persona.get('name')}) "
                              f"필터 탈락: {reason}")
                        stats[f"filtered_{polarity}"] += 1
                        continue

                    # 학습 레코드
                    train_rec = to_qlora_record(domain_info, persona, survey, answers)
                    record = {
                        "sample_id":   f"{ts}_{written:05d}",
                        "domain":      domain_info["domain"],
                        "survey_type": domain_info.get("survey_type", ""),
                        "scale_label": domain_info.get("scale_label", ""),
                        "category":    domain_info["category"],
                        "price_range": domain_info["price_range"],
                        "polarity":    polarity,
                        "persona_id":  persona.get("persona_id"),
                        "persona":     persona,
                        "rep":         rep,
                        "n_questions": len(survey["questions"]),
                        "metrics":     metrics,
                        "survey":      survey,
                        "responses":   answers,
                        "train":       train_rec,
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()
                    try:
                        os.fsync(f_out.fileno())
                    except (OSError, AttributeError):
                        pass  # 일부 환경에서 실패해도 buffering=1 + flush로 충분
                    written += 1
                    stats[f"ok_{polarity}"] += 1
                    print(f"  ✓ {polarity} #{rep+1} ({persona.get('name'):<8s}) "
                          f"mean={metrics.get('mean', '?')} "
                          f"std={metrics.get('std', '?')} "
                          f"consistency={metrics.get('consistency', '?')}")

    elapsed = time.time() - t0

    # 통계 저장
    final_stats = {
        "generated_at":      ts,
        "total_written":     written,
        "elapsed_sec":       round(elapsed, 1),
        "elapsed_per_sample": round(elapsed / max(written, 1), 1),
        "by_status":         dict(stats),
        "cost":              COST.summary(),
    }
    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump(final_stats, f, ensure_ascii=False, indent=2)

    print(f"\n{'─' * 70}")
    print(f"  완료! 학습 sample {written}개")
    print(f"  소요: {elapsed:.0f}초 / 평균 {elapsed/max(written,1):.1f}초/sample")
    print(f"\n  상세 통계:")
    for k, v in sorted(stats.items()):
        print(f"    {k:25s}: {v}")
    print(f"\n  저장:")
    print(f"    학습 데이터: {out_jsonl}")
    print(f"    생성 로그:   {out_log}")
    print(f"    통계:        {out_stats}")
    print(f"\n  비용: {COST.summary()}")


# ════════════════════════════════════════════════════════════════════════════
# Stage 5: 학습 데이터 분석
# ════════════════════════════════════════════════════════════════════════════

def cmd_analyze(args):
    """생성된 JSONL 파일 통계 분석"""
    if not os.path.exists(args.jsonl):
        print(f"[오류] 파일 없음: {args.jsonl}")
        return

    print(f"\n{'═' * 70}")
    print(f"  [분석] {args.jsonl}")
    print(f"{'═' * 70}\n")

    by_polarity   = Counter()
    by_domain     = Counter()
    by_category   = Counter()
    by_price      = Counter()
    by_persona    = Counter()
    consistencies = []
    means_by_pol  = defaultdict(list)
    fb_ratios     = []
    n_q_dist      = Counter()
    total = 0

    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            by_polarity[rec["polarity"]] += 1
            by_domain[rec["domain"]] += 1
            by_category[rec.get("category", "?")] += 1
            by_price[rec.get("price_range", "?")] += 1
            by_persona[rec.get("persona_id", "?")] += 1

            m = rec.get("metrics", {})
            if m.get("consistency") is not None:
                consistencies.append(m["consistency"])
            if m.get("mean") is not None:
                means_by_pol[rec["polarity"]].append(m["mean"])
            if m.get("fallback_ratio") is not None:
                fb_ratios.append(m["fallback_ratio"])

            n_q_dist[rec.get("n_questions", 0)] += 1

    avg_consist = sum(consistencies) / len(consistencies) if consistencies else 0
    avg_fb = sum(fb_ratios) / len(fb_ratios) if fb_ratios else 0

    analysis = {
        "file":              args.jsonl,
        "total_samples":     total,
        "by_polarity":       dict(by_polarity),
        "by_price_range":    dict(by_price),
        "n_unique_domains":  len(by_domain),
        "n_unique_personas": len(by_persona),
        "n_questions_distribution": dict(n_q_dist),
        "avg_consistency":   round(avg_consist, 3),
        "avg_fallback_ratio": round(avg_fb, 3),
        "polarity_means": {
            pol: round(sum(vs)/len(vs), 2) if vs else None
            for pol, vs in means_by_pol.items()
        },
        "top_categories":    dict(by_category.most_common(15)),
        "top_personas":      dict(by_persona.most_common(15)),
    }

    print(f"  총 샘플: {total}")
    print(f"\n  polarity 분포:")
    for pol in POLARITY_LIST:
        n = by_polarity.get(pol, 0)
        mean = analysis["polarity_means"].get(pol)
        print(f"    {pol:6s}: {n:>5d}  (척도 평균: {mean})")

    print(f"\n  가격대 분포:        {dict(by_price.most_common())}")
    print(f"  고유 도메인 수:     {len(by_domain)}")
    print(f"  고유 페르소나 수:   {len(by_persona)}")
    print(f"\n  품질 지표:")
    print(f"    평균 일관성:      {avg_consist:.3f}")
    print(f"    평균 fallback:    {avg_fb:.3f}")
    print(f"\n  문항 수 분포:")
    for n_q, cnt in sorted(n_q_dist.items()):
        print(f"    {n_q:3d}문항: {cnt}")

    if args.output is None:
        # 기본: <logs-root-dir>/analysis_TS.json
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = os.path.join(args.logs_root_dir, f"analysis_{ts}.json")
    else:
        output = args.output

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print(f"\n  저장: {output}")


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def add_dir_args(p):
    """모든 서브커맨드 공통: 디렉토리 경로 인자"""
    p.add_argument("--data-dir",        default=ACL_DATA,
                   help=f"데이터 폴더 (domains.json, personas.json, surveys/) "
                        f"[기본: {ACL_DATA}]")
    p.add_argument("--output-root-dir", default=ACL_OUTPUT,
                   help=f"출력 폴더 (학습 데이터 jsonl) [기본: {ACL_OUTPUT}]")
    p.add_argument("--logs-root-dir",   default=ACL_LOGS,
                   help=f"로그/통계 폴더 [기본: {ACL_LOGS}]")


def parse_args():
    p = argparse.ArgumentParser(
        description="Auto-Survey 데이터 생성 (4단계 파이프라인 + 분석)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # 1. gen-domains
    p1 = sub.add_parser("gen-domains", help="도메인 풀 생성")
    p1.add_argument("--output",      default=None,
                    help=f"저장 경로 (기본: <data-dir>/{DOMAINS_FILE})")
    p1.add_argument("--n",           type=int, default=100)
    p1.add_argument("--server",      default=LLAMA_SERVER)
    p1.add_argument("--model",       default=MODEL_NAME)
    p1.add_argument("--temperature", type=float, default=0.9)
    add_dir_args(p1)

    # 2. gen-personas
    p2 = sub.add_parser("gen-personas", help="페르소나 풀 생성")
    p2.add_argument("--output",          default=None,
                    help=f"저장 경로 (기본: <data-dir>/{PERSONAS_FILE})")
    p2.add_argument("--n-per-polarity",  type=int, default=10,
                    help="polarity 1종당 페르소나 수 (기본 10)")
    p2.add_argument("--server",          default=LLAMA_SERVER)
    p2.add_argument("--model",           default=MODEL_NAME)
    p2.add_argument("--temperature",     type=float, default=0.85)
    p2.add_argument("--seed",            type=int, default=42)
    add_dir_args(p2)

    # 3. gen-surveys
    p3 = sub.add_parser("gen-surveys", help="도메인 → 설문지")
    p3.add_argument("--domains",     default=None,
                    help=f"도메인 JSON (기본: <data-dir>/{DOMAINS_FILE})")
    p3.add_argument("--output-dir",  default=None,
                    help=f"설문지 저장 폴더 (기본: <data-dir>/{SURVEYS_SUBDIR})")
    p3.add_argument("--server",      default=LLAMA_SERVER)
    p3.add_argument("--model",       default=MODEL_NAME)
    p3.add_argument("--temperature", type=float, default=0.7)
    p3.add_argument("--retry",       type=int, default=3)
    p3.add_argument("--overwrite",   action="store_true")
    add_dir_args(p3)

    # 4. gen-responses
    p4 = sub.add_parser("gen-responses", help="설문지 + 페르소나 → 응답")
    p4.add_argument("--surveys-dir",     default=None,
                    help=f"설문지 폴더 (기본: <data-dir>/{SURVEYS_SUBDIR})")
    p4.add_argument("--personas",        default=None,
                    help=f"페르소나 JSON (기본: <data-dir>/{PERSONAS_FILE})")
    p4.add_argument("--output-dir",      default=None,
                    help="학습 데이터(jsonl) 저장 폴더 (기본: <output-root-dir>)")
    p4.add_argument("--logs-dir",        default=None,
                    help="로그/통계 저장 폴더 (기본: <logs-root-dir>)")
    p4.add_argument("--n-per-polarity",  type=int, default=3,
                    help="설문지 × polarity별 반복 횟수")
    p4.add_argument("--polarities",      default=None,
                    help="쉼표 구분 (예: '강부정,약부정'). 미지정 시 5종")
    p4.add_argument("--server",      default=LLAMA_SERVER)
    p4.add_argument("--model",       default=MODEL_NAME)
    p4.add_argument("--temperature", type=float, default=0.7)
    p4.add_argument("--seed",        type=int, default=42)
    add_dir_args(p4)

    # 5. analyze
    p5 = sub.add_parser("analyze", help="학습 데이터 통계 분석")
    p5.add_argument("--jsonl",  required=True,
                    help="분석할 train_data_*.jsonl 경로")
    p5.add_argument("--output", default=None,
                    help="분석 결과 JSON 저장 경로 (기본: <logs-root-dir>/analysis_TS.json)")
    add_dir_args(p5)

    return p.parse_args()


def main():
    args = parse_args()
    if hasattr(args, "seed"):
        random.seed(args.seed)

    try:
        if args.cmd == "gen-domains":
            cmd_gen_domains(args)
        elif args.cmd == "gen-personas":
            cmd_gen_personas(args)
        elif args.cmd == "gen-surveys":
            cmd_gen_surveys(args)
        elif args.cmd == "gen-responses":
            cmd_gen_responses(args)
        elif args.cmd == "analyze":
            cmd_analyze(args)
    except KeyboardInterrupt:
        print(f"\n\n  [중단] 사용자 취소. 비용: {COST.summary()}")
        sys.exit(130)
    except Exception:
        print(f"\n\n  [오류] 예외 발생:")
        traceback.print_exc()
        print(f"\n  비용: {COST.summary()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
