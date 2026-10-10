#!/bin/bash
# ============================================================
# merge_verify.sh — 3대 부정 데이터 병합 + 검증 (machine1에서 실행)
#   [1] machine2/machine3 jsonl 수집 (scp)
#   [2] 중복 제거 병합 → neg_train_final.jsonl / other_polarity.jsonl
#   [3] 품질 검증 리포트 (극성 평균·std·consistency·fallback·분야 분포)
#   [4] 15K 균형 서브셋 (강부정 7,500 + 약부정 7,500) 생성
# ============================================================
set -e
BASE="$HOME/ACL_EN"
mkdir -p "$BASE/merged/g2" "$BASE/merged/g3"

echo "[1/2] machine2 / machine3 데이터 수집"
scp 'USER@HOST2:~/ACL_EN/output/train_data_*.jsonl' "$BASE/merged/g2/" \
    || echo "  (machine2 수집 실패 — 이미 수집돼 있으면 무시)"
scp 'USER@HOST3:~/ACL_EN/output/train_data_*.jsonl' "$BASE/merged/g3/" \
    || echo "  (machine3 수집 실패 — 이미 수집돼 있으면 무시)"

echo ""
echo "[2/2] 병합 + 검증"
python3 - << 'PYEOF'
import glob, json, os, random, statistics
from collections import Counter

BASE = os.path.expanduser("~/ACL_EN")
SOURCES = [
    ("machine1", os.path.join(BASE, "output",  "train_data_*.jsonl")),
    ("machine2", os.path.join(BASE, "merged", "g2", "*.jsonl")),
    ("machine3", os.path.join(BASE, "merged", "g3", "*.jsonl")),
]
NEG = ("strongly_negative", "weakly_negative")
EXPECTED = {"strongly_negative": 1.4, "weakly_negative": 2.3}

seen, dup, broken = set(), 0, 0
pol_cnt = Counter()                      # 최종(중복 제거 후) 극성 분포
machine_neg = Counter()                  # 머신별 부정 기여
stype_cnt = Counter()                    # 부정 샘플의 설문 분야 분포
domains = set()
mets = {p: {"mean": [], "std": [], "cons": [], "fb": []} for p in NEG}
neg_lines = {p: [] for p in NEG}

oth_path = os.path.join(BASE, "other_polarity.jsonl")
neg_path = os.path.join(BASE, "neg_train_final.jsonl")
oth = open(oth_path, "w", encoding="utf-8")

for machine, pat in SOURCES:
    for f in sorted(glob.glob(pat)):
        for line in open(f, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                s = json.loads(line)
            except Exception:
                broken += 1
                continue
            key = (s.get("sample_id"), s.get("domain"),
                   s.get("persona_id"), s.get("rep"))
            if key in seen:
                dup += 1
                continue
            seen.add(key)
            pol = s.get("polarity", "?")
            pol_cnt[pol] += 1
            if pol in NEG:
                machine_neg[machine] += 1
                stype_cnt[s.get("survey_type") or s.get("category") or "?"] += 1
                domains.add(s.get("domain"))
                m = s.get("metrics", {})
                mets[pol]["mean"].append(m.get("mean", 0))
                mets[pol]["std"].append(m.get("std", 0))
                mets[pol]["cons"].append(m.get("consistency", 0))
                mets[pol]["fb"].append(m.get("fallback_ratio", 0))
                neg_lines[pol].append(line)
            else:
                oth.write(line + "\n")
oth.close()

with open(neg_path, "w", encoding="utf-8") as out:
    for p in NEG:
        for line in neg_lines[p]:
            out.write(line + "\n")

neg_total = sum(len(neg_lines[p]) for p in NEG)

print("=" * 62)
print("  최종 병합 리포트")
print("=" * 62)
print(f"  깨진 라인 제외: {broken}   중복 제거: {dup}")
print()
print("  ── 머신별 부정 기여 ──")
for mch in ("machine1", "machine2", "machine3"):
    print(f"    {mch}: {machine_neg[mch]:,}")
print()
print("  ── 극성 분포 (중복 제거 후) ──")
for k, v in pol_cnt.most_common():
    tag = "  <- 학습용" if k in NEG else ""
    print(f"    {k:20s}{v:>7,}{tag}")
print(f"    {'NEG TOTAL':20s}{neg_total:>7,}")
print()
print("  ── 부정 샘플 품질 검증 ──")
ok_all = True
for p in NEG:
    mm = statistics.mean(mets[p]["mean"])
    ms = statistics.mean(mets[p]["std"])
    mc = statistics.mean(mets[p]["cons"])
    mf = statistics.mean(mets[p]["fb"])
    flag = "OK" if abs(mm - EXPECTED[p]) < 0.5 else "WARN"
    ok_all &= (flag == "OK")
    print(f"    {p}")
    print(f"      mean={mm:.3f} (기대 {EXPECTED[p]}) [{flag}]  "
          f"std={ms:.2f}  consistency={mc:.3f}  fallback={mf:.4f}")
print()
print("  ── 커버리지 ──")
print(f"    고유 도메인: {len(domains)}개")
for k, v in stype_cnt.most_common():
    print(f"    {k:20s}{v:>7,} ({100*v/max(neg_total,1):.1f}%)")
print()

# 15K 균형 서브셋
S, W = neg_lines["strongly_negative"], neg_lines["weakly_negative"]
if len(S) >= 7500 and len(W) >= 7500:
    random.seed(42)
    sub = random.sample(S, 7500) + random.sample(W, 7500)
    random.shuffle(sub)
    p15 = os.path.join(BASE, "neg_train_15k.jsonl")
    with open(p15, "w", encoding="utf-8") as out:
        out.write("\n".join(sub) + "\n")
    print(f"  15K 균형 서브셋 저장: {p15} (7,500 + 7,500, seed=42)")
else:
    print(f"  [알림] 15K 서브셋 생략 — 강부정 {len(S):,} / 약부정 {len(W):,} (각 7,500 필요)")
print()
print(f"  전체 부정 데이터:   {neg_path}")
print(f"  중립/긍정 (보관용): {oth_path}")
print("=" * 62)
print("  결론:", "데이터 구축 완료 — 극성 평균 정상 범위" if ok_all
      else "WARN 항목 확인 필요 (기대 평균 ±0.5 초과)")
print("=" * 62)
PYEOF
