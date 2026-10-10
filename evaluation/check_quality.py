import pandas as pd, json, glob, re, os
qs = json.load(open('eval/converted/eval_questions.json'))
EXPECT_ROWS = sum(len(q['options']) for q in qs)   # 94문항 옵션 총합
print(f"기준: 문항 94 · 총 행 {EXPECT_ROWS}")
print(f"{'파일':42s}{'행':>6s}{'문항':>5s}{'응답중앙값':>9s}  판정")
bad = 0
for f in sorted(glob.glob('eval/dists/*.csv') + glob.glob('eval/panel_h100/*.csv')):
    d = pd.read_csv(f)
    nrow = len(d); nq = d.q_id.nunique()
    med = int(d[d.option_num==1].n_responses.median()) if 'n_responses' in d.columns else -1
    ok = (nrow == EXPECT_ROWS and nq == 94)
    if med >= 0 and med < 300: ok = False       # 패널: 유효 응답 60% 미만이면 경고
    mark = "OK" if ok else "⚠ 확인 필요"
    if not ok: bad += 1
    print(f"{os.path.basename(f):42s}{nrow:>6d}{nq:>5d}{med:>9d}  {mark}")
print(f"\n→ 전체 {len(glob.glob('eval/dists/*.csv'))+len(glob.glob('eval/panel_h100/*.csv'))}개 중 경고 {bad}개")
