"""
버스 판정(dong_bus_link.csv)과 유형(dong_typology_v1.csv)을 합쳐 연계 공백 동 표(gap_final.csv)를 만들고,
연계 DRT 도입 효과를 '현재(보행·버스 중 빠른 쪽)' 대비로 계산한다.

필요 파일: dong_typology_v1.csv, dong_bus_link.csv (analysis_pipeline.py, bus_link.py 결과)
실행: python make_gap_final.py   → gap_final.csv, bus_sensitivity.csv

가정: DRT 평균 속도 20km/h(333 m/분) + 호출 대기 8분
"""
import numpy as np
import pandas as pd

DRT_WAIT_MIN = 8
DRT_M_MIN = 333

t = pd.read_csv("dong_typology_v1.csv", dtype={"adm_cd2": str})
b = pd.read_csv("dong_bus_link.csv", dtype={"adm_cd2": str})
cols = ["adm_cd2", "연계노선수", "연계빈도_대시", "합성배차_분", "첫차", "막차", "승차거리_중앙m", "정류장수",
        "연계정류장비율", "도보분", "버스분", "실질분", "버스판정", "야간공백"]
g = t[t["유형"] != "비공백"].merge(b[cols], on="adm_cd2", how="left")
g["short"] = g["adm_nm"].str.split().str[-1]
g["gu"] = g["sggnm"]
g["DRT분"] = DRT_WAIT_MIN + g["S1_중앙m"] / DRT_M_MIN
g["절감_현재대비"] = g["실질분"] - g["DRT분"]
g["고령절감_현재"] = g["고령인구"] * g["절감_현재대비"].clip(lower=0)
g.to_csv("gap_final.csv", index=False, encoding="utf-8-sig")

# 버스 판정 기준 민감도: 실질 시간 T분 이하 & 합성 배차 H분 이하이면 '버스 보완'
rows = []
for T in (12, 15, 20):
    for H in (10, 15, 20):
        ok = (g["실질분"] <= T) & (g["합성배차_분"] <= H)
        rows.append({"시간기준_분": T, "배차기준_분": H, "실질공백_동수": int((~ok).sum()),
                     "실질공백_인구": int(g.loc[~ok, "총인구"].sum())})
s = pd.DataFrame(rows)
s.to_csv("bus_sensitivity.csv", index=False, encoding="utf-8-sig")
print(s.to_string(index=False))
always = None
for T in (12, 15, 20):
    for H in (10, 15, 20):
        r = set(g.loc[~((g["실질분"] <= T) & (g["합성배차_분"] <= H)), "adm_nm"])
        always = r if always is None else always & r
print("9가지 기준 모두 실질 공백:", sorted(x.split()[-1] for x in always))
print("  이 중 인구 1만 명 이상:", sorted(x.split()[-1] for x in always if g.loc[g.adm_nm == x, "총인구"].iloc[0] >= 10000))
