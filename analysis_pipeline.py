"""
인천 철도 연계 공백의 '탄소-격차' 분석 파이프라인
  1) 이중부담 유형 분류 (규칙 기반) + 민감도 분석
  2) AI 지역 유형화 (K-means 군집분석)
  3) XGBoost 계열 모델 + SHAP: 철도 접근성 격차를 만드는 요인 해석
  4) 정책 시나리오: 철도 연계 전환에 따른 탄소 감축량, 고령층 이동시간 단축

필요 파일 (같은 폴더)
  dong_access_S0_S1.csv, commute_mode_by_gu_2020.csv, car_registration_by_gu_202609.csv,
  incheon_admdong_population_202609.csv, incheon_admdong_pop_202609.geojson
설치: pip install pandas numpy geopandas scikit-learn shap matplotlib
실행: python analysis_pipeline.py
"""
import warnings

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------- 가정값 ----------------
GAP_SHARE = 70            # '연계 공백' 기준: S1 이후에도 동 내 지점의 70% 이상이 역에서 800m 밖
COMMUTER_RATE = 0.45      # 인구 대비 통근자 비율 (2020 인구총조사 통근 인구/인구 근사)
WORKDAYS = 250            # 연 통근일수
SPEED_KMH = 25            # 통근 평균 통행속도 (소요시간 -> 거리 환산)
EF_CAR, EF_RAIL, EF_BUS = 173, 18, 55   # gCO2/인·km (KRRI 2010 인용값)
EF_CAR_ALT = 163.1        # 민감도: 다른 선행연구 승용차 값
CONV_RATES = [0.05, 0.10, 0.20]         # 승용차 통근 -> 철도 전환율 시나리오
FEEDER_CAP_KM = 5         # 연계 DRT 이동거리 상한
WALK_M_PER_MIN, DRT_M_PER_MIN, DRT_WAIT_MIN = 75, 333, 8


def z(s):
    return (s - s.mean()) / s.std()


def map_gu(series_by_old_gu, weights):
    """2020년(개편 전) 구 값을 새 구에 배정: 서구->서해·검단, 중구->영종, 제물포=중구·동구 가중평균"""
    s = series_by_old_gu
    w = weights
    return {
        "연수구": s["연수구"], "남동구": s["남동구"], "부평구": s["부평구"],
        "계양구": s["계양구"], "미추홀구": s["미추홀구"],
        "서해구": s["서구"], "검단구": s["서구"], "영종구": s["중구"],
        "제물포구": (s["중구"] * w["중구"] + s["동구"] * w["동구"]) / (w["중구"] + w["동구"]),
    }


def load():
    d = pd.read_csv("dong_access_S0_S1.csv", dtype={"adm_cd2": str})
    pop = pd.read_csv("incheon_admdong_population_202609.csv", usecols=["adm_cd2", "청년비율_20_39"],
                      dtype={"adm_cd2": str})
    d = d.merge(pop, on="adm_cd2", how="left")
    cm = pd.read_csv("commute_mode_by_gu_2020.csv")
    cm = cm[cm["구분"] == "통근"].set_index("구")
    car = pd.read_csv("car_registration_by_gu_202609.csv").set_index("구")
    w = cm["인구"]
    d["승용차분담률"] = d["sggnm"].map(map_gu(cm["승용차분담률"], w))
    d["통근분"] = d["sggnm"].map(map_gu(cm["평균소요분"], w))
    d["자가용천명당"] = d["sggnm"].map(car["자가용승용_천명당"])
    g = gpd.read_file("incheon_admdong_pop_202609.geojson")[["adm_cd2", "geometry"]].to_crs(5179)
    g["adm_cd2"] = g["adm_cd2"].astype(str)
    g["면적km2"] = g.geometry.area / 1e6
    d = d.merge(g[["adm_cd2", "면적km2"]], on="adm_cd2", how="left")
    d["인구밀도"] = d["총인구"] / d["면적km2"]
    return d


def classify(d, gap_share=GAP_SHARE, zero="mean"):
    G = (z(d["고령비율_65"]) + z(d["후기고령비율_75"])) / 2
    C = (z(d["승용차분담률"]) + z(d["자가용천명당"])) / 2
    gcut, ccut = (0, 0) if zero == "mean" else (G.median(), C.median())
    gap = d["S1_800m초과비율"] >= gap_share
    t = np.select(
        [~gap, (G > gcut) & (C > ccut), G > gcut, C > ccut],
        ["비공백", "이중부담형", "격차형", "탄소형"], default="저부담형")
    return G, C, pd.Series(t, index=d.index)


def typology(d):
    d["격차지수G"], d["탄소지수C"], d["유형"] = classify(d)
    # 민감도: 공백 기준 60/70/80% x 기준선(평균/중앙값)
    variants = {}
    for gs in (60, 70, 80):
        for zr in ("mean", "median"):
            variants[f"{gs}_{zr}"] = classify(d, gs, zr)[2]
    V = pd.DataFrame(variants)
    base_gap = d["유형"] != "비공백"
    stable = (V.nunique(axis=1) == 1) & base_gap
    d["유형_안정"] = stable
    print("\n[1] 유형 분류 (공백 기준 70%, 기준선=평균)")
    print(d.groupby("유형").agg(동수=("adm_nm", "size"), 인구=("총인구", "sum")).round(0))
    print(f"민감도: 기준 6가지(공백 60/70/80% x 평균/중앙값)에서 유형이 한 번도 안 바뀐 동 "
          f"{int(stable.sum())}/{int(base_gap.sum())}개 (공백 동 중)")
    print("기준별 유형 분포:")
    print(V.apply(lambda s: s.value_counts()).fillna(0).astype(int).T.to_string())
    return d


def cluster(d):
    feats = ["S1_중앙m", "S1_800m초과비율", "고령비율_65", "후기고령비율_75", "승용차분담률", "자가용천명당", "인구밀도"]
    X = d[feats].copy()
    X["S1_중앙m"] = np.log1p(X["S1_중앙m"])
    X["인구밀도"] = np.log1p(X["인구밀도"])
    Xs = StandardScaler().fit_transform(X)
    scores = {}
    for k in range(3, 8):
        km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(Xs)
        scores[k] = silhouette_score(Xs, km.labels_)
    print("\n[2] 군집 수별 실루엣:", {k: round(v, 3) for k, v in scores.items()})
    k = max(range(4, 7), key=lambda k: scores[k])   # 해석 가능한 4~6개 중 최적
    km = KMeans(n_clusters=k, n_init=50, random_state=0).fit(Xs)
    d["군집"] = km.labels_
    prof = d.groupby("군집")[feats + ["총인구"]].agg({**{f: "mean" for f in feats}, "총인구": "sum"})
    prof.insert(0, "동수", d.groupby("군집").size())
    prof = prof.round(1)
    print(f"선택 k={k}")
    print(prof.to_string())
    for c in sorted(d["군집"].unique()):
        names = d[d["군집"] == c].sort_values("총인구", ascending=False)["adm_nm"].str.replace("인천광역시 ", "").head(6)
        print(f"  군집 {c} 대표 동: {', '.join(names)}")
    prof.to_csv("cluster_profile.csv", encoding="utf-8-sig")
    return d


def shap_step(d):
    import shap
    feats = {"log_인구": np.log(d["총인구"]), "log_인구밀도": np.log(d["인구밀도"]),
             "고령비율_65": d["고령비율_65"], "후기고령비율_75": d["후기고령비율_75"],
             "청년비율_20_39": d["청년비율_20_39"], "승용차분담률(구)": d["승용차분담률"], "자가용천명당(구)": d["자가용천명당"]}
    X = pd.DataFrame(feats)
    y = np.log1p(d["S1_중앙m"])
    model = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)
    pred = cross_val_predict(model, X, y, cv=KFold(5, shuffle=True, random_state=0))
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"\n[3] 접근성 격차(log S1 중앙 보행거리) 예측 5-fold CV R² = {r2:.2f}  (n={len(d)})")
    model.fit(X, y)
    sv = shap.TreeExplainer(model).shap_values(X)
    imp = pd.DataFrame({"변수": X.columns, "평균|SHAP|": np.abs(sv).mean(0)}).sort_values("평균|SHAP|", ascending=False)
    imp["상관방향(Spearman)"] = [round(pd.Series(sv[:, i]).corr(X.iloc[:, i], method="spearman"), 2) for i in imp.index]
    print(imp.round(3).to_string(index=False))
    imp.to_csv("shap_importance.csv", index=False, encoding="utf-8-sig")
    plt.figure()
    shap.summary_plot(sv, X, show=False)
    plt.title(f"SHAP: 철도 접근성 격차 요인 (CV R²={r2:.2f})")
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=150)
    plt.close()
    return r2


def scenarios(d):
    tgt = d[d["유형"].isin(["탄소형", "이중부담형"])].copy()
    tgt["승용차통근자"] = tgt["총인구"] * COMMUTER_RATE * tgt["승용차분담률"] / 100
    tgt["편도km"] = tgt["통근분"] / 60 * SPEED_KMH
    tgt["연계km"] = (tgt["S1_중앙m"] / 1000).clip(upper=FEEDER_CAP_KM)
    base_t = (tgt["승용차통근자"] * 2 * WORKDAYS * tgt["편도km"] * EF_CAR / 1e6).sum()

    def net_kg(ef_car, ef_feeder):
        per = 2 * WORKDAYS * (tgt["편도km"] * (ef_car - EF_RAIL) - tgt["연계km"] * ef_feeder) / 1000
        return per.clip(lower=0)

    rows = []
    for name, (ec, ef) in {"기본(승용차173, 연계 55)": (EF_CAR, EF_BUS),
                           "보수적(승용차163, 연계=승용차수준)": (EF_CAR_ALT, EF_CAR_ALT)}.items():
        net = net_kg(ec, ef)
        for r in CONV_RATES:
            rows.append({"가정": name, "전환율": f"{int(r*100)}%",
                         "전환통근자_명": int((tgt["승용차통근자"] * r).sum()),
                         "연간감축_tCO2": round(float((tgt["승용차통근자"] * r * net).sum() / 1000), 0)})
    sc = pd.DataFrame(rows)
    sc.to_csv("carbon_scenarios.csv", index=False, encoding="utf-8-sig")
    print("\n[4-1] 탄소 감축 시나리오 (탄소형+이중부담형 동 %d개)" % len(tgt))
    print(f"해당 동 승용차 통근 배출(추정): {base_t:,.0f} tCO2/년, 승용차 통근자 {int(tgt['승용차통근자'].sum()):,}명")
    print(sc.to_string(index=False))

    gap = d[d["유형"].isin(["격차형", "이중부담형"])].copy()
    gap["고령인구"] = gap["총인구"] * gap["고령비율_65"] / 100
    walk = gap["S1_중앙m"] / WALK_M_PER_MIN
    drt = gap["S1_중앙m"] / DRT_M_PER_MIN + DRT_WAIT_MIN
    gap["단축분"] = (walk - drt).clip(lower=0)
    w = gap["고령인구"]
    print("\n[4-2] 이동 격차 완화 (격차형+이중부담형 동 %d개)" % len(gap))
    print(f"65세 이상 {int(w.sum()):,}명, 역까지 도보 중앙 {walk.median():.0f}분 -> 연계 DRT 가정 시 "
          f"고령인구 가중 평균 {(gap['단축분'] * w).sum() / w.sum():.1f}분 단축")
    d.loc[tgt.index, "승용차통근자"] = tgt["승용차통근자"]
    d.loc[tgt.index, "감축10pct_tCO2"] = tgt["승용차통근자"] * 0.10 * net_kg(EF_CAR, EF_BUS) / 1000
    d.loc[gap.index, "고령인구"] = gap["고령인구"]
    d.loc[gap.index, "DRT단축분"] = gap["단축분"]
    d.loc[gap.index, "고령절감_인분"] = gap["고령인구"] * gap["단축분"]
    return d


SMALL_POP = 10000   # 이보다 인구가 적은 공백 동은 '소규모 동'으로 따로 표시


def priority(d):
    """거리 순위가 아니라 '영향받는 사람 수' 기준 우선순위.
    탄소: 10% 전환 시 연간 감축량(tCO2) / 격차: 65세 이상 인구 x DRT 단축분(인·분, 1회 이동 기준)"""
    gap = d["유형"] != "비공백"
    d["소규모동"] = gap & (d["총인구"] < SMALL_POP)
    cols = ["adm_nm", "유형", "총인구", "S1_중앙m"]
    # 소규모 동은 공항·항만 등 비주거지가 넓어 보행거리가 과대 계산되므로 순위에서 빼고 별도 표시
    main = ~d["소규모동"]
    c = d[main].dropna(subset=["감축10pct_tCO2"]).sort_values("감축10pct_tCO2", ascending=False)
    g = d[main].dropna(subset=["고령절감_인분"]).sort_values("고령절감_인분", ascending=False)
    print("\n[5] 영향 인구 기준 우선순위")
    print("탄소 우선(10% 전환 시 연간 감축 tCO2):")
    print(c[cols + ["감축10pct_tCO2"]].head(8).round(0).to_string(index=False))
    print("격차 우선(65세 이상 x 단축분, 인·분):")
    print(g[cols + ["고령인구", "DRT단축분", "고령절감_인분"]].head(8).round(1).to_string(index=False))
    d.loc[c.index, "탄소순위"] = range(1, len(c) + 1)
    d.loc[g.index, "격차순위"] = range(1, len(g) + 1)
    s = d[d["소규모동"]]
    print(s[["adm_nm", "유형", "총인구", "S1_중앙m"]].sort_values("총인구").to_string(index=False))
    print(f"소규모 공백 동(인구 {SMALL_POP:,}명 미만): {len(s)}개, 인구 {int(s['총인구'].sum()):,}명 -> 복지형 소규모 DRT 별도 검토")
    out = pd.concat([c.assign(구분="탄소").head(10), g.assign(구분="격차").head(10), s.assign(구분="소규모")])
    out[["구분"] + cols + ["감축10pct_tCO2", "고령인구", "DRT단축분", "고령절감_인분", "소규모동"]].to_csv(
        "priority_top10.csv", index=False, encoding="utf-8-sig")
    return d


def main():
    d = load()
    d = typology(d)
    d = cluster(d)
    shap_step(d)
    d = scenarios(d)
    d = priority(d)
    d["품질주의"] = (d["S1_도달불가비율"] > 10) | (d["총인구"] < 5000)
    d.to_csv("dong_typology_v1.csv", index=False, encoding="utf-8-sig")
    print("\n저장: dong_typology_v1.csv, cluster_profile.csv, shap_importance.csv, shap_summary.png, carbon_scenarios.csv")


if __name__ == "__main__":
    main()
