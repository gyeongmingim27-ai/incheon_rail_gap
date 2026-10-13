"""
버스 연계를 반영한 '실질 연계 공백' 판정

질문: 역에서 걸어서 멀어도(보행 공백), 역으로 가는 버스가 자주 다니면 실제로는 공백이 아니다.
      어떤 공백 동이 '버스로 메워지는 공백'이고, 어떤 동이 '버스도 뜸한 진짜 공백'인가?

필요 파일 (같은 폴더)
  bus_routes.csv, bus_route_stops.csv          : fetch_incheon_bus.py 결과
  stations_S1_incheon_buffer5km.csv             : S0+S1 역 목록
  incheon_admdong_pop_202609.geojson            : 행정동 경계
  dong_typology_v1.csv                          : analysis_pipeline.py 결과 (유형, S1_중앙m 등)
설치: pip install pandas geopandas
실행: python bus_link.py

방법 (동 단위)
  1) 역 연계 노선: S1 역에서 300m 이내에 정류장이 하나라도 있는 노선
  2) 동의 연계 노선: 동 경계(+100m) 안 정류장을 지나는 역 연계 노선
  3) 연계 빈도(대/시) = 노선별 60/평균배차 의 합 → 합성 배차간격 = 60/빈도
  4) 버스 이용 시 역까지 시간 = 정류장 접근 5분 + 대기(합성 배차/2, 최대 30분)
                               + 탑승(보행거리 중앙값 x 1.3 / 15km/h)
     도보 시간 = 보행거리 중앙값 / 75m/분
  5) 실질 시간 = min(도보, 버스)
     '버스 보완'  : 보행 공백이지만 실질 시간 <= 15분 이고 합성 배차 <= 15분
     '실질 공백'  : 그 외 (버스가 없거나 뜸함)
     '야간 공백'  : 연계 노선 막차가 23시 이전(또는 연계 노선 없음)
산출물: dong_bus_link.csv, 콘솔 요약
"""
import geopandas as gpd
import numpy as np
import pandas as pd

CRS_M = 5179
STATION_R = 300        # 노선이 역을 지나는지 판정 반경(m)
DONG_BUF = 100         # 동 경계 여유(m)
STOP_ACCESS_MIN = 5    # 정류장까지 걷는 시간(분, 가정)
BUS_SPEED_M_MIN = 250  # 15km/h
DETOUR = 1.3           # 버스 경로 우회계수
WALK_M_MIN = 75
MAX_WAIT = 30
OK_TIME = 15           # 실질 시간 기준(분)
OK_HEADWAY = 15        # 합성 배차 기준(분)
NIGHT_LAST = 2300      # 막차가 이보다 이르면 야간 공백
MAX_RIDE_M = 8000      # 동 정류장에서 탄 뒤 노선 경로상 이 거리 안에 역 인접 정류장이 있어야 '연계'


def hhmm(x):
    """'0530' / '530' -> 530 (분 단위 아님, hhmm 정수). 해석 불가는 NaN"""
    try:
        return int(float(str(x).strip()))
    except ValueError:
        return np.nan


def load():
    routes = pd.read_csv("bus_routes.csv", dtype={"ROUTEID": str})
    stops = pd.read_csv("bus_route_stops.csv", dtype={"ROUTEID": str, "BSTOPID": str})
    st = pd.read_csv("stations_S1_incheon_buffer5km.csv")
    st = st[st["포함시나리오"].str.contains("S1")]
    st = gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st["경도"], st["위도"]), crs=4326).to_crs(CRS_M)
    stops = gpd.GeoDataFrame(stops, geometry=gpd.points_from_xy(stops["lon"], stops["lat"]), crs=4326).to_crs(CRS_M)
    dong = gpd.read_file("incheon_admdong_pop_202609.geojson")
    dong["adm_cd2"] = dong["adm_cd2"].astype(str)
    dong = dong[dong["분석대상"].astype(str).str.lower().isin(["true", "1"])].to_crs(CRS_M)
    typ = pd.read_csv("dong_typology_v1.csv", dtype={"adm_cd2": str})
    return routes, stops, st, dong, typ


def route_table(routes):
    r = routes.copy()
    g = r[["MIN_ALLOCGAP", "MAX_ALLOCGAP"]].apply(pd.to_numeric, errors="coerce")
    r["배차"] = g.mean(axis=1)
    # 배차간격 0/0 은 '정보 없음(비정기 운행)'이므로 결측 처리
    r.loc[(g["MIN_ALLOCGAP"] <= 0) & (g["MAX_ALLOCGAP"] <= 0), "배차"] = np.nan
    r.loc[r["배차"] <= 0, "배차"] = np.nan
    f = r["FBUS_DEPHMS"].map(hhmm)
    l = r["LBUS_DEPHMS"].map(hhmm)
    l = l.where(~(l < f), l + 2400)          # 막차가 첫차보다 이르면 자정 이후 운행(+24h)
    l = l.where(f != l)                       # 첫차=막차는 1회 운행/정보 없음 -> 결측
    r["첫차"], r["막차"] = f, l
    print(f"배차 정보 없는 노선 {r['배차'].isna().sum()}개, 막차 정보 없는 노선 {r['막차'].isna().sum()}개 -> 해당 값은 판정에서 제외(보수적)")
    return r.set_index("ROUTEID")


def main():
    routes, stops, st, dong, typ = load()
    r = route_table(routes)

    # 1) 정류장별: 역 300m 이내 여부, 소속 동
    stops = stops.copy()
    stops["seq"] = pd.to_numeric(stops["BSTOPSEQ"], errors="coerce")
    stops = stops.sort_values(["ROUTEID", "seq"]).reset_index(drop=True)
    near = gpd.sjoin(stops[["geometry"]], gpd.GeoDataFrame(geometry=st.buffer(STATION_R), crs=CRS_M), predicate="within")
    stops["역인접"] = stops.index.isin(near.index)
    db = dong[["adm_cd2", "geometry"]].copy()
    db["geometry"] = db.buffer(DONG_BUF)
    sd = gpd.sjoin(stops[["geometry"]], db, predicate="within")[["adm_cd2"]]
    print(f"노선 {len(r)}개 중 역 인접 정류장을 지나는 노선 {stops.loc[stops['역인접'], 'ROUTEID'].nunique()}개")

    # 2) 노선 경로를 따라 누적거리 계산 -> 동 안 각 정류장에서 역 인접 정류장까지의 승차거리
    x, y = stops.geometry.x.to_numpy(), stops.geometry.y.to_numpy()
    recs = []
    for rid, grp in stops.groupby("ROUTEID"):
        ix = grp.index.to_numpy()
        step = np.r_[0.0, np.hypot(np.diff(x[ix]), np.diff(y[ix]))]
        cum = pd.Series(np.cumsum(step), index=ix)
        st_idx = ix[stops.loc[ix, "역인접"].to_numpy()]
        if len(st_idx) == 0:
            continue
        cs = cum.loc[st_idx].to_numpy()
        for i in sd.index.intersection(ix):
            ride = float(np.abs(cum.loc[i] - cs).min())
            if ride <= MAX_RIDE_M:
                for cd in np.atleast_1d(sd.loc[i, "adm_cd2"]):
                    recs.append((cd, stops.loc[i, "BSTOPID"], rid, ride))
    rides = pd.DataFrame(recs, columns=["adm_cd2", "BSTOPID", "ROUTEID", "승차거리m"])
    print(f"정류장-노선 연계 {len(rides)}건 (승차거리 {MAX_RIDE_M/1000:.0f}km 이내)")

    # 3) 정류장 단위 빈도 -> 동 단위는 '동 안 정류장의 중앙값' (한 정류장에 서는 노선만 이용 가능)
    all_stops_in_dong = pd.DataFrame({"adm_cd2": sd["adm_cd2"].to_numpy(),
                                      "BSTOPID": stops.loc[sd.index, "BSTOPID"].to_numpy()}).drop_duplicates()
    rows = []
    for cd in dong["adm_cd2"]:
        sids = all_stops_in_dong.loc[all_stops_in_dong["adm_cd2"] == cd, "BSTOPID"]
        per = []
        for sid in sids:
            rr = rides[(rides["adm_cd2"] == cd) & (rides["BSTOPID"] == sid)]
            rids = [q for q in rr["ROUTEID"] if q in r.index]
            sub = r.loc[rids] if rids else r.iloc[0:0]
            hw = sub["배차"].dropna()
            per.append({"n": len(rids),
                        "freq": float((60 / hw.clip(lower=1)).sum()) if len(hw) else 0.0,
                        "last": sub["막차"].max() if len(sub) else np.nan,
                        "first": sub["첫차"].min() if len(sub) else np.nan,
                        "ride": float(rr["승차거리m"].median()) if len(rr) else np.nan})
        p = pd.DataFrame(per)
        if p.empty:
            rows.append({"adm_cd2": cd, "연계노선수": 0, "연계빈도_대시": 0.0, "합성배차_분": np.nan,
                         "첫차": np.nan, "막차": np.nan, "승차거리_중앙m": np.nan, "정류장수": 0, "연계정류장비율": 0.0})
            continue
        freq = float(p["freq"].median())
        rows.append({"adm_cd2": cd, "연계노선수": int(rides.loc[rides["adm_cd2"] == cd, "ROUTEID"].nunique()),
                     "연계빈도_대시": round(freq, 1),
                     "합성배차_분": round(60 / freq, 1) if freq > 0 else np.nan,
                     "첫차": p["first"].median(), "막차": p["last"].median(),
                     "승차거리_중앙m": float(p["ride"].median()) if p["ride"].notna().any() else np.nan,
                     "정류장수": len(p), "연계정류장비율": round((p["n"] > 0).mean() * 100, 0)})
    b = pd.DataFrame(rows)

    d = typ.merge(b, on="adm_cd2", how="left")
    walk = d["S1_중앙m"] / WALK_M_MIN
    wait = (d["합성배차_분"] / 2).clip(upper=MAX_WAIT)
    bus = STOP_ACCESS_MIN + wait + d["승차거리_중앙m"] / BUS_SPEED_M_MIN
    d["도보분"] = walk.round(1)
    d["버스분"] = bus.round(1)
    d["실질분"] = np.fmin(walk, bus).round(1)

    gap = d["유형"] != "비공백"
    ok = (d["실질분"] <= OK_TIME) & (d["합성배차_분"] <= OK_HEADWAY)
    d["버스판정"] = np.select([~gap, gap & ok], ["비공백", "버스 보완"], default="실질 공백")
    d["야간공백"] = d["막차"].isna() | (d["막차"] < NIGHT_LAST)
    d.to_csv("dong_bus_link.csv", index=False, encoding="utf-8-sig")

    g = d[gap]
    print(f"\n[공백 동 {int(gap.sum())}개의 버스 연계 판정]")
    print(g.groupby("버스판정").agg(동수=("adm_nm", "size"), 인구=("총인구", "sum")).to_string())
    print("\n[유형 x 버스판정]")
    print(pd.crosstab(g["유형"], g["버스판정"], margins=True).to_string())
    print(f"\n야간 공백(연계 막차 23시 이전 또는 연계 노선 없음): 공백 동 {int(g['야간공백'].sum())}개, "
          f"인구 {int(g.loc[g['야간공백'], '총인구'].sum()):,}명")
    cols = ["adm_nm", "유형", "총인구", "S1_중앙m", "연계노선수", "연계정류장비율", "합성배차_분", "승차거리_중앙m", "도보분", "버스분", "실질분"]
    print("\n[실질 공백 중 실질 시간이 긴 동]")
    print(g[g["버스판정"] == "실질 공백"].sort_values("실질분", ascending=False)[cols].head(12).to_string(index=False))
    print("\n저장: dong_bus_link.csv")


if __name__ == "__main__":
    main()
