"""
인천 행정동별 철도역 보행 접근성 계산 (S0 현재 / S1 확정노선 개통 후)

필요 파일 (이 스크립트와 같은 폴더)
  - incheon_admdong_pop_202609.geojson     : 행정동 경계 + 인구 (분석대상 열 포함)
  - stations_S1_incheon_buffer5km.csv       : S0+S1 역 목록 (포함시나리오 열 포함)

설치
  pip install osmnx geopandas scipy matplotlib

실행
  python access_calc.py
  (처음 실행 시 OSM 보행망을 내려받느라 5~15분 걸릴 수 있음. 이후에는 cache/ 폴더에서 재사용)

방법
  1) 분석 대상 138개 행정동 + 2km 버퍼 범위의 OSM 보행망을 내려받음
  2) 시나리오별 역을 가장 가까운 보행망 노드에 연결
  3) 모든 역을 출발점으로 한 번에 최단거리 계산(multi-source Dijkstra)
     -> 모든 노드의 '가장 가까운 역까지 보행거리'
  4) 노드를 행정동에 배정하고 동별 평균·중앙값·800m/1.5km 초과 비율 계산
     (동 안의 인구 분포는 보행망 노드 분포로 근사)

산출물
  - dong_access_S0_S1.csv   : 행정동별 접근성 지표, S1-S0 변화, 인구·고령비율
  - access_nodes.gpkg       : 노드별 거리 (QGIS 등에서 지도화용)
  - access_map_S0_S1.png    : 동별 평균 보행거리 지도
"""
import os
import sys

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

DONG_FILE = "incheon_admdong_pop_202609.geojson"
STATION_FILE = "stations_S1_incheon_buffer5km.csv"
CRS_M = "EPSG:5179"          # 미터 단위 좌표계
BUFFER_M = 2000              # 경계 밖 보행망 여유 (경계 밖 역 이용 반영)
SNAP_MAX_M = 300             # 역-노드 연결 허용 거리 (이보다 멀면 범위 밖 역으로 보고 제외)
CUTOFF_M = 15000             # 최단거리 계산 상한
THRESHOLDS = [800, 1500]     # 도보 약 10분 / 약 20분


def load_inputs():
    for f in (DONG_FILE, STATION_FILE):
        if not os.path.exists(f):
            sys.exit(f"입력 파일이 없습니다: {f}")
    dong = gpd.read_file(DONG_FILE)
    dong = dong[dong["분석대상"].astype(str).str.lower().isin(["true", "1"])].to_crs(CRS_M)
    st = pd.read_csv(STATION_FILE)
    st = gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st["경도"], st["위도"]), crs=4326).to_crs(CRS_M)
    print(f"행정동 {len(dong)}개, 역 {len(st)}개 (S1 신설 {int((st['시나리오'] == 'S1').sum())}개)")
    return dong, st


def download_graph(dong):
    import osmnx as ox
    ox.settings.use_cache = True
    ox.settings.log_console = False
    area = gpd.GeoSeries([dong.union_all().buffer(BUFFER_M)], crs=CRS_M).to_crs(4326).iloc[0]
    print("OSM 보행망 내려받는 중... (처음엔 오래 걸릴 수 있음)")
    # retain_all=True: 영종도처럼 보행로로 육지와 이어지지 않은 지역도 버리지 않음
    G = ox.graph_from_polygon(area, network_type="walk", simplify=True, retain_all=True)
    G = ox.project_graph(G, to_crs=CRS_M)
    print(f"노드 {G.number_of_nodes():,}개, 링크 {G.number_of_edges():,}개")
    return G


def node_table(G):
    ids = np.fromiter(G.nodes, dtype=np.int64, count=G.number_of_nodes())
    xy = np.array([(G.nodes[n]["x"], G.nodes[n]["y"]) for n in ids])
    return ids, xy


def nearest_station_distance(G, ids, xy, stations):
    """시나리오 역 집합 -> 모든 노드의 가장 가까운 역까지 보행거리(m)"""
    tree = cKDTree(xy)
    sxy = np.c_[stations.geometry.x, stations.geometry.y]
    snap_d, idx = tree.query(sxy)
    ok = snap_d <= SNAP_MAX_M
    sources = set(ids[idx[ok]].tolist())
    Gu = G.to_undirected() if G.is_directed() else G
    dist = nx.multi_source_dijkstra_path_length(Gu, sources, cutoff=CUTOFF_M, weight="length")
    # 역에서 노드까지 붙인 거리(snap)는 무시: 역 출입구 위치 오차 수준
    d = np.array([dist.get(n, np.nan) for n in ids])
    return d, int(ok.sum())


def summarize(nodes, dong, col):
    j = gpd.sjoin(nodes[[col, "geometry"]], dong[["adm_cd2", "geometry"]], predicate="within", how="inner")
    g = j.groupby("adm_cd2")[col]
    out = pd.DataFrame({
        f"{col}_평균m": g.mean().round(0),
        f"{col}_중앙m": g.median().round(0),
        f"{col}_노드수": g.size(),
        f"{col}_도달불가비율": g.apply(lambda s: s.isna().mean() * 100).round(1),
    })
    # 초과 비율은 '역까지 도달 가능한 노드'만 대상으로 계산 (OSM 보행망이 끊긴 조각 때문에
    # 도달불가가 된 노드가 비율을 부풀리지 않도록). 도달불가비율은 데이터 품질 점검용으로 따로 둠.
    for t in THRESHOLDS:
        out[f"{col}_{t}m초과비율"] = g.apply(
            lambda s, t=t: (s.dropna() > t).mean() * 100 if s.notna().any() else np.nan).round(1)
    return out


def make_map(dong, res):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    m = dong.merge(res, left_on="adm_cd2", right_index=True)
    vmax = np.nanpercentile(m[["S0_평균m", "S1_평균m"]].values, 95)
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for ax, col, title in [(axes[0], "S0_평균m", "S0 현재: 역까지 평균 보행거리(m)"),
                           (axes[1], "S1_평균m", "S1 확정노선 개통 후"),
                           (axes[2], "개선m", "S1-S0 개선폭(m)")]:
        kw = dict(vmin=0, vmax=vmax, cmap="viridis_r") if col != "개선m" else dict(cmap="Reds")
        m.plot(column=col, ax=ax, legend=True, edgecolor="white", linewidth=0.2, **kw)
        ax.set_title(title)
        ax.set_axis_off()
    plt.tight_layout()
    plt.savefig("access_map_S0_S1.png", dpi=150)
    print("지도 저장: access_map_S0_S1.png")


def main():
    dong, st = load_inputs()
    G = download_graph(dong)
    ids, xy = node_table(G)
    nodes = gpd.GeoDataFrame({"osmid": ids}, geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=CRS_M)

    scen = {
        "S0": st[st["포함시나리오"].str.contains("S0")],
        "S1": st[st["포함시나리오"].str.contains("S1")],
    }
    for name, s in scen.items():
        nodes[name], n_used = nearest_station_distance(G, ids, xy, s)
        print(f"{name}: 역 {n_used}개 연결, 노드 평균 거리 {np.nanmean(nodes[name]):.0f} m")

    res = summarize(nodes, dong, "S0").join(summarize(nodes, dong, "S1"))
    res["개선m"] = (res["S0_평균m"] - res["S1_평균m"]).round(0)
    res["S1_순위(멀수록1)"] = res["S1_평균m"].rank(ascending=False, method="min").astype(int)

    keep = ["adm_cd2", "adm_nm", "sggnm", "총인구", "고령비율_65", "후기고령비율_75"]
    out = dong[keep].set_index("adm_cd2").join(res).sort_values("S1_평균m", ascending=False)
    out.to_csv("dong_access_S0_S1.csv", encoding="utf-8-sig")
    print("저장: dong_access_S0_S1.csv")

    nodes.to_file("access_nodes.gpkg", driver="GPKG")
    make_map(dong, res)

    print("\n[S1 기준 역까지 평균 보행거리가 먼 행정동 상위 15]")
    print(out[["adm_nm", "총인구", "고령비율_65", "S0_평균m", "S1_평균m", "S1_1500m초과비율"]].head(15).to_string())
    print("\n[S1로 개선 폭이 큰 행정동 상위 10]")
    print(out.sort_values("개선m", ascending=False)[["adm_nm", "S0_평균m", "S1_평균m", "개선m"]].head(10).to_string())


if __name__ == "__main__":
    main()
