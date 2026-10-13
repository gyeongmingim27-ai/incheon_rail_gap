"""
원자료(공공데이터포털·통계청·SGIS 등) → 분석용 가공 파일 전처리 스크립트

입력(raw/ 폴더에 둔다; 출처는 README 참고)
  raw/HangJeongDong_ver20260701_5.geojson            행정동 경계(전국)
  raw/202609_202609_연령별인구현황_월간*.xlsx          행정안전부 주민등록 연령별 인구(구별 11개 파일)
  raw/소요시간별_이용_교통수단별_통근_통학_인구_12세_이상_시군구_*.xlsx   KOSIS 2020 인구총조사(인천 시군구 파일)
  raw/_붙임2__2026__9월말_군__구별_자동차_등록_현황.hwpx   인천시 구·군별 자동차 등록 현황
  raw/전체_도시철도역사정보_20260630.xlsx             국가철도공단 도시철도 역사 정보
  raw/stations_S1_manual.csv                          계획 노선(7호선 청라 연장) 역 좌표(수기 입력)

출력(data/ 폴더)
  incheon_admdong_20260701.geojson, incheon_admdong_pop_202609.geojson, incheon_admdong_population_202609.csv,
  commute_mode_by_gu_2020.csv, car_registration_by_gu_202609.csv,
  stations_S0_incheon_buffer5km.csv, stations_S1_incheon_buffer5km.csv

사용: python prepare_data.py --raw raw --out data
"""
import argparse, glob, json, os, re, warnings, zipfile
import numpy as np
import pandas as pd
import geopandas as gpd

warnings.filterwarnings("ignore")
CRS_M = 5179
EXCLUDE_GU = ("강화군", "옹진군")        # 분석대상 제외(도서·농촌 지역)
STATION_BUFFER_M = 5000                  # 인천 경계 밖 5 km 안의 역까지 포함
NUM = lambda x: pd.to_numeric(str(x).replace(",", "").strip(), errors="coerce")


def prep_dong(raw, out):
    g = gpd.read_file(f"{raw}/HangJeongDong_ver20260701_5.geojson")
    g = g[g["sidonm"] == "인천광역시"].copy()
    g.to_file(f"{out}/incheon_admdong_20260701.geojson", driver="GeoJSON")
    return g


def prep_population(raw, out, g):
    rows = []
    for f in sorted(glob.glob(f"{raw}/202609_202609_연령별인구현황_월간*.xlsx")):
        d = pd.read_excel(f, header=None)
        hdr = d.iloc[3].tolist()
        d = d.iloc[4:].copy(); d.columns = hdr
        age_cols = [c for c in hdr if isinstance(c, str) and c.endswith(("세", "상")) and "~" in c or c == "100세 이상"]
        for _, r in d.iterrows():
            nm = str(r["행정기관"]).strip().replace(".", "·")          # 'ㅇㅇ1.ㅇㅇ동' → 경계 파일의 'ㅇㅇ1·ㅇㅇ동'
            nm = re.sub(r"(\S+면)\S*출장소$", r"\1", nm)               # 출장소 인구는 소속 면에 합산
            ages = {c: NUM(r[c]) for c in age_cols}
            rows.append({"adm_nm": nm, "총인구": NUM(r["총 인구수"]),
                         "65세이상": sum(v for c, v in ages.items() if int(re.match(r"\d+", c).group()) >= 65),
                         "75세이상": sum(v for c, v in ages.items() if int(re.match(r"\d+", c).group()) >= 75),
                         "20_39세": sum(v for c, v in ages.items() if 20 <= int(re.match(r"\d+", c).group()) <= 39)})
    p = pd.DataFrame(rows)
    p = p[p["adm_nm"].isin(g["adm_nm"])].groupby("adm_nm", as_index=False).sum()   # 동 단위 행만(출장소는 합산됨)
    m = g.merge(p, on="adm_nm", how="left")
    m["고령비율_65"] = (m["65세이상"] / m["총인구"] * 100).round(2)
    m["후기고령비율_75"] = (m["75세이상"] / m["총인구"] * 100).round(2)
    m["청년비율_20_39"] = (m["20_39세"] / m["총인구"] * 100).round(2)
    m["분석대상"] = ~m["sggnm"].isin(EXCLUDE_GU)
    for c in ["총인구", "65세이상", "75세이상", "20_39세"]:
        m[c] = m[c].astype("Int64")
    m.to_file(f"{out}/incheon_admdong_pop_202609.geojson", driver="GeoJSON")
    cols = ["adm_cd2", "adm_nm", "sggnm", "분석대상", "총인구", "65세이상", "75세이상", "20_39세",
            "고령비율_65", "후기고령비율_75", "청년비율_20_39"]
    m[cols].to_csv(f"{out}/incheon_admdong_population_202609.csv", index=False, encoding="utf-8-sig")
    return m


def prep_commute(raw, out):
    f = [x for x in glob.glob(f"{raw}/소요시간별_이용_교통수단별_통근_통학*.xlsx")
         if "인천" in "".join(pd.read_excel(x, header=None).iloc[:, 0].dropna().astype(str).tolist()[:3]) or True]
    mids = {"15분 미만": 7.5, "15~30": 22.5, "30~45": 37.5, "45~60": 52.5, "60~90": 75, "90~120": 105, "120분 이상": 135}
    out_rows = []
    for x in f:
        d = pd.read_excel(x, header=None)
        h = d.iloc[1].tolist(); d = d.iloc[2:].copy(); d.columns = h
        d.iloc[:, 0] = d.iloc[:, 0].ffill(); d.iloc[:, 1] = d.iloc[:, 1].ffill()
        d["gu"] = d.iloc[:, 0].astype(str).str.replace("　", "").str.strip()
        for gu in ["강화군", "계양구", "남동구", "동구", "미추홀구", "부평구", "서구", "연수구", "옹진군", "중구"]:
            for kind in ["계", "통근", "통학"]:
                s = d[(d["gu"] == gu) & (d.iloc[:, 1] == kind)]
                if s.empty:
                    continue
                tot = NUM(s[s.iloc[:, 2] == "계"].iloc[0]["통근·통학 인구(12세 이상)"])
                s = s.copy()
                v = lambda c: NUM(s[s.iloc[:, 2] == "계"].iloc[0][c])
                t = s[s.iloc[:, 2].isin(mids)]
                cnt = t["통근·통학 인구(12세 이상)"].map(NUM).fillna(0)   # "-" = 0명
                avg = float((cnt * t.iloc[:, 2].map(mids)).sum() / cnt.sum())
                long45 = float(cnt[t.iloc[:, 2].isin(["45~60", "60~90", "90~120", "120분 이상"])].sum() / tot * 100)
                out_rows.append({"구": gu, "구분": kind, "인구": int(tot),
                                 "승용차분담률": round(v("승용차,승합차(벤)") / tot * 100, 1),
                                 "버스": round(v("시내,좌석,마을버스") / tot * 100, 1),
                                 "지하철": round(v("전철,지하철") / tot * 100, 1),
                                 "도보자전거": round((v("걸어서") + v("자전거")) / tot * 100, 1),
                                 "복합": round(v("복합수단") / tot * 100, 1),
                                 "평균소요분": round(avg, 1), "장거리45분이상": round(long45, 1)})
    r = pd.DataFrame(out_rows).drop_duplicates(["구", "구분"]).sort_values(["구", "구분"], key=lambda c: c.map({"계": 0, "통근": 1, "통학": 2}) if c.name == "구분" else c)
    r.to_csv(f"{out}/commute_mode_by_gu_2020.csv", index=False, encoding="utf-8-sig")


def prep_car(raw, out, m):
    z = zipfile.ZipFile(glob.glob(f"{raw}/*자동차_등록_현황.hwpx")[0])
    x = z.read("Contents/section0.xml").decode("utf-8")
    res = {}
    for tr in re.findall(r"<hp:tr>(.*?)</hp:tr>", x, flags=re.S):
        cells = [("".join(re.findall(r"<hp:t>(.*?)</hp:t>", c))).replace(",", "").strip()
                 for c in re.findall(r"<hp:tc[ >].*?</hp:tc>", tr, flags=re.S)]
        if len(cells) >= 20 and cells[0] and not cells[0].isdigit():
            res[cells[0]] = cells            # [구, 합계, 관용, 자가용, 영업용, 승용계, 승용관용, 승용자가용, 승용영업용, ...]
    pop = m.groupby("sggnm").agg(총인구=("총인구", "sum"), p65=("65세이상", "sum"))
    out_rows = []
    for gu, c in res.items():
        if gu not in pop.index:
            continue
        tot, p65 = float(pop.loc[gu, "총인구"]), float(pop.loc[gu, "p65"])
        total, own, biz = float(c[5]), float(c[7]), float(c[8])
        out_rows.append({"구": gu, "총인구": int(tot), "승용_계": int(total), "승용_자가용": int(own), "승용_영업용": int(biz),
                         "자가용승용_천명당": round(own / tot * 1000), "영업용승용_비중": round(biz / total * 100, 1),
                         "고령비율": round(p65 / tot * 100, 1)})
    pd.DataFrame(out_rows).to_csv(f"{out}/car_registration_by_gu_202609.csv", index=False, encoding="utf-8-sig")


def prep_stations(raw, out, g):
    s = pd.read_excel(f"{raw}/전체_도시철도역사정보_20260630.xlsx")
    s = s.dropna(subset=["역위도", "역경도"])
    s = s[~s["노선명"].astype(str).str.contains("자기부상")]            # 공항 자기부상철도는 제외
    s["역사명"] = s["역사명"].astype(str).str.replace(r"\(.*?\)", "", regex=True).str.replace(r"(?<=.)역$", "", regex=True).str.strip()   # '가정(루원시티)'→'가정', '간석역'→'간석'
    s["노선명"] = s["노선명"].astype(str).str.split(",")
    s = s.explode("노선명"); s["노선명"] = s["노선명"].str.strip()
    pts = gpd.GeoDataFrame(s, geometry=gpd.points_from_xy(s["역경도"], s["역위도"]), crs=4326).to_crs(CRS_M)
    area = g.to_crs(CRS_M).union_all()
    pts["d"] = pts.geometry.distance(area)
    pts = pts[pts["d"] <= STATION_BUFFER_M]
    rows = []
    for name, grp in pts.groupby("역사명"):
        grp = grp.sort_values("d")
        lines = sorted(set(grp["노선명"].astype(str)))
        r0 = grp.iloc[0]
        rows.append({"역명": name, "노선": ", ".join(lines), "노선수": len(lines), "위도": grp["역위도"].mean(), "경도": grp["역경도"].mean(),
                     "인천경계까지거리_m": round(float(r0["d"])), "인천내부": bool(r0["d"] == 0)})
    S0 = pd.DataFrame(rows).sort_values(["인천경계까지거리_m", "역명"]).reset_index(drop=True)   # 인천 내부(거리 0)는 가나다순, 외부는 경계에서 가까운 순
    S0.insert(0, "station_id", [f"S0_{i+1:03d}" for i in range(len(S0))])
    S0["시나리오"] = "S0"
    S0.to_csv(f"{out}/stations_S0_incheon_buffer5km.csv", index=False, encoding="utf-8-sig")
    man = pd.read_csv(f"{raw}/stations_S1_manual.csv")      # 7호선 청라 연장 신설역(수기 좌표)
    a = S0.copy(); a["좌표상태"] = "원자료"; a["포함시나리오"] = "S0,S1"
    # 7호선 청라연장은 공항철도 청라국제도시역에서 환승 → 해당 역을 환승역(노선수 2)으로 표기
    t = a["역명"] == "청라국제도시"
    a.loc[t, "노선"] = a.loc[t, "노선"] + ", 7호선 청라연장"; a.loc[t, "노선수"] = a.loc[t, "노선수"] + 1
    b = man.copy()
    pd.concat([a, b], ignore_index=True).to_csv(f"{out}/stations_S1_incheon_buffer5km.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--raw", default="raw"); ap.add_argument("--out", default="data")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    g = prep_dong(a.raw, a.out)
    m = prep_population(a.raw, a.out, g)
    prep_commute(a.raw, a.out)
    prep_car(a.raw, a.out, m)
    prep_stations(a.raw, a.out, g)
    print("완료:", sorted(os.listdir(a.out)))
