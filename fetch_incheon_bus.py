"""
인천 버스노선 데이터 수집 (공공데이터포털 '인천광역시 버스정보 버스노선 조회' API)

사용법
  1) pip install requests pandas pyproj
  2) 환경변수에 인증키 설정 (마이페이지 > 일반 인증키(Decoding) 복사)
       Windows(PowerShell):  $env:DATA_GO_KR_KEY="인증키"
       Mac/Linux/Colab    :  export DATA_GO_KR_KEY="인증키"
  3) python fetch_incheon_bus.py --stations stations_S0_incheon_buffer5km.csv

동작
  - 1단계: 노선번호목록 조회(getBusRouteNo)를 숫자 0~9로 호출해 노선 목록 + 첫차/막차/배차간격 수집
  - 2단계: 노선별 경유 정류소 목록(getBusRouteSectionList) 수집 (노선당 1회 호출)
  - 호출 결과는 bus_cache/ 폴더에 저장되므로, 일일 한도(개발계정 1,000건)에 걸리면
    다음 날 같은 명령을 다시 실행하면 이어서 진행됨
  - 3단계: TM좌표(Bessel TM127 중부원점)를 위경도로 변환.
    변환식 후보(EPSG:5174, 2097)를 S0 철도역과 이름이 같은 정류소로 검증해 더 맞는 쪽을 자동 선택

산출물: bus_routes.csv, bus_route_stops.csv
"""
import argparse
import csv
import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from urllib.parse import unquote

import requests

BASE = "http://apis.data.go.kr/6280000/busRouteService"
CACHE = "bus_cache"
DAILY_LIMIT = 950  # 개발계정 1,000건/일 (여유 50)
calls_made = 0
THROTTLE = 0.5  # 호출 사이 대기(초): 초당 요청제한 회피
HEADERS = {  # 서버가 브라우저가 아닌 요청을 막아서 브라우저 헤더를 흉내냄
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)
consec_fail = 0
MAX_CONSEC_FAIL = 10


def get_key():
    key = os.environ.get("DATA_GO_KR_KEY", "").strip()
    if not key:
        sys.exit("환경변수 DATA_GO_KR_KEY 가 설정되지 않았습니다.")
    return unquote(key) if "%" in key else key


def parse_response(content):
    """XML 응답 -> (resultCode, resultMsg, totalCount, [dict, ...])"""
    root = ET.fromstring(content)
    code = root.findtext(".//resultCode")
    msg = root.findtext(".//resultMsg")
    if code is None:
        msg = root.findtext(".//returnAuthMsg") or root.findtext(".//errMsg") or "알 수 없는 응답"
    total = root.findtext(".//totalCount")
    items = [{c.tag: (c.text or "").strip() for c in it} for it in root.iter("itemList")]
    return code, msg, int(total) if total and total.isdigit() else 0, items


def call(op, key, **params):
    global calls_made, consec_fail
    for attempt in range(6):
        if calls_made >= DAILY_LIMIT:
            raise RuntimeError("DAILY_LIMIT")
        time.sleep(THROTTLE)
        try:
            r = SESSION.get(f"{BASE}/{op}", params={"serviceKey": key, **params}, timeout=30)
            calls_made += 1
            code, msg, total, items = parse_response(r.content)
        except (requests.RequestException, ET.ParseError):
            if attempt == 5:
                raise
            time.sleep(2)
            continue
        if code in ("0", "00"):
            consec_fail = 0
            return total, items
        if msg and ("초당" in msg or "제한" in msg):  # 초당 요청제한: 잠시 쉬고 재시도 (실패로 세지 않음)
            time.sleep(3 + attempt * 2)
            continue
        consec_fail += 1
        if consec_fail >= MAX_CONSEC_FAIL:
            sys.exit(f"\n연속 {MAX_CONSEC_FAIL}회 실패로 중단합니다 (호출 한도 보호). 마지막 응답: {msg}\n"
                     f"원본 응답: {r.text[:500]}\n요청: {r.url.split('serviceKey')[0]}serviceKey=***")
        raise RuntimeError(f"API 오류 {code}: {msg}")
    raise RuntimeError("초당 요청제한이 계속됩니다. 잠시 후 다시 실행하세요.")


def cached(name, fn):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    data = fn()
    json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    time.sleep(0.05)
    return data


def discover_routes(key):
    """노선번호 접두 검색. 한 번에 너무 많이 나와 서버가 에러를 내면 한 자리씩 더 쪼개서 재시도."""
    routes = {}
    failed = []
    PAGE = 100
    MAX_DEPTH = 3

    def query(prefix):
        def fetch():
            out, page = [], 1
            while True:
                total, items = call("getBusRouteNo", key, routeNo=prefix, numOfRows=PAGE, pageNo=page)
                out += items
                if len(out) >= total or not items:
                    return out
                page += 1
        return cached(f"routeno_{prefix}.json", fetch)

    def run(prefix, depth):
        try:
            items = query(prefix)
        except RuntimeError as e:
            if str(e) == "DAILY_LIMIT":
                raise
            if depth < MAX_DEPTH and prefix.isdigit():
                print(f"  '{prefix}' 실패({e}) -> 한 자리 더 쪼개서 재시도")
                for d in "0123456789":
                    run(prefix + d, depth + 1)
            else:
                failed.append(prefix)
                print(f"  '{prefix}' 건너뜀: {e}")
            return
        for it in items:
            routes[it["ROUTEID"]] = it

    for d in "123456789":
        run(d, 1)
    for p in ["M", "E", "N", "급행", "광역", "순환", "마을", "강화", "옹진", "공항"]:
        run(p, MAX_DEPTH)

    print(f"노선 {len(routes)}개 발견" + (f" (실패한 검색어: {failed})" if failed else ""))
    return routes


def fetch_sections(key, route_id):
    def fetch():
        out, page = [], 1
        while True:
            total, items = call("getBusRouteSectionList", key, routeId=route_id, numOfRows=200, pageNo=page)
            out += items
            if len(out) >= total or not items:
                return out
            page += 1
    return cached(f"sections_{route_id}.json", fetch)


def haversine(lon1, lat1, lon2, lat2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl, dp = math.radians(lon2 - lon1), p2 - p1
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(a))


def choose_crs(stops, stations_csv):
    """후보 좌표계 중 S0 역과 이름이 같은 정류소가 가장 가깝게 떨어지는 쪽을 선택"""
    import pandas as pd
    from pyproj import Transformer
    cands = {"EPSG:5174": 5174, "EPSG:2097": 2097}
    if not stations_csv or not os.path.exists(stations_csv):
        print("stations 파일이 없어 EPSG:5174를 사용합니다 (검증 불가).")
        return 5174
    st = pd.read_csv(stations_csv)
    st = st[st["시나리오"] == "S0"]
    name2 = {}
    for _, r in st.iterrows():
        name2[r["역명"] + "역"] = (r["경도"], r["위도"])
        name2[r["역명"]] = (r["경도"], r["위도"])
    sample = [s for s in stops if s["BSTOPNM"] in name2][:400]
    best, best_med = 5174, None
    for label, epsg in cands.items():
        tr = Transformer.from_crs(epsg, 4326, always_xy=True)
        ds = []
        for s in sample:
            lon, lat = tr.transform(float(s["POSX"]), float(s["POSY"]))
            ds.append(min(haversine(lon, lat, *name2[s["BSTOPNM"]]), 5000))
        if ds:
            ds.sort()
            med = ds[len(ds) // 2]
            print(f"  {label}: 역 이름 일치 정류소 {len(ds)}개, 중앙 오차 {med:.0f} m")
            if best_med is None or med < best_med:
                best, best_med = epsg, med
    print(f"-> EPSG:{best} 선택")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", default="stations_S0_incheon_buffer5km.csv")
    args, _ = ap.parse_known_args()
    key = get_key()

    try:
        routes = discover_routes(key)
        all_stops = []
        for i, rid in enumerate(routes, 1):
            for s in fetch_sections(key, rid):
                all_stops.append(s)
            if i % 25 == 0:
                print(f"  경유 정류소 수집 {i}/{len(routes)} (오늘 호출 {calls_made}건)")
    except RuntimeError as e:
        if str(e) == "DAILY_LIMIT":
            print(f"\n일일 한도({DAILY_LIMIT}건)에 도달했습니다. 내일 같은 명령을 다시 실행하면 이어서 진행됩니다.")
            return
        raise

    from pyproj import Transformer
    epsg = choose_crs(all_stops, args.stations)
    tr = Transformer.from_crs(epsg, 4326, always_xy=True)

    with open("bus_routes.csv", "w", newline="", encoding="utf-8-sig") as f:
        cols = ["ROUTEID", "ROUTENO", "ROUTETPCD", "ADMINNM", "ROUTELEN", "FBUS_DEPHMS", "LBUS_DEPHMS",
                "MIN_ALLOCGAP", "MAX_ALLOCGAP", "ORIGIN_BSTOPNM", "DEST_BSTOPNM"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(routes.values())

    with open("bus_route_stops.csv", "w", newline="", encoding="utf-8-sig") as f:
        cols = ["ROUTEID", "BSTOPID", "BSTOPNM", "ADMINNM", "BSTOPSEQ", "DIRCD", "lon", "lat"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for s in all_stops:
            lon, lat = tr.transform(float(s["POSX"]), float(s["POSY"]))
            w.writerow({**{k: s.get(k, "") for k in cols[:6]}, "lon": round(lon, 6), "lat": round(lat, 6)})
    print(f"저장 완료: bus_routes.csv ({len(routes)}개 노선), bus_route_stops.csv ({len(all_stops)}행), 총 호출 {calls_made}건")


if __name__ == "__main__":
    main()
