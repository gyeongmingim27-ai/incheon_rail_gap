"""보고서(HWP 양식) 삽입용 그림: 칸 폭 120mm 기준 인쇄 크기로 생성"""
import geopandas as gpd, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
plt.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["hatch.linewidth"] = 0.4
MM = 1 / 25.4
COL = {"이중부담형": "#c0392b", "탄소형": "#e8873a", "격차형": "#3a7cc3", "저부담형": "#9ccb9a", "비공백": "#e4e4e4"}
dong = gpd.read_file("incheon_admdong_pop_202609.geojson"); dong["adm_cd2"] = dong["adm_cd2"].astype(str)
dong = dong[dong["분석대상"].astype(str).str.lower().isin(["true", "1"])].to_crs(5179)
d = pd.read_csv("dong_bus_link.csv", dtype={"adm_cd2": str})
acc = pd.read_csv("dong_access_S0_S1.csv", dtype={"adm_cd2": str})
m = dong[["adm_cd2", "geometry"]].merge(d, on="adm_cd2"); m["short"] = m["adm_nm"].str.split().str[-1]

# 그림 1: 접근성 지도 3장 (120 x 44 mm)
a1 = dong[["adm_cd2", "geometry"]].merge(acc, on="adm_cd2")
vmax = np.nanpercentile(a1[["S0_평균m", "S1_평균m"]].values, 95)
fig, axes = plt.subplots(1, 3, figsize=(120 * MM, 46 * MM))
for ax, col, title in [(axes[0], "S0_평균m", "S0 현재 (m)"), (axes[1], "S1_평균m", "S1 7호선 청라 연장 후 (m)"),
                       (axes[2], "개선m", "S1 − S0 개선폭 (m)")]:
    kw = dict(vmin=0, vmax=vmax, cmap="viridis_r") if col != "개선m" else dict(cmap="Reds", vmin=0)
    a1.plot(column=col, ax=ax, legend=True, edgecolor="white", linewidth=0.1,
            legend_kwds={"shrink": 0.7, "orientation": "horizontal", "pad": 0.02, **({"extend": "max"} if col != "개선m" else {})}, **kw)
    ax.set_axis_off(); ax._ttl = title
    cb = ax.get_figure().axes[-1]; cb.tick_params(labelsize=5.5, length=1.5, pad=1)
plt.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.02, wspace=0.05)
for ax in axes:
    bb = ax.get_position(); fig.text((bb.x0 + bb.x1) / 2, 0.955, ax._ttl, ha="center", va="center", fontsize=7)
plt.savefig("h_fig1.png", dpi=300); plt.close()

# 그림 2: 유형·버스 판정·우선 대상 지도 (110 x 100 mm)
CARB = ["송도5동", "불로대곡동", "청라2동", "서창2동", "동춘1동"]
GAPP = ["청학동", "계양1동", "장수서창동", "가좌1동", "학익2동"]
fig, ax = plt.subplots(figsize=(110 * MM, 100 * MM))
for t, c in COL.items():
    s = m[m["유형"] == t]
    if len(s): s.plot(ax=ax, color=c, edgecolor="white", linewidth=0.25)
real = m[m["버스판정"] == "실질 공백"]
real.plot(ax=ax, facecolor="none", edgecolor="#222222", hatch="/////", linewidth=0.0)
real.boundary.plot(ax=ax, color="#222222", linewidth=0.35)
for lab, names in [("C", CARB), ("G", GAPP)]:
    for i, n in enumerate(names, 1):
        s = m[m["short"] == n]; s.boundary.plot(ax=ax, color="black", linewidth=1.2)
        p = s.geometry.iloc[0].representative_point()
        ax.text(p.x, p.y, f"{lab}{i}", ha="center", va="center", fontsize=6, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="black", lw=0.4))
cnt = m.groupby("유형").agg(n=("short", "size"), pop=("총인구", "sum"))
hand = [Patch(fc=COL[t], ec="none", label=f"{t} {cnt.loc[t,'n']}개 동 · {cnt.loc[t,'pop']/1e4:.1f}만")
        for t in ["이중부담형", "탄소형", "격차형", "저부담형", "비공백"]]
hand += [Patch(fc="white", ec="#222222", hatch="/////", label=f"실질 공백(버스로도 15분 초과) {len(real)}개 동"),
         Line2D([0], [0], color="black", lw=1.2, label="C1~5 탄소 우선 / G1~5 격차 우선")]
ax.legend(handles=hand, loc="upper left", fontsize=6, frameon=False, borderaxespad=0, handlelength=1.6)
ax.set_axis_off(); plt.subplots_adjust(0, 0, 1, 1)
plt.savefig("h_fig2.png", dpi=300); plt.close()

# 그림 4: 감축 시나리오 + 유형별 인구 (120 x 46 mm)
sc = pd.read_csv("carbon_scenarios.csv")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(120 * MM, 48 * MM), gridspec_kw={"width_ratios": [1, 1.05]})
base = sc[sc["가정"].str.startswith("기본")]["연간감축_tCO2"].to_numpy() / 1e4
cons = sc[sc["가정"].str.startswith("보수")]["연간감축_tCO2"].to_numpy() / 1e4
x = np.arange(3); w = 0.38
b1 = ax1.bar(x - w/2, base, w, color="#2e7d5b", label="기본(승용차 173)")
b2 = ax1.bar(x + w/2, cons, w, color="#9fcc9f", label="보수적(승용차 163)")
for bars in (b1, b2):
    for r in bars:
        ax1.text(r.get_x() + r.get_width()/2, r.get_height() + 0.06, f"{r.get_height():.1f}", ha="center", fontsize=5.5)
ax1.set_xticks(x); ax1.set_xticklabels(["5% 전환", "10% 전환", "20% 전환"], fontsize=6)
ax1.set_ylabel("연간 감축량 (만 tCO₂)", fontsize=6); ax1.tick_params(labelsize=5.5, length=2)
ax1.legend(fontsize=5.5, frameon=False, loc="upper left"); ax1.set_ylim(0, 5)
ax1.set_title("탄소·이중부담형 26개 동 통근 전환 시 감축량", fontsize=6.5, pad=3)
ax1.spines[["top", "right"]].set_visible(False)
types = ["탄소형", "격차형", "이중부담형", "저부담형"]
gg = m[m["유형"].isin(types)]
bus = gg[gg["버스판정"] == "버스 보완"].groupby("유형")["총인구"].sum().reindex(types).fillna(0) / 1e4
rl = gg[gg["버스판정"] == "실질 공백"].groupby("유형")["총인구"].sum().reindex(types).fillna(0) / 1e4
y = np.arange(len(types))[::-1]
ax2.barh(y, rl, color=[COL[t] for t in types], edgecolor="#222222", hatch="/////", lw=0.3)
ax2.barh(y, bus, left=rl, color=[COL[t] for t in types], alpha=0.35)
for yi, t in zip(y, types):
    ax2.text(rl[t] + bus[t] + 1, yi, f"{rl[t]:.1f} / {rl[t]+bus[t]:.1f}만", va="center", fontsize=5.5)
ax2.set_yticks(y); ax2.set_yticklabels(types, fontsize=6); ax2.set_xlim(0, 85)
ax2.set_xlabel("인구 (만 명)", fontsize=6); ax2.tick_params(labelsize=5.5, length=2)
ax2.legend(handles=[Patch(fc="#888888", ec="#222222", hatch="/////", label="실질 공백"),
                    Patch(fc="#888888", alpha=0.35, label="버스 보완")], fontsize=5.5, frameon=False, loc="lower right")
ax2.set_title("유형별 인구: 실질 공백 / 공백 전체", fontsize=6.5, pad=3)
ax2.spines[["top", "right"]].set_visible(False)
plt.tight_layout(pad=0.4)
plt.savefig("h_fig4.png", dpi=300); plt.close()
print("ok")
