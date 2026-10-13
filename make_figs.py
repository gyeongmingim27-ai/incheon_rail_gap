import geopandas as gpd, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from PIL import Image
plt.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["hatch.linewidth"] = 0.6

COL = {"이중부담형": "#c0392b", "탄소형": "#e8873a", "격차형": "#3a7cc3", "저부담형": "#9ccb9a", "비공백": "#e4e4e4"}
dong = gpd.read_file("incheon_admdong_pop_202609.geojson")
dong["adm_cd2"] = dong["adm_cd2"].astype(str)
dong = dong[dong["분석대상"].astype(str).str.lower().isin(["true", "1"])].to_crs(5179)
d = pd.read_csv("dong_bus_link.csv", dtype={"adm_cd2": str})
g = pd.read_csv("gap_final.csv", dtype={"adm_cd2": str})
acc = pd.read_csv("dong_access_S0_S1.csv", dtype={"adm_cd2": str})
m = dong[["adm_cd2", "geometry"]].merge(d, on="adm_cd2").merge(g[["adm_cd2", "고령절감_현재"]], on="adm_cd2", how="left")
m["short"] = m["adm_nm"].str.split().str[-1]

# ---------- 그림 2: 유형 + 버스 판정 + 우선 대상 ----------
CARB = ["송도5동", "불로대곡동", "청라2동", "서창2동", "동춘1동"]
GAPP = ["청학동", "계양1동", "장수서창동", "가좌1동", "학익2동"]
fig, ax = plt.subplots(figsize=(7.2, 7.2))
for t, c in COL.items():
    s = m[m["유형"] == t]
    if len(s):
        s.plot(ax=ax, color=c, edgecolor="white", linewidth=0.3)
real = m[m["버스판정"] == "실질 공백"]
real.plot(ax=ax, facecolor="none", edgecolor="#222222", hatch="////", linewidth=0.0)
real.boundary.plot(ax=ax, color="#222222", linewidth=0.5)
for lab, names in [("C", CARB), ("G", GAPP)]:
    for i, n in enumerate(names, 1):
        s = m[m["short"] == n]
        s.boundary.plot(ax=ax, color="black", linewidth=1.8)
        p = s.geometry.iloc[0].representative_point()
        ax.text(p.x, p.y, f"{lab}{i}", ha="center", va="center", fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="black", lw=0.6))
cnt = m.groupby("유형").agg(n=("short", "size"), pop=("총인구", "sum"))
hand = [Patch(fc=COL[t], ec="none", label=f"{t} {cnt.loc[t,'n']}개 동 · {cnt.loc[t,'pop']/1e4:.1f}만")
        for t in ["이중부담형", "탄소형", "격차형", "저부담형", "비공백"]]
nreal = len(real); preal = real["총인구"].sum() / 1e4
hand += [Patch(fc="white", ec="#222222", hatch="////", label=f"실질 공백(버스로도 15분 초과) {nreal}개 동 · {preal:.1f}만"),
         Line2D([0], [0], color="black", lw=1.8, label="C1~5 탄소 우선 / G1~5 격차 우선")]
ax.legend(handles=hand, loc="upper left", fontsize=7.2, frameon=False, borderaxespad=0)
ax.set_title("연계 공백 유형, 버스 판정, 우선 적용 대상", fontsize=10)
ax.set_axis_off()
plt.tight_layout()
plt.savefig("w_fig2_map.png", dpi=220, bbox_inches="tight")
plt.close()

# 그림 2·3 합치기
a = Image.open("w_fig2_map.png").convert("RGB"); b = Image.open("shap_summary.png").convert("RGB")
h = a.height
b = b.resize((int(b.width * h * 0.62 / b.height), int(h * 0.62)))
canvas = Image.new("RGB", (a.width + b.width + 40, h), "white")
canvas.paste(a, (0, 0)); canvas.paste(b, (a.width + 40, (h - b.height) // 2))
canvas.save("w_fig23.png")

# ---------- 그림 4: 시나리오 + 유형별 인구(버스 판정) ----------
sc = pd.read_csv("carbon_scenarios.csv")
print(sc.head(6))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.3), gridspec_kw={"width_ratios": [1.1, 1]})
rates = ["5%", "10%", "20%"]
base = sc[sc["가정"].str.startswith("기본")]["연간감축_tCO2"].to_numpy() / 1e4
cons = sc[sc["가정"].str.startswith("보수")]["연간감축_tCO2"].to_numpy() / 1e4
x = np.arange(3); w = 0.36
b1 = ax1.bar(x - w/2, base, w, color="#2e7d5b", label="기본 (승용차 173, 연계 55)")
b2 = ax1.bar(x + w/2, cons, w, color="#9fcc9f", label="보수적 (승용차 163, 연계=승용차)")
for bars in (b1, b2):
    for r in bars:
        ax1.text(r.get_x() + r.get_width()/2, r.get_height() + 0.05, f"{r.get_height():.1f}", ha="center", fontsize=8)
ax1.set_xticks(x); ax1.set_xticklabels([f"{r} 전환" for r in rates], fontsize=8.5)
ax1.set_ylabel("연간 감축량 (만 tCO₂)", fontsize=8.5); ax1.tick_params(labelsize=8)
ax1.legend(fontsize=7.5, frameon=False, loc="upper left")
ax1.set_title("승용차 통근 → 철도 전환 시 연간 감축량 (탄소·이중부담형 26개 동)", fontsize=9)
ax1.spines[["top", "right"]].set_visible(False)

types = ["탄소형", "격차형", "이중부담형", "저부담형"]
gg = m[m["유형"].isin(types)]
bus = gg[gg["버스판정"] == "버스 보완"].groupby("유형")["총인구"].sum().reindex(types).fillna(0) / 1e4
rl = gg[gg["버스판정"] == "실질 공백"].groupby("유형")["총인구"].sum().reindex(types).fillna(0) / 1e4
y = np.arange(len(types))[::-1]
ax2.barh(y, rl, color=[COL[t] for t in types], edgecolor="#222222", hatch="////", lw=0.4, label="실질 공백")
ax2.barh(y, bus, left=rl, color=[COL[t] for t in types], alpha=0.35, label="버스 보완")
for yi, t in zip(y, types):
    ax2.text(rl[t] + bus[t] + 0.8, yi, f"{rl[t]:.1f} / {rl[t]+bus[t]:.1f}만", va="center", fontsize=8)
ax2.set_yticks(y); ax2.set_yticklabels(types, fontsize=8.5)
ax2.set_xlim(0, 80); ax2.set_xlabel("인구 (만 명)", fontsize=8.5); ax2.tick_params(labelsize=8)
ax2.legend(handles=[Patch(fc="#888888", ec="#222222", hatch="////", label="실질 공백"),
                    Patch(fc="#888888", alpha=0.35, label="버스 보완")], fontsize=7.5, frameon=False, loc="lower right")
ax2.set_title("유형별 인구: 실질 공백 / 공백 전체", fontsize=9)
ax2.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("w_fig4.png", dpi=220, bbox_inches="tight")
plt.close()

# ---------- 그림 1: 접근성 지도 ----------
a1 = dong[["adm_cd2", "geometry"]].merge(acc, on="adm_cd2")
vmax = np.nanpercentile(a1[["S0_평균m", "S1_평균m"]].values, 95)
fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
for ax, col, title in [(axes[0], "S0_평균m", "S0 현재: 역까지 평균 보행거리 (m)"),
                       (axes[1], "S1_평균m", "S1 7호선 청라 연장 개통 후 (m)"),
                       (axes[2], "개선m", "S1 − S0 개선폭 (m)")]:
    kw = dict(vmin=0, vmax=vmax, cmap="viridis_r", legend_kwds={"extend": "max", "shrink": 0.75}) if col != "개선m" \
        else dict(cmap="Reds", vmin=0, legend_kwds={"shrink": 0.75})
    a1.plot(column=col, ax=ax, legend=True, edgecolor="white", linewidth=0.2, **kw)
    ax.set_title(title, fontsize=10); ax.set_axis_off()
plt.tight_layout()
plt.savefig("w_fig1.png", dpi=200, bbox_inches="tight")
print("ok")
