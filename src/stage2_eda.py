# 2단계: 기초 분석 (기술통계, 시계열 그래프, 상관분석, ADF 검정)
# 사전 설치: pip install pandas numpy matplotlib statsmodels
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 레포 최상위에서 실행되도록
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

plt.rcParams["font.family"] = "Malgun Gothic"   # 한글 폰트 (Windows)
plt.rcParams["axes.unicode_minus"] = False

# 색 (dataviz 기본 팔레트)
BLUE, RED, GRAY_MID = "#2a78d6", "#e34948", "#f0efec"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

OUT = "output"
os.makedirs(OUT, exist_ok=True)
c = pd.read_csv("data/processed/merged_2020_2025_clean.csv", parse_dates=["Date"])

LEVELS = ["USD_KRW", "DXY", "VIX", "Brent", "GPR"]
MODEL = ["r_USD_KRW", "r_DXY", "VIX", "r_Brent", "GPR"]   # 회귀에 넣을 형태
LABEL = {
    "USD_KRW": "원/달러 환율 (원)", "DXY": "달러 인덱스", "VIX": "VIX (공포지수)",
    "Brent": "브렌트유 ($/배럴)", "GPR": "GPR (지정학 리스크)",
    "r_USD_KRW": "환율 변화율(%)", "r_DXY": "DXY 변화율(%)", "r_Brent": "유가 변화율(%)",
    "d_VIX": "VIX 변화량", "d_GPR": "GPR 변화량",
}
EVENTS = [
    ("2020-01-03", "미-이란 솔레이마니"), ("2020-03-11", "코로나 팬데믹"),
    ("2022-02-24", "러-우 침공"), ("2023-10-07", "하마스 공격"),
    ("2024-12-03", "비상계엄"), ("2025-04-02", "미 관세 발표"),
    ("2025-06-22", "미국 이란 핵시설 공격"),
]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# ---------- 1. 기술통계 ----------
cols = LEVELS + ["r_USD_KRW", "r_DXY", "r_Brent", "d_VIX", "d_GPR"]
desc = c[cols].describe().T
desc["skew"] = c[cols].skew()
desc["kurt"] = c[cols].kurt()
desc = desc[["count", "mean", "std", "min", "25%", "50%", "75%", "max", "skew", "kurt"]].round(3)
desc.insert(0, "설명", [LABEL[v] for v in cols])
desc.to_csv(f"{OUT}/table1_descriptive.csv", encoding="utf-8-sig")
print("=== 기술통계 ===\n", desc.to_string())

# ---------- 2. ADF 검정 ----------
rows = []
for v in cols:
    stat, p = adfuller(c[v].dropna(), autolag="AIC", result_object=False)[:2]
    rows.append({"변수": v, "설명": LABEL[v], "ADF 통계량": round(stat, 3),
                 "p값": round(p, 4), "판정": "정상" if p < 0.05 else "비정상"})
adf = pd.DataFrame(rows)
adf.to_csv(f"{OUT}/table2_adf.csv", index=False, encoding="utf-8-sig")
print("\n=== ADF ===\n", adf.to_string(index=False))

# ---------- 3. 상관분석 (회귀에 넣을 형태 기준) ----------
corr = c[MODEL].corr().round(3)
corr.to_csv(f"{OUT}/table3_correlation.csv", encoding="utf-8-sig")
print("\n=== 상관계수 ===\n", corr.to_string())

# 다중공선성 확인 (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor
X = c[MODEL[1:]].assign(const=1).values
vif = pd.Series([variance_inflation_factor(X, i) for i in range(len(MODEL) - 1)],
                index=MODEL[1:]).round(2)
print("\n=== VIF (10 넘으면 문제) ===\n", vif.to_string())

# ---------- 그림 1: 5개 변수 시계열 (변수마다 따로, 사건 표시) ----------
fig, axes = plt.subplots(5, 1, figsize=(11, 11), sharex=True)
for ax, v in zip(axes, LEVELS):
    ax.plot(c["Date"], c[v], color=BLUE, linewidth=1.2)
    ax.set_title(LABEL[v], loc="left", fontsize=11, color=INK, pad=4)
    for d, _ in EVENTS:
        ax.axvline(pd.Timestamp(d), color=INK2, linewidth=0.7, linestyle=(0, (3, 3)), alpha=0.6)
    style(ax)
for k, (d, _) in enumerate(EVENTS, 1):  # 사건 번호는 맨 아래 패널 위에만
    axes[-1].text(pd.Timestamp(d), 0.97, str(k), transform=axes[-1].get_xaxis_transform(),
                  ha="center", va="top", fontsize=8, color=INK,
                  bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec=INK2, lw=0.6))
fig.suptitle("5개 변수 흐름 (2020~2025)", x=0.01, ha="left", fontsize=14, color=INK)
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.text(0.01, 0.0,"점선 = 주요 사건   " + "   ".join(f"{k} {n}" for k, (_, n) in enumerate(EVENTS, 1)),
         fontsize=9, color=INK2)
fig.savefig(f"{OUT}/fig1_timeseries.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------- 그림 2: 상관계수 히트맵 (파랑-회색-빨강) ----------
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("div", [BLUE, GRAY_MID, RED])
fig, ax = plt.subplots(figsize=(6.5, 5.5))
shown = np.ma.masked_where(np.eye(len(MODEL), dtype=bool), corr.values)  # 대각선(자기 자신=1)은 회색
cmap.set_bad(GRID)
im = ax.imshow(shown, cmap=cmap, vmin=-1, vmax=1)
names = [LABEL[v] for v in MODEL]
ax.set_xticks(range(len(MODEL)), names, rotation=30, ha="right", fontsize=9, color=INK2)
ax.set_yticks(range(len(MODEL)), names, fontsize=9, color=INK2)
for i in range(len(MODEL)):
    for j in range(len(MODEL)):
        ax.text(j, i, "—" if i == j else f"{corr.values[i, j]:.2f}", ha="center", va="center",
                fontsize=10, color=INK2 if i == j else INK)
ax.set_xticks(np.arange(-.5, len(MODEL)), minor=True)
ax.set_yticks(np.arange(-.5, len(MODEL)), minor=True)
ax.grid(which="minor", color="white", linewidth=2)
ax.tick_params(which="both", length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, shrink=0.8)
cb.outline.set_visible(False)
cb.ax.tick_params(colors=INK2, labelsize=8, length=0)
ax.set_title("상관계수 (회귀에 넣을 형태)", loc="left", fontsize=12, color=INK)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_correlation.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------- 그림 3: GPR 수준별 환율 변화율 평균 ----------
c["GPR_group"] = pd.qcut(c["GPR"], 5, labels=["매우 낮음", "낮음", "보통", "높음", "매우 높음"])
g = c.groupby("GPR_group", observed=True)["r_USD_KRW"].agg(["mean", "count", "std"])
g["se"] = g["std"] / np.sqrt(g["count"])
g.round(4).to_csv(f"{OUT}/table4_fx_by_gpr_group.csv", encoding="utf-8-sig")
print("\n=== GPR 5분위별 환율 변화율 ===\n", g.round(4).to_string())

fig, ax = plt.subplots(figsize=(7, 4.2))
x = np.arange(len(g))
ax.bar(x, g["mean"], width=0.6, color=BLUE, edgecolor="white", linewidth=2)
ax.errorbar(x, g["mean"], yerr=1.96 * g["se"], fmt="none", ecolor=INK2, elinewidth=1, capsize=4)
ax.axhline(0, color=INK2, linewidth=0.8)
for xi, m, se in zip(x, g["mean"], g["se"]):   # 값은 신뢰구간 위쪽 끝 위에
    ax.text(xi, m + 1.96 * se + 0.004, f"{m:+.3f}%".replace("-0.000", "0.000"),
            ha="center", va="bottom", fontsize=9, color=INK)
ax.set_xticks(x, [f"{i}\n(GPR {lo:.0f}~{hi:.0f})" for i, (lo, hi) in zip(
    g.index, c.groupby("GPR_group", observed=True)["GPR"].agg(["min", "max"]).values)], fontsize=9)
ax.set_ylabel("하루 환율 변화율 평균 (%)", color=INK2, fontsize=9)
ax.set_ylim(top=ax.get_ylim()[1] + 0.015)   # 위쪽 숫자 공간
ax.set_title("GPR이 높은 날, 환율이 더 올랐을까?  (막대 = 평균, 선 = 95% 신뢰구간)",
             loc="left", fontsize=11, color=INK)
style(ax)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_fx_by_gpr.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"\n저장 완료 → {OUT}/")
