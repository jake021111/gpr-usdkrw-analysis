# 3단계: 회귀분석 (Newey-West 표준오차)
# 사전 설치: pip install pandas numpy matplotlib statsmodels
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 레포 최상위에서 실행되도록
import os

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
BLUE, INK, INK2, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"

OUT = "output"
os.makedirs(OUT, exist_ok=True)
c = pd.read_csv("data/processed/merged_2020_2025_clean.csv", parse_dates=["Date"])
HAC_LAGS = int(4 * (len(c) / 100) ** (2 / 9))       # Newey-West 시차 (자주 쓰는 공식)

# GPR은 100 단위로 해석 (계수 = GPR 100 오를 때 환율 변화율 %p)
for v in ["GPR", "GPR_ACT", "GPR_THREAT", "GPR_lag1", "d_GPR"]:
    c[v + "_100"] = c[v] / 100
c["r_DXY_lag1"] = c["r_DXY"].shift(1)                # DXY는 뉴욕 오후 종가, 환율은 뉴욕 정오라 시점이 어긋남
c["r_Brent_lag1"] = c["r_Brent"].shift(1)
c["d_VIX_lag1"] = c["d_VIX"].shift(1)
c["year"] = c["Date"].dt.year

CTRL = "r_DXY + VIX + r_Brent"


def fit(formula, data):
    return smf.ols(formula, data).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})


noncovid = c[c["covid"] == 0]
P1 = c[c["year"].between(2020, 2021)]
P2 = c[c["year"].between(2022, 2023)]
P3 = c[c["year"].between(2024, 2025)]

SPECS = [  # (이름, 볼 변수, 식, 데이터)
    ("① 기본 모형", "GPR_100", f"r_USD_KRW ~ {CTRL} + GPR_100", c),
    ("② 실제 사건(ACT)", "GPR_ACT_100", f"r_USD_KRW ~ {CTRL} + GPR_ACT_100 + GPR_THREAT_100", c),
    ("② 위협(THREAT)", "GPR_THREAT_100", f"r_USD_KRW ~ {CTRL} + GPR_ACT_100 + GPR_THREAT_100", c),
    ("③ GPR 하루 늦춤", "GPR_lag1_100", f"r_USD_KRW ~ {CTRL} + GPR_lag1_100", c),
    ("④ GPR 변화량", "d_GPR_100", "r_USD_KRW ~ r_DXY + d_VIX + r_Brent + d_GPR_100", c),
    ("⑤ 코로나 구간 제외", "GPR_100", f"r_USD_KRW ~ {CTRL} + GPR_100", noncovid),
    ("⑥ 시점 보정(전날 통제변수 추가)", "GPR_100",
     f"r_USD_KRW ~ {CTRL} + r_DXY_lag1 + r_Brent_lag1 + d_VIX_lag1 + GPR_100", c),
    ("⑦ 2020~2021", "GPR_100", f"r_USD_KRW ~ {CTRL} + GPR_100", P1),
    ("⑧ 2022~2023", "GPR_100", f"r_USD_KRW ~ {CTRL} + GPR_100", P2),
    ("⑨ 2024~2025", "GPR_100", f"r_USD_KRW ~ {CTRL} + GPR_100", P3),
]

rows = []
for name, var, f, d in SPECS:
    r = fit(f, d)
    # GPR을 뺀 모형과 비교한 R² 증가분
    f0 = f
    for t in [t for t in r.params.index if "GPR" in t]:
        f0 = f0.replace(" + " + t, "")
    r0 = smf.ols(f0, d.loc[r.model.data.row_labels]).fit()
    lo, hi = r.conf_int().loc[var]
    rows.append({"모형": name, "N": int(r.nobs), "GPR 계수(100당,%p)": r.params[var], "표준오차": r.bse[var],
                 "95%하한": lo, "95%상한": hi, "p값": r.pvalues[var], "R²": r.rsquared,
                 "GPR 추가 R² 증가": r.rsquared - r0.rsquared})
res = pd.DataFrame(rows)
res.round(4).to_csv(f"{OUT}/table5_regression_gpr.csv", index=False, encoding="utf-8-sig")
print(res.round(4).to_string(index=False))

# 기본 모형 전체 계수
base = fit(f"r_USD_KRW ~ {CTRL} + GPR_100", c)
full = pd.DataFrame({"계수": base.params, "표준오차": base.bse, "t값": base.tvalues, "p값": base.pvalues})
full.round(4).to_csv(f"{OUT}/table6_regression_base.csv", encoding="utf-8-sig")
print("\n=== 기본 모형 ===\n", full.round(4).to_string())
print("N", int(base.nobs), "R2", round(base.rsquared, 4), "HAC lags", HAC_LAGS)
timed = fit(SPECS[6][2], c)
print("\n=== 시점 보정 모형 ===\n", pd.DataFrame({"계수": timed.params, "p값": timed.pvalues}).round(4).to_string())
print("R2", round(timed.rsquared, 4))
print("GPR 1표준편차 효과(%):", round(base.params["GPR_100"] * c["GPR_100"].std(), 4),
      " GPR 표준편차:", round(c["GPR"].std(), 1), " 환율변화율 표준편차:", round(c["r_USD_KRW"].std(), 3))
print("ACT/THREAT 동시 검정 p:", round(float(fit(SPECS[1][2], c).wald_test(
    "GPR_ACT_100 = 0, GPR_THREAT_100 = 0", scalar=True).pvalue), 4))

# 그림: GPR 계수와 95% 신뢰구간
fig, ax = plt.subplots(figsize=(8.5, 5))
y = list(range(len(res)))[::-1]
ax.hlines(y, res["95%하한"], res["95%상한"], color=BLUE, linewidth=2)
ax.scatter(res["GPR 계수(100당,%p)"], y, color=BLUE, s=45, zorder=3, edgecolor="white", linewidth=1.5)
ax.axvline(0, color=INK2, linewidth=0.9)
ax.set_yticks(y, res["모형"], fontsize=9, color=INK)
ax.set_xlabel("GPR이 100 오를 때 하루 환율 변화율 차이 (%p)", fontsize=9, color=INK2)
ax.set_title("GPR 계수와 95% 신뢰구간: 선이 0을 지나면 통계적으로 뚜렷하지 않음", loc="left", fontsize=11, color=INK)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(length=0, labelsize=9, colors=INK2)
ax.grid(axis="x", color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_regression_gpr.png", dpi=150, bbox_inches="tight", facecolor="white")
