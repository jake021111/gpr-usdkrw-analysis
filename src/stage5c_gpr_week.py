# 5단계 추가: GPR과 이후 일주일(0~5거래일) 환율 변화 비교
# 먼저 stage5b_events_expanded.py 실행 필요
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 레포 최상위에서 실행되도록
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ev = pd.read_csv("output/table9_events_expanded.csv")
v = ev.dropna(subset=["GPR 배수"]).copy()
Y = {"환율 그대로": "환율 0~5일%", "달러 효과 제거": "달러제거 0~5일%"}

# 1. 사건별 GPR 배수와 일주일 환율 변화
rows = []
for name, col in Y.items():
    f = smf.ols(f"Q('{col}') ~ Q('GPR 배수')", v).fit(cov_type="HC1")
    rows.append({"기준": name, "사건 수": len(v), "상관계수": v["GPR 배수"].corr(v[col]),
                 "순위 상관": v["GPR 배수"].corr(v[col], method="spearman"),
                 "GPR 배수 1 증가당 %p": f.params.iloc[1], "p값": f.pvalues.iloc[1]})
r1 = pd.DataFrame(rows)
print(r1.round(3).to_string(index=False))

# 2. GPR 배수 상·하위 절반 비교
med = v["GPR 배수"].median()
v["GPR 반응"] = np.where(v["GPR 배수"] >= med, "크게 뜀", "적게 뜀")
g = v.groupby("GPR 반응").agg(사건수=("사건", "size"), GPR배수=("GPR 배수", "median"),
                            환율=("환율 0~5일%", "mean"), 달러제거=("달러제거 0~5일%", "mean"),
                            상승비율=("달러제거 0~5일%", lambda x: (x > 0).mean()))
print("\n중앙값", round(med, 2), "\n", g.round(3).to_string())

# 3. 유형별 상관
print("\n유형별 상관 (달러 제거)")
for t, d in v.groupby("유형"):
    print(t, len(d), round(d["GPR 배수"].corr(d["달러제거 0~5일%"]), 3))

# 4. 전체 거래일: 오늘 GPR과 오늘 포함 6거래일 환율 변화
c = pd.read_csv("data/processed/merged_2020_2025_clean.csv", parse_dates=["Date"])
c["GPR_100"] = c["GPR"] / 100
c["fx6"] = c["r_USD_KRW"].rolling(6).sum().shift(-5)
c["dxy6"] = c["r_DXY"].rolling(6).sum().shift(-5)
c["brent6"] = c["r_Brent"].rolling(6).sum().shift(-5)
d = c.dropna(subset=["fx6", "dxy6", "brent6"])
m1 = smf.ols("fx6 ~ GPR_100", d).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
m2 = smf.ols("fx6 ~ GPR_100 + dxy6 + VIX + brent6", d).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
r2 = []
for name, m in [("GPR만", m1), ("달러·VIX·유가 통제", m2)]:
    lo, hi = m.conf_int().loc["GPR_100"]
    r2.append({"모형": name, "N": int(m.nobs), "GPR 100당 %p": m.params["GPR_100"], "95%하한": lo, "95%상한": hi,
               "p값": m.pvalues["GPR_100"], "R²": m.rsquared})
r2 = pd.DataFrame(r2)
print("\n", r2.round(3).to_string(index=False))
print("일주일 환율 변화 표준편차:", round(d["fx6"].std(), 2))
pd.concat([r1.assign(분석="사건 46개"), r2.assign(분석="전체 거래일")]).round(4).to_csv(
    "output/table13_gpr_week.csv", index=False, encoding="utf-8-sig")
