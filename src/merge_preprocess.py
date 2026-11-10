# 사전 설치: pip install pandas numpy statsmodels
# 5개 변수 합치기 + 전처리
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 레포 최상위에서 실행되도록
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

START, END = "2020-01-01", "2025-12-31"


def load(path, date_col, value_col, name, **kw):
    """FRED/Yahoo CSV 읽기 → Date, name 두 컬럼. FRED 빈칸('.')은 NaN"""
    d = pd.read_csv(path, na_values=[".", ""], **kw)
    d = d[[date_col, value_col]].rename(columns={date_col: "Date", value_col: name})
    d["Date"] = pd.to_datetime(d["Date"])
    d[name] = pd.to_numeric(d[name], errors="coerce")
    return d[(d["Date"] >= START) & (d["Date"] <= END)]


fx = load("data/raw/USDKRW_FRED_20200101_20251231.csv", "DATE", "DEXKOUS", "USD_KRW")
dxy = load("data/raw/DXY_ICE_2020102_20251230.csv", "Price", "Close", "DXY", skiprows=[1, 2])  # Yahoo 헤더 3줄
vix = load("data/raw/VIXCLS_(1).csv", "observation_date", "VIXCLS", "VIX")
brent = load("data/raw/DCOILBRENTEU.csv", "observation_date", "DCOILBRENTEU", "Brent")

# ---------- 1. 원자료 점검 ----------
print("=== 원자료 점검 (기간 자른 뒤) ===")
for name, d in [("USD_KRW", fx), ("DXY", dxy), ("VIX", vix), ("Brent", brent)]:
    print(f"{name:8s} {len(d):5d}행  {d['Date'].min().date()} ~ {d['Date'].max().date()}  "
          f"빈칸 {d[name].isna().sum():3d}  min {d[name].min():8.2f}  max {d[name].max():8.2f}")

# ---------- 2. 기준 날짜 = 환율 값이 있는 날 ----------
fx = fx.dropna()
dates = fx["Date"].sort_values().reset_index(drop=True)

# ---------- 3. GPR: 쉬는 날(주말·미국 공휴일) 값을 다음 거래일에 평균 ----------
# 예) 월요일 GPR = 토·일·월 평균, 공휴일 다음날 GPR = 공휴일·그날 평균
gpr = pd.read_csv("data/processed/GPR_daily_2020_2025.csv", parse_dates=["date"])
gpr = gpr[["date", "GPRD", "GPRD_ACT", "GPRD_THREAT"]].rename(
    columns={"GPRD": "GPR", "GPRD_ACT": "GPR_ACT", "GPRD_THREAT": "GPR_THREAT"})


def fold_gpr(gpr, dates):
    """각 날짜의 GPR을 그 날짜 이후 첫 거래일로 옮겨 평균 (빠진 날의 뉴스도 버리지 않음)"""
    dates = pd.Series(dates).sort_values().reset_index(drop=True)
    pos = dates.searchsorted(gpr["date"])              # 각 날짜 이후 첫 거래일
    g = gpr[pos < len(dates)].copy()
    g["Date"] = dates.values[pos[pos < len(dates)]]
    n = g.groupby("Date").size().rename("GPR_days")    # 몇 일치를 평균했는지
    return g.groupby("Date")[["GPR", "GPR_ACT", "GPR_THREAT"]].mean().join(n).reset_index()


gpr_daily = gpr
gpr = fold_gpr(gpr_daily, dates)

# ---------- 4. 합치기 (환율 날짜 기준 left join) ----------
m = fx
for d in (dxy, vix, brent, gpr):
    m = m.merge(d, on="Date", how="left")
m.to_csv("data/processed/merged_2020_2025_raw.csv", index=False, encoding="utf-8-sig")

cols = ["USD_KRW", "DXY", "VIX", "Brent", "GPR"]
print(f"\n=== 합친 뒤 빈칸 (총 {len(m)}일) ===")
print(m[cols].isna().sum().to_string())
na_rows = m[m[cols].isna().any(axis=1)]
print(f"하나라도 빈 날: {len(na_rows)}일 →", ", ".join(na_rows["Date"].dt.strftime("%Y-%m-%d")))

# ---------- 5. 빈칸 처리: 하나라도 비면 그날 제외 ----------
c = m.dropna(subset=cols).reset_index(drop=True)
# 뺀 날은 다음 행 변화율에 이틀치 환율 변화가 들어가므로, GPR도 그 기간을 평균해서 다시 넣음
c = c.drop(columns=["GPR", "GPR_ACT", "GPR_THREAT", "GPR_days"]).merge(
    fold_gpr(gpr_daily, c["Date"]), on="Date", how="left")

# ---------- 6. 변수 변환 ----------
for v in ["USD_KRW", "DXY", "Brent"]:                  # 로그수익률(%)
    c[f"r_{v}"] = np.log(c[v]).diff() * 100
c["d_VIX"] = c["VIX"].diff()                           # 변화량
c["d_GPR"] = c["GPR"].diff()
c["GPR_lag1"] = c["GPR"].shift(1)                      # 하루 늦춘 GPR
c["log_GPR"] = np.log(c["GPR"])

# 급등일 (머신러닝용): 환율 상승률 상위 10%
c["surge"] = (c["r_USD_KRW"] >= c["r_USD_KRW"].quantile(0.90)).astype(int)
# 코로나 구간 표시 (강건성 체크용)
c["covid"] = c["Date"].between("2020-02-01", "2020-06-30").astype(int)

c = c.iloc[1:].reset_index(drop=True)                  # 첫날은 수익률 계산 불가
c.to_csv("data/processed/merged_2020_2025_clean.csv", index=False, encoding="utf-8-sig")
print(f"\n전처리 후: {c.shape[0]}일, {c['Date'].min().date()} ~ {c['Date'].max().date()}")

# ---------- 7. 기술통계 ----------
use = ["r_USD_KRW", "r_DXY", "VIX", "d_VIX", "r_Brent", "GPR", "d_GPR"]
print("\n=== 기술통계 ===")
print(c[use].describe().T[["mean", "std", "min", "max"]].round(3))

# 이상치: 하루 변화가 평균에서 4표준편차 넘는 날 (표시만, 삭제 안 함)
print("\n=== 극단값 (|z| > 4) ===")
for v in ["r_USD_KRW", "r_DXY", "r_Brent", "d_VIX", "d_GPR"]:
    z = (c[v] - c[v].mean()) / c[v].std()
    big = c.loc[z.abs() > 4, ["Date", v]]
    print(f"{v:10s} {len(big)}일:", ", ".join(f"{d:%Y-%m-%d}({x:+.1f})" for d, x in big.values))

# ---------- 8. ADF 정상성 검정 (p < 0.05면 분석에 바로 써도 됨) ----------
print("\n=== ADF 검정 ===")
for v in ["USD_KRW", "DXY", "Brent"] + use:
    p = adfuller(c[v].dropna(), autolag="AIC", result_object=False)[1]
    print(f"{v:10s} p = {p:.4f}  {'정상 ✅' if p < 0.05 else '비정상 ❌'}")

print("\n급등일:", c["surge"].sum(), "일 (기준: 하루 +",
      round(c["r_USD_KRW"].quantile(0.90), 3), "% 이상)")
