# 사전 설치: pip install pandas xlrd
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 레포 최상위에서 실행되도록
import pandas as pd

START, END = "2020-01-01", "2025-12-31"
TAG = "2020_2025"

# ---------- 0. 원자료 → CSV (기간 자르기만) ----------
df = pd.read_excel("data/raw/data_gpr_daily_recent.xls")
df["date"] = pd.to_datetime(df["date"])
raw = df[(df["date"] >= START) & (df["date"] <= END)]
raw.to_csv(f"data/processed/GPR_daily_{TAG}.csv", index=False, encoding="utf-8-sig")
print("원자료:", raw.shape, raw["date"].min().date(), "~", raw["date"].max().date())

# ---------- 1. 결측치·품질 점검 ----------
print("\n=== 결측치 개수 ===")
na = raw.isna().sum()
print(pd.DataFrame({"결측치": na, "비율(%)": (na / len(raw) * 100).round(2)}))

full_range = pd.date_range(START, END, freq="D")
missing_dates = full_range.difference(raw["date"])
print("\n빠진 날짜 수:", len(missing_dates))
print("중복 날짜 수:", raw["date"].duplicated().sum())

num_cols = ["GPRD", "GPRD_ACT", "GPRD_THREAT", "GPRD_MA7", "GPRD_MA30"]
print("\n음수 값 개수:\n", (raw[num_cols] < 0).sum())
print("\n0인 값 개수:\n", (raw[num_cols] == 0).sum())

print("\n=== 기술통계 ===")
print(raw[num_cols].describe().round(2))

# 이상치 (IQR 1.5배 기준) - 표시만 하고 삭제하지 않음
q1, q3 = raw["GPRD"].quantile([0.25, 0.75])
upper = q3 + 1.5 * (q3 - q1)
outlier = raw["GPRD"] > upper
print(f"\nGPRD 이상치(> {upper:.1f}): {outlier.sum()}개")
print(raw.loc[outlier].nlargest(10, "GPRD")[["date", "GPRD", "event"]].to_string(index=False))

# ---------- 2. 전처리 ----------
gpr = raw[["date"] + num_cols].rename(columns={"date": "Date"})
gpr["weekday"] = gpr["Date"].dt.dayofweek < 5          # 월~금 True
gpr["GPRD_outlier"] = outlier.values                   # 이상치 표시

# (a) 전체 일자 (주말 포함)
gpr.to_csv(f"data/processed/GPR_daily_{TAG}_clean.csv", index=False, encoding="utf-8-sig")

# (b) 평일만 (주말 버림)
wk = gpr[gpr["weekday"]].drop(columns="weekday")
wk.to_csv(f"data/processed/GPR_weekday_{TAG}_clean.csv", index=False, encoding="utf-8-sig")

# (c) 주말 값을 월요일에 합친 평일 파일 (월요일 GPR = 토·일·월 평균)
g = gpr.set_index("Date")[["GPRD", "GPRD_ACT", "GPRD_THREAT"]]
# 토·일은 다음 월요일 날짜로 옮긴 뒤 날짜별 평균
shift = g.index.dayofweek.map({5: 2, 6: 1}).fillna(0).astype(int)
g.index = g.index + pd.to_timedelta(shift, unit="D")
wk_avg = g.groupby(level=0).mean()
wk_avg = wk_avg[wk_avg.index <= END].reset_index()    # 마지막 주말이 기간 밖 월요일로 넘어가는 것 방지
wk_avg.to_csv(f"data/processed/GPR_weekday_weekendavg_{TAG}.csv", index=False, encoding="utf-8-sig")

print("\n전체:", gpr.shape, " 평일만:", wk.shape, " 주말→월요일 평균:", wk_avg.shape)
