# 원/달러 환율과 브렌트유 가격 비교 그림 (위아래 두 칸, 일별 원자료)
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 레포 최상위에서 실행되도록
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
TEAL, ORANGE, INK, INK2, GRID = "#2BB0BC", "#D47636", "#262626", "#56606C", "#DCE3E8"

fx = pd.read_csv("data/raw/USDKRW_FRED_20200101_20251231.csv", parse_dates=["DATE"], na_values=[".", ""]).dropna()
br = pd.read_csv("data/raw/DCOILBRENTEU.csv", parse_dates=["observation_date"], na_values=[".", ""]).dropna()
br = br[br["observation_date"] <= "2025-12-31"]

fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)
panels = [(axes[0], fx["DATE"], fx["DEXKOUS"], TEAL, "원/달러 환율 (원)"),
          (axes[1], br["observation_date"], br["DCOILBRENTEU"], ORANGE, "브렌트유 가격 (달러/배럴)")]
for ax, x, y, color, title in panels:
    ax.plot(x, y, color=color, linewidth=1.2)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)

# 눈에 띄는 지점만 표시
def note(ax, date, value, text, dy, dx=0):
    ax.annotate(text, (pd.Timestamp(date), value), xytext=(dx, dy), textcoords="offset points",
                ha="center" if dx == 0 else "left", va="center" if dx else "baseline",
                fontsize=9, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))

note(axes[0], "2021-01-04", 1081.62, "최저 1,082원 (2021.01)", -26)
note(axes[0], "2025-12-23", 1481.45, "최고 1,481원 (2025.12)", 12)
note(axes[1], "2020-04-21", 9.12, "코로나 폭락 9달러 (2020.04)", 0, 40)
note(axes[1], "2022-03-08", 133.18, "러-우 전쟁 133달러 (2022.03)", 8)

fig.suptitle("2022년 상반기엔 함께 올랐지만, 그 뒤로 유가는 내리고 환율은 올랐다",
             x=0.01, ha="left", fontsize=14, color=INK, y=1.0)
fig.text(0.01, -0.01, "자료: FRED DEXKOUS(뉴욕 정오 기준), DCOILBRENTEU (일별, 2020-01-02 ~ 2025-12-31)",
         fontsize=8.5, color=INK2)
fig.tight_layout()
fig.savefig("output/fig11_fx_brent_timeseries.png", dpi=200, bbox_inches="tight", facecolor="white")
print("max brent", br.loc[br["DCOILBRENTEU"].idxmax()].tolist(), "min", br.loc[br["DCOILBRENTEU"].idxmin()].tolist())
