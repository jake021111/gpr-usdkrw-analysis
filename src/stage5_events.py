# 5단계: 사건 유형별 비교 (이벤트 스터디)
# 사전 설치: pip install pandas numpy matplotlib statsmodels
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 레포 최상위에서 실행되도록
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
INK, INK2, GRID, LIGHT = "#0b0b0b", "#52514e", "#e4e3df", "#c9c8c2"
# 유형 색과 모양 (색만으로 구분하지 않도록 모양도 다르게)
TYPES = {"전쟁·군사충돌": ("#2a78d6", "o"), "중동 긴장": ("#eb6834", "s"),
         "북한": ("#1baf7a", "^"), "국내 정치": ("#4a3aa7", "D")}

# 날짜는 사건 발생일(현지 기준). 0일 = 그 날짜 이후 첫 거래일
EVENTS = [
    ("2022-02-24", "러시아 우크라이나 침공", "전쟁·군사충돌"),
    ("2023-10-07", "하마스 이스라엘 공격", "전쟁·군사충돌"),
    ("2025-06-13", "이스라엘 이란 공습", "전쟁·군사충돌"),
    ("2025-06-22", "미국 이란 핵시설 공격", "전쟁·군사충돌"),
    ("2020-01-03", "미국 솔레이마니 제거", "중동 긴장"),
    ("2024-01-12", "미·영 후티 공습", "중동 긴장"),
    ("2024-04-13", "이란 이스라엘 첫 직접 공격", "중동 긴장"),
    ("2024-10-01", "이란 이스라엘 미사일 공격", "중동 긴장"),
    ("2020-06-16", "남북 공동연락사무소 폭파", "북한"),
    ("2022-03-24", "화성-17 ICBM 발사", "북한"),
    ("2022-10-04", "일본 상공 미사일 발사", "북한"),
    ("2022-11-02", "NLL 이남 미사일 발사", "북한"),
    ("2023-07-12", "화성-18 ICBM 발사", "북한"),
    ("2024-01-05", "연평도 인근 포격", "북한"),
    ("2024-10-15", "경의선·동해선 폭파", "북한"),
    ("2024-12-03", "비상계엄 선포", "국내 정치"),
    ("2024-12-14", "대통령 탄핵소추안 가결", "국내 정치"),
    ("2024-12-27", "권한대행 탄핵소추안 가결", "국내 정치"),
    ("2025-04-04", "대통령 파면 결정", "국내 정치"),
]
PRE, POST = 5, 10
OUT = "output"
os.makedirs(OUT, exist_ok=True)

c = pd.read_csv("data/processed/merged_2020_2025_clean.csv", parse_dates=["Date"])
c["r_DXY_lag1"] = c["r_DXY"].shift(1).fillna(0)
day0 = [int(c["Date"].searchsorted(pd.Timestamp(d))) for d, _, _ in EVENTS]

# 달러 효과 제거: 사건 구간 밖 날짜로 '평소 관계'를 추정해서 뺀 값 = 초과 변화율
mask = np.ones(len(c), bool)
for i in day0:
    mask[max(i - PRE, 0):i + POST + 1] = False
m = smf.ols("r_USD_KRW ~ r_DXY + r_DXY_lag1", c[mask]).fit()
c["ar"] = c["r_USD_KRW"] - m.predict(c)
sigma = c.loc[mask, "ar"].std()
print("달러 관계:", m.params.round(3).to_dict(), "R2", round(m.rsquared, 3), "평소 하루 초과변동 표준편차", round(sigma, 3))


def cum(col, i, a, b):
    lo, hi = max(i + a, 0), min(i + b, len(c) - 1)
    return c[col].iloc[lo:hi + 1].sum()


rows, paths = [], []
for (d, name, typ), i in zip(EVENTS, day0):
    base = c["GPR"].iloc[max(i - 20, 0):i].mean() if i >= 5 else np.nan  # 이전 자료가 부족하면 계산 안 함
    peak = c["GPR"].iloc[i:i + 3].max()
    car5 = cum("ar", i, 0, 5)
    rows.append({"사건일": d, "사건": name, "유형": typ, "0일(거래일)": c["Date"].iloc[i].date(),
                 "사건전5일 변화%": cum("r_USD_KRW", i, -PRE, -1) if i >= PRE else np.nan,
                 "환율 0~1일%": cum("r_USD_KRW", i, 0, 1), "환율 0~5일%": cum("r_USD_KRW", i, 0, 5),
                 "환율 0~10일%": cum("r_USD_KRW", i, 0, 10),
                 "달러제거 0~1일%": cum("ar", i, 0, 1), "달러제거 0~5일%": car5,
                 "달러제거 0~10일%": cum("ar", i, 0, 10),
                 "t값(0~5일)": car5 / (sigma * np.sqrt(6)),
                 "GPR 이전20일평균": base, "GPR 0~2일최고": peak, "GPR 배수": peak / base})
    if i >= PRE:
        a = c["ar"].iloc[i - PRE:i + POST + 1].to_numpy()
        paths.append((name, typ, a.cumsum() - a[:PRE].sum()))    # 사건 전날 끝 = 0
ev = pd.DataFrame(rows)
ev.round(6).to_csv(f"{OUT}/table7_event_list_results.csv", index=False, encoding="utf-8-sig")
print(ev.drop(columns=["사건일"]).round(2).to_string(index=False))

# 유형별 평균 + 무작위 날짜 비교 (같은 개수의 날짜를 뽑아 평균이 이만큼 나올 확률)
rng = np.random.default_rng(42)
ok = np.arange(PRE, len(c) - POST - 1)
ok = ok[mask[ok]]                                     # 사건 구간 밖 날짜만
csum = {k: np.r_[0, c[k].cumsum().to_numpy()] for k in ["r_USD_KRW", "ar"]}


def place(col, n, b, trials=20000):
    idx = rng.choice(ok, size=(trials, n))
    return (csum[col][idx + b + 1] - csum[col][idx]).mean(axis=1)


out = []
for typ in TYPES:
    g = ev[ev["유형"] == typ]
    row = {"유형": typ, "사건 수": len(g)}
    for key, col, b in [("환율 0~1일%", "r_USD_KRW", 1), ("환율 0~5일%", "r_USD_KRW", 5),
                        ("달러제거 0~1일%", "ar", 1), ("달러제거 0~5일%", "ar", 5),
                        ("달러제거 0~10일%", "ar", 10)]:
        mean = g[key].mean()
        pl = place(col, len(g), b)
        row[key] = mean
        row[key + " p"] = (np.abs(pl - pl.mean()) >= abs(mean - pl.mean())).mean()
    row["GPR 배수(중앙값)"] = g["GPR 배수"].median()
    row["오른 사건 수(달러제거 0~5일)"] = int((g["달러제거 0~5일%"] > 0).sum())
    out.append(row)
gr = pd.DataFrame(out)
gr.round(6).to_csv(f"{OUT}/table8_event_type_summary.csv", index=False, encoding="utf-8-sig")
print("\n", gr.round(3).T.to_string())
print("\nGPR 배수와 달러제거 0~1일 상관:", round(ev["GPR 배수"].corr(ev["달러제거 0~1일%"]), 3))


def clean(ax, gridaxis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis=gridaxis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# 그림 1: 유형별 환율 반응 곡선 (달러 효과 제거)
days = np.arange(-PRE, POST + 1)
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), sharex=True, sharey=True)
for ax, (typ, (col, _)) in zip(axes.ravel(), TYPES.items()):
    ps = [p[2] for p in paths if p[1] == typ]
    for a in ps:
        ax.plot(days, a, color=LIGHT, linewidth=1)
    ax.plot(days, np.mean(ps, axis=0), color=col, linewidth=2.4)
    ax.axvline(0, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_title(f"{typ} (사건 {len(ps)}개)", loc="left", fontsize=11, color=INK)
    clean(ax)
for ax in axes[1]:
    ax.set_xlabel("사건일로부터 거래일", fontsize=9, color=INK2)
for ax in axes[:, 0]:
    ax.set_ylabel("누적 환율 변화 (%, 달러 효과 제거)", fontsize=9, color=INK2)
fig.suptitle("사건 유형별 환율 반응: 굵은 선 = 평균, 회색 선 = 개별 사건 (0 = 사건 전날)",
             x=0.01, ha="left", fontsize=13, color=INK)
fig.tight_layout()
fig.savefig(f"{OUT}/fig5_event_paths.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)

# 그림 2: 사건별 막대 (달러 제거 0~5일)
order = {t: k for k, t in enumerate(list(TYPES)[::-1])}
s = ev.assign(_o=ev["유형"].map(order)).sort_values(["_o", "달러제거 0~5일%"]).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(9.5, 7))
ax.barh(s.index, s["달러제거 0~5일%"], color=[TYPES[t][0] for t in s["유형"]], height=0.7,
        edgecolor="white", linewidth=1.5)
ax.set_yticks(s.index, [f"{n} ({d[:7]})" for n, d in zip(s["사건"], s["사건일"])], fontsize=9, color=INK)
for y, x, t in zip(s.index, s["달러제거 0~5일%"], s["t값(0~5일)"]):
    ax.text(x + (0.04 if x >= 0 else -0.04), y, f"{x:+.2f}%" + (" *" if abs(t) >= 1.96 else ""),
            va="center", ha="left" if x >= 0 else "right", fontsize=8.5, color=INK)
ax.axvline(0, color=INK2, linewidth=0.9)
ax.margins(x=0.14)
clean(ax, "x")
ax.spines["left"].set_visible(False)
ax.set_xlabel("사건 후 0~5거래일 누적 환율 변화 (%, 달러 효과 제거, +면 원화 약세)", fontsize=9, color=INK2)
ax.set_title("사건별 환율 반응 (* = 평소 변동 범위를 벗어남)", loc="left", fontsize=12, color=INK)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=v[0], label=k) for k, v in TYPES.items()],
          frameon=False, fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(f"{OUT}/fig6_event_bars.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)

# 그림 3: GPR 반응 vs 환율 반응
fig, ax = plt.subplots(figsize=(9, 5.8))
v = ev.dropna(subset=["GPR 배수"])
for typ, (col, mk) in TYPES.items():
    g = v[v["유형"] == typ]
    ax.scatter(g["GPR 배수"], g["달러제거 0~1일%"], color=col, marker=mk, s=70, edgecolor="white",
               linewidth=1.5, label=typ, zorder=3)
LABELS = {"권한대행 탄핵소추안 가결": (-4, -15), "대통령 파면 결정": (8, 3), "미·영 후티 공습": (8, 3),
          "이란 이스라엘 첫 직접 공격": (-60, 9), "경의선·동해선 폭파": (8, 3), "일본 상공 미사일 발사": (8, 3),
          "하마스 이스라엘 공격": (-40, -15), "러시아 우크라이나 침공": (-52, 9), "비상계엄 선포": (8, 3)}
for _, r in v.iterrows():
    if r["사건"] in LABELS:
        ax.annotate(r["사건"], (r["GPR 배수"], r["달러제거 0~1일%"]), xytext=LABELS[r["사건"]],
                    textcoords="offset points", fontsize=8.5, color=INK)
ax.axhline(0, color=INK2, linewidth=0.8)
ax.axvline(1, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
clean(ax)
ax.grid(axis="x", color=GRID, linewidth=0.6)
ax.set_xlabel("GPR 반응 (사건 직후 최고값 ÷ 이전 20일 평균)", fontsize=9, color=INK2)
ax.set_ylabel("사건 후 0~1일 환율 변화 (%, 달러 효과 제거)", fontsize=9, color=INK2)
ax.set_title("세계 GPR이 크게 뛴 사건일수록 환율도 더 올랐을까?", loc="left", fontsize=12, color=INK, pad=28)
ax.legend(frameon=False, fontsize=9, ncol=4, loc="lower left", bbox_to_anchor=(-0.01, 1.0))
fig.tight_layout()
fig.savefig(f"{OUT}/fig7_gpr_vs_fx.png", dpi=150, bbox_inches="tight", facecolor="white")
