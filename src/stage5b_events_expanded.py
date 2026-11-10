# 5단계 확장: 규칙으로 사건을 늘린 이벤트 스터디 (기존 19개 결과와 비교)
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
TYPES = {"전쟁·군사충돌": ("#2a78d6", "o"), "중동 긴장": ("#eb6834", "s"),
         "북한": ("#1baf7a", "^"), "국내 정치": ("#4a3aa7", "D")}

# 규칙 (환율 결과를 보기 전에 고정)
# - 기존: 처음 고른 19개
# - GPR급등: 2020~2025 일별 GPR 상위 30개(서로 10일 이상 떨어진 날)에서 직전 사건을 찾아 추가
# - 북한규칙: ICBM 발사 전부 + 일본 상공 통과, NLL 이남 낙탄, 군사정찰위성, 포격, 남북 시설 폭파
# - 전쟁·군사충돌: 전쟁 개시·국가 간 대규모 공격, 러-우 전쟁 확대, 중동 밖 국가 간 충돌
# - 중동 긴장: 중동 지역 사건 중 위에 해당하지 않는 것
# (날짜, 사건, 유형, 어떻게 들어왔나, 지역)
EVENTS = [
    ("2022-02-24", "러시아 우크라이나 침공", "전쟁·군사충돌", "기존", "러-우"),
    ("2023-10-07", "하마스 이스라엘 공격", "전쟁·군사충돌", "기존", "중동"),
    ("2025-06-13", "이스라엘 이란 공습", "전쟁·군사충돌", "기존", "중동"),
    ("2025-06-22", "미국 이란 핵시설 공격", "전쟁·군사충돌", "기존", "중동"),
    ("2022-01-24", "미국 병력 8,500명 대기 명령", "전쟁·군사충돌", "GPR급등", "러-우"),
    ("2022-09-21", "러시아 부분 동원령", "전쟁·군사충돌", "GPR급등", "러-우"),
    ("2022-10-10", "러시아 우크라이나 전역 미사일 공격", "전쟁·군사충돌", "GPR급등", "러-우"),
    ("2024-11-17", "미국 ATACMS 러시아 공격 허용", "전쟁·군사충돌", "GPR급등", "러-우"),
    ("2025-03-03", "미국 우크라이나 군사 지원 중단", "전쟁·군사충돌", "GPR급등", "러-우"),
    ("2025-05-07", "인도 파키스탄 공습", "전쟁·군사충돌", "GPR급등", "기타"),
    ("2025-06-01", "우크라이나 스파이더웹 작전", "전쟁·군사충돌", "GPR급등", "러-우"),
    ("2025-09-10", "러시아 드론 폴란드 영공 침범", "전쟁·군사충돌", "GPR급등", "러-우"),
    ("2020-01-03", "미국 솔레이마니 제거", "중동 긴장", "기존", "중동"),
    ("2024-01-12", "미·영 후티 공습", "중동 긴장", "기존", "중동"),
    ("2024-04-13", "이란 이스라엘 첫 직접 공격", "중동 긴장", "기존", "중동"),
    ("2024-10-01", "이란 이스라엘 미사일 공격", "중동 긴장", "기존", "중동"),
    ("2023-10-17", "가자 알아흘리 병원 폭발", "중동 긴장", "GPR급등", "중동"),
    ("2023-10-31", "자발리아 난민촌 공습", "중동 긴장", "GPR급등", "중동"),
    ("2024-01-09", "후티 홍해 최대 규모 공격", "중동 긴장", "GPR급등", "중동"),
    ("2024-09-23", "이스라엘 레바논 대규모 공습", "중동 긴장", "GPR급등", "중동"),
    ("2024-11-27", "시리아 반군 대공세 시작", "중동 긴장", "GPR급등", "중동"),
    ("2024-12-08", "시리아 아사드 정권 붕괴", "중동 긴장", "GPR급등", "중동"),
    ("2025-03-18", "이스라엘 가자 공습 재개", "중동 긴장", "GPR급등", "중동"),
    ("2025-09-09", "이스라엘 카타르 도하 공습", "중동 긴장", "GPR급등", "중동"),
    ("2025-10-03", "트럼프 가자 최후통첩·하마스 응답", "중동 긴장", "GPR급등", "중동"),
    ("2020-06-16", "남북 공동연락사무소 폭파", "북한", "기존", "북한"),
    ("2022-03-24", "ICBM 발사 (2022-03-24)", "북한", "기존", "북한"),
    ("2022-10-04", "일본 상공 미사일 발사", "북한", "기존", "북한"),
    ("2022-11-02", "NLL 이남 미사일 발사", "북한", "기존", "북한"),
    ("2023-07-12", "ICBM 화성-18 발사 (2023-07)", "북한", "기존", "북한"),
    ("2024-01-05", "연평도 인근 포격", "북한", "기존", "북한"),
    ("2024-10-15", "경의선·동해선 폭파", "북한", "기존", "북한"),
    ("2022-03-16", "ICBM 발사 실패 (2022-03-16)", "북한", "북한규칙", "북한"),
    ("2022-05-25", "ICBM 발사 (2022-05)", "북한", "북한규칙", "북한"),
    ("2022-11-03", "ICBM 발사 (2022-11-03)", "북한", "북한규칙", "북한"),
    ("2022-11-18", "ICBM 화성-17 발사 (2022-11-18)", "북한", "북한규칙", "북한"),
    ("2023-02-18", "ICBM 화성-15 발사 (2023-02)", "북한", "북한규칙", "북한"),
    ("2023-03-16", "ICBM 화성-17 발사 (2023-03)", "북한", "북한규칙", "북한"),
    ("2023-04-13", "ICBM 화성-18 발사 (2023-04)", "북한", "북한규칙", "북한"),
    ("2023-11-21", "군사정찰위성 발사", "북한", "북한규칙", "북한"),
    ("2023-12-18", "ICBM 화성-18 발사 (2023-12)", "북한", "북한규칙", "북한"),
    ("2024-10-31", "ICBM 화성-19 발사", "북한", "북한규칙", "북한"),
    ("2024-12-03", "비상계엄 선포", "국내 정치", "기존", "한국"),
    ("2024-12-14", "대통령 탄핵소추안 가결", "국내 정치", "기존", "한국"),
    ("2024-12-27", "권한대행 탄핵소추안 가결", "국내 정치", "기존", "한국"),
    ("2025-04-04", "대통령 파면 결정", "국내 정치", "기존", "한국"),
]
# GPR 급등일 상위 30개를 어떻게 처리했는지
SPIKES = {
    "2020-01-07": "포함: 솔레이마니 제거(기존)", "2022-01-25": "포함: 미국 병력 대기 명령",
    "2022-02-15": "제외: 하나의 새 사건 확인 안 됨", "2022-03-01": "포함: 러-우 침공(기존)",
    "2022-03-16": "제외: 전쟁 지속 보도", "2022-03-28": "제외: 전쟁 지속 보도",
    "2022-04-14": "제외: 전쟁 지속 보도", "2022-04-25": "제외: 전쟁 지속 보도",
    "2022-09-22": "포함: 러시아 부분 동원령", "2022-10-11": "포함: 러시아 전역 미사일 공격",
    "2022-11-08": "제외: 사건 확인 안 됨", "2023-10-09": "포함: 하마스 공격(기존)",
    "2023-10-19": "포함: 알아흘리 병원 폭발", "2023-11-01": "포함: 자발리아 난민촌 공습",
    "2024-01-09": "포함: 후티 홍해 공격", "2024-04-16": "포함: 이란 직접 공격(기존)",
    "2024-09-26": "포함: 레바논 대규모 공습", "2024-11-18": "포함: ATACMS 허용",
    "2024-11-29": "포함: 시리아 반군 대공세", "2024-12-12": "포함: 아사드 정권 붕괴",
    "2025-02-20": "제외: 군사 충돌 사건 확인 안 됨", "2025-03-04": "포함: 미국 지원 중단",
    "2025-03-19": "포함: 가자 공습 재개", "2025-05-07": "포함: 인도 파키스탄 공습",
    "2025-05-19": "제외: 사건 확인 안 됨", "2025-06-02": "포함: 스파이더웹 작전",
    "2025-06-23": "포함: 미국 이란 공격(기존)", "2025-09-10": "포함: 도하 공습, 폴란드 영공 침범",
    "2025-10-03": "포함: 가자 최후통첩", "2025-11-03": "제외: 사건 확인 안 됨",
}
PRE, POST = 5, 10
OUT = "output"
os.makedirs(OUT, exist_ok=True)
pd.DataFrame([{"GPR 급등일": k, "처리": v} for k, v in SPIKES.items()]).to_csv(
    f"{OUT}/table11_gpr_spike_screening.csv", index=False, encoding="utf-8-sig")

c = pd.read_csv("data/processed/merged_2020_2025_clean.csv", parse_dates=["Date"])
c["r_DXY_lag1"] = c["r_DXY"].shift(1).fillna(0)
day0 = [int(c["Date"].searchsorted(pd.Timestamp(e[0]))) for e in EVENTS]

mask = np.ones(len(c), bool)
for i in day0:
    mask[max(i - PRE, 0):i + POST + 1] = False
m = smf.ols("r_USD_KRW ~ r_DXY + r_DXY_lag1", c[mask]).fit()
c["ar"] = c["r_USD_KRW"] - m.predict(c)
sigma = c.loc[mask, "ar"].std()
print("평소 날짜 수:", mask.sum(), "달러 관계:", m.params.round(3).to_dict(), "R2", round(m.rsquared, 3), "σ", round(sigma, 3))


def cum(col, i, a, b):
    lo, hi = max(i + a, 0), min(i + b, len(c) - 1)
    return c[col].iloc[lo:hi + 1].sum()


rows, paths = [], []
for e, i in zip(EVENTS, day0):
    base = c["GPR"].iloc[max(i - 20, 0):i].mean() if i >= 5 else np.nan
    car1, car5 = cum("ar", i, 0, 1), cum("ar", i, 0, 5)
    rows.append({"사건일": e[0], "사건": e[1], "유형": e[2], "출처": e[3], "지역": e[4], "i": i, "0일(거래일)": c["Date"].iloc[i].date(),
                 "환율 0~1일%": cum("r_USD_KRW", i, 0, 1), "환율 0~5일%": cum("r_USD_KRW", i, 0, 5),
                 "달러제거 0~1일%": car1, "달러제거 0~5일%": car5, "달러제거 0~10일%": cum("ar", i, 0, 10),
                 "t값(0~1일)": car1 / (sigma * np.sqrt(2)), "t값(0~5일)": car5 / (sigma * np.sqrt(6)),
                 "GPR 배수": c["GPR"].iloc[i:i + 3].max() / base})
    if i >= PRE:
        a = c["ar"].iloc[i - PRE:i + POST + 1].to_numpy()
        paths.append((e[2], a.cumsum() - a[:PRE].sum()))
ev = pd.DataFrame(rows).sort_values(["유형", "사건일"])
ev.drop(columns="i").round(6).to_csv(f"{OUT}/table9_events_expanded.csv", index=False, encoding="utf-8-sig")
print(ev[["유형", "0일(거래일)", "사건", "출처", "GPR 배수", "달러제거 0~1일%", "달러제거 0~5일%", "t값(0~5일)"]].round(2).to_string(index=False))

rng = np.random.default_rng(42)
ok = np.arange(PRE, len(c) - POST - 1)
ok = ok[mask[ok]]
csum = np.r_[0, c["ar"].cumsum().to_numpy()]


def place(mean, n, b, trials=20000):
    """사건 없는 날짜를 n개씩 뽑은 평균과 비교한 양쪽 p값"""
    idx = rng.choice(ok, size=(trials, n))
    pl = (csum[idx + b + 1] - csum[idx]).mean(axis=1)
    return (np.abs(pl - pl.mean()) >= abs(mean - pl.mean())).mean()


def thin(g, gap=POST + 1):
    """분석 구간이 겹치지 않게 앞에서부터 사건을 남김"""
    keep, last = [], -10**9
    for _, r in g.sort_values("i").iterrows():
        if r["i"] - last >= gap:
            keep.append(r)
            last = r["i"]
    return pd.DataFrame(keep)


def summarize(g, label):
    return {"구분": label, "사건 수": len(g), "GPR 배수(중앙값)": g["GPR 배수"].median(),
            "0~1일%": g["달러제거 0~1일%"].mean(), "0~1일 p": place(g["달러제거 0~1일%"].mean(), len(g), 1),
            "0~5일%": g["달러제거 0~5일%"].mean(), "0~5일 p": place(g["달러제거 0~5일%"].mean(), len(g), 5),
            "0~10일%": g["달러제거 0~10일%"].mean(), "0~10일 p": place(g["달러제거 0~10일%"].mean(), len(g), 10),
            "0~1일 상승 비율": (g["달러제거 0~1일%"] > 0).mean(), "0~5일 상승 비율": (g["달러제거 0~5일%"] > 0).mean()}


gr = pd.DataFrame([summarize(ev[ev["유형"] == t], t) for t in TYPES])
# 유형 4개를 함께 검정했으니 Holm 보정 (주지표 0~1일)
o = gr["0~1일 p"].sort_values()
holm, run = {}, 0
for k, (idx, p) in enumerate(o.items()):
    run = max(run, min(1, p * (len(o) - k)))
    holm[idx] = run
gr["0~1일 보정 p"] = pd.Series(holm)
gr.round(6).to_csv(f"{OUT}/table10_event_type_expanded.csv", index=False, encoding="utf-8-sig")
print("\n=== 유형별 ===\n", gr.round(3).T.to_string())

chk = []
for t in TYPES:
    chk.append(summarize(thin(ev[ev["유형"] == t]), f"{t} (구간 겹침 제거)"))
for t in ["전쟁·군사충돌", "중동 긴장", "북한"]:
    chk.append(summarize(ev[(ev["유형"] == t) & (ev["출처"] != "기존")], f"{t} (새로 추가한 사건만)"))
for r in ["러-우", "중동"]:
    chk.append(summarize(ev[ev["지역"] == r], f"지역: {r} 전체"))
chk.append(summarize(ev[(ev["유형"] == "북한") & ev["사건"].str.contains("ICBM")], "북한 ICBM만"))
chk.append(summarize(ev[(ev["유형"] == "북한") & (ev["사건일"].str[:4] != "2022")], "북한 2022년 제외"))
chk = pd.DataFrame(chk)
chk.round(6).to_csv(f"{OUT}/table12_event_robustness.csv", index=False, encoding="utf-8-sig")
print("\n=== 확인 ===\n", chk.round(3).to_string(index=False))
v = ev.dropna(subset=["GPR 배수"])
print("\nGPR 배수-0~1일 상관:", round(v["GPR 배수"].corr(v["달러제거 0~1일%"]), 3), "n", len(v),
      " 스피어만:", round(v["GPR 배수"].corr(v["달러제거 0~1일%"], method="spearman"), 3))


def clean(ax, gridaxis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis=gridaxis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


days = np.arange(-PRE, POST + 1)
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), sharex=True, sharey=True)
for ax, (t, (col, _)) in zip(axes.ravel(), TYPES.items()):
    ps = [p[1] for p in paths if p[0] == t]
    for a in ps:
        ax.plot(days, a, color=LIGHT, linewidth=0.8)
    ax.plot(days, np.mean(ps, axis=0), color=col, linewidth=2.4)
    ax.axvline(0, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_title(f"{t} (사건 {len(ps)}개)", loc="left", fontsize=11, color=INK)
    clean(ax)
for ax in axes[1]:
    ax.set_xlabel("사건일로부터 거래일", fontsize=9, color=INK2)
for ax in axes[:, 0]:
    ax.set_ylabel("누적 환율 변화 (%, 달러 효과 제거)", fontsize=9, color=INK2)
fig.suptitle("사건을 늘린 뒤 유형별 환율 반응: 굵은 선 = 평균, 회색 선 = 개별 사건 (0 = 사건 전날)",
             x=0.01, ha="left", fontsize=13, color=INK)
fig.tight_layout()
fig.savefig(f"{OUT}/fig8_event_paths_expanded.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)

# 유형별 분포: 점 = 사건, 막대 = 평균
fig, ax = plt.subplots(figsize=(9, 4.6))
names = list(TYPES)[::-1]
for y, t in enumerate(names):
    g = ev[ev["유형"] == t]
    col, mk = TYPES[t]
    jit = np.linspace(-0.18, 0.18, len(g))
    ax.scatter(g["달러제거 0~1일%"], y + jit, color=col, marker=mk, s=38, alpha=0.75, edgecolor="white", linewidth=0.8, zorder=3)
    mean = g["달러제거 0~1일%"].mean()
    ax.plot([mean, mean], [y - 0.32, y + 0.32], color=INK, linewidth=2.2, zorder=4)
    ax.text(ax.get_xlim()[1], y, "", fontsize=9)
    ax.annotate(f"평균 {mean:+.2f}%", (mean, y + 0.34), ha="center", va="bottom", fontsize=9, color=INK)
ax.axvline(0, color=INK2, linewidth=0.9)
ax.set_yticks(range(len(names)), [f"{t} ({(ev['유형'] == t).sum()}개)" for t in names], fontsize=10, color=INK)
ax.set_ylim(-0.6, len(names) - 0.3)
clean(ax, "x")
ax.spines["left"].set_visible(False)
ax.set_xlabel("사건 후 0~1거래일 환율 변화 (%, 달러 효과 제거, +면 원화 약세)", fontsize=9, color=INK2)
ax.set_title("사건별 환율 반응과 유형 평균 (점 = 사건 하나, 검은 선 = 평균)", loc="left", fontsize=12, color=INK)
fig.tight_layout()
fig.savefig(f"{OUT}/fig9_event_dist_expanded.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)

# GPR 반응 vs 환율 반응
fig, ax = plt.subplots(figsize=(9, 5.6))
for t, (col, mk) in TYPES.items():
    g = v[v["유형"] == t]
    ax.scatter(g["GPR 배수"], g["달러제거 0~1일%"], color=col, marker=mk, s=60, edgecolor="white", linewidth=1.2, label=t, zorder=3)
ax.axhline(0, color=INK2, linewidth=0.8)
ax.axvline(1, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
clean(ax)
ax.grid(axis="x", color=GRID, linewidth=0.6)
ax.set_xlabel("GPR 반응 (사건 직후 최고값 ÷ 이전 20일 평균)", fontsize=9, color=INK2)
ax.set_ylabel("사건 후 0~1일 환율 변화 (%, 달러 효과 제거)", fontsize=9, color=INK2)
ax.set_title("사건을 늘려도 GPR이 크게 뛴 사건일수록 더 오르지는 않았다", loc="left", fontsize=12, color=INK, pad=28)
ax.legend(frameon=False, fontsize=9, ncol=4, loc="lower left", bbox_to_anchor=(-0.01, 1.0))
fig.tight_layout()
fig.savefig(f"{OUT}/fig10_gpr_vs_fx_expanded.png", dpi=150, bbox_inches="tight", facecolor="white")
