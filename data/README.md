# 데이터 받는 방법

원자료는 출처마다 재배포 조건이 달라 레포에 넣지 않았습니다. 아래에서 내려받아 `data/raw/`에 **표의 파일 이름 그대로** 저장하세요. 기간은 모두 2020-01-01 ~ 2025-12-31입니다.

| 파일 이름 | 출처 | 받는 곳 |
|---|---|---|
| `data_gpr_daily_recent.xls` | Caldara & Iacoviello, Daily GPR | https://www.matteoiacoviello.com/gpr.htm → "Recent GPR index - daily data" |
| `USDKRW_FRED_20200101_20251231.csv` | FRED `DEXKOUS` | https://fred.stlouisfed.org/series/DEXKOUS |
| `VIXCLS_(1).csv` | FRED `VIXCLS` | https://fred.stlouisfed.org/series/VIXCLS |
| `DCOILBRENTEU.csv` | FRED `DCOILBRENTEU` | https://fred.stlouisfed.org/series/DCOILBRENTEU |
| `DXY_ICE_2020102_20251230.csv` | Yahoo Finance `DX-Y.NYB` | 예: `yfinance.download("DX-Y.NYB", start="2020-01-01", end="2026-01-01").to_csv(...)` |

- FRED 파일은 빈칸이 `.`으로 들어 있어도 코드에서 처리합니다.
- 분석에 쓴 DXY 파일은 2025-12-30에서 끝나서 2025-12-31이 빠져 있습니다. `end`를 2026-01-01로 받으면 그날이 추가되어, 결과 숫자가 조금 달라질 수 있습니다.
- `data/processed/`는 `src/preprocess_gpr.py`와 `src/merge_preprocess.py`를 실행하면 만들어집니다.
