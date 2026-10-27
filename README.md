# GridSense — Load Prediction + Load Balance (Streamlit)

Modern dashboard built from your three notebooks and `SecureElite_DeviceID19_KW_gt_1.xlsx`.

| Page | What it does | Source notebook |
|---|---|---|
| Overview | KPIs, load timeline, latest phase split, typical day | — |
| Load Prediction | 1/5/15/30-min KW forecast; Extra Trees, Random Forest, HistGB, Ridge, GRU (NumPy), RLS online, SARIMA vs persistence & 20-min MA; leaderboard, residuals, next-step outlook | KW_Load_Forecasting_Drift_GRU_TL_SARIMA_noKW |
| Load Balance | Rolling ARIMA on R/Y/B kW → imbalance %, alerts (precision/recall), accuracy vs persistence, 95 % outlook | imbalance_arima_forecasting |
| Drift Monitor | PSI/KS before-vs-after a date, ADWIN level alarms | drift section |

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Host on Streamlit Community Cloud
1. Push this folder to a GitHub repo (keep `data/` and `.streamlit/`).
2. share.streamlit.io → New app → pick repo, main file `app.py` → Deploy.

## Notes
* Models train on first visit to a page/horizon (≈10–25 s) and are cached.
* Only ~9 % of 30-min bins in the file are real measurements; Load Balance scores on measured bins by default.
* Upload another meter file (same columns) from the sidebar → *Data source*.
* Dataset ends 17 Oct 2024, so "next forecast" is from the latest available reading.
