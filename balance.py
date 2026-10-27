"""Phase load-balance forecasting: rolling ARIMA on R/Y/B phase kW -> imbalance % -> alerts.
Logic follows imbalance_arima_forecasting notebook."""
import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

FEATS = ["RKW", "YKW", "BKW"]
HORIZONS = {"30 min": 1, "1 hour": 2, "3 hours": 6, "6 hours": 12}      # in 30-min steps (scaled by freq below)


def imbalance_pct(frame):
    """Max phase deviation from the 3-phase mean, in % of the mean (same definition as the notebook)."""
    mean = frame[FEATS].mean(axis=1).replace(0, np.nan)
    return frame[FEATS].sub(mean, axis=0).abs().max(axis=1) / mean * 100


def status(pct, thr):
    return ("good", "Balanced") if pct < thr * .6 else (("warn", "Watch") if pct < thr else ("bad", "Imbalanced"))


def prep(full, freq):
    x = full.set_index("Timestamps")[FEATS].resample(freq).mean()
    measured = x.notna().all(axis=1)
    return x.interpolate(limit_direction="both"), measured


def _rolling(series, order, n_roll, max_steps):
    fit_end = len(series) - n_roll
    res = ARIMA(series.iloc[:fit_end], order=order).fit(); out = np.full((n_roll, max_steps), np.nan)
    for i in range(n_roll):
        out[i] = res.forecast(max_steps).values
        res = res.append(series.iloc[[fit_end + i]], refit=False)
    return out


@st.cache_data(show_spinner="Running rolling ARIMA on each phase…")
def run(_full, dkey, freq, order, test_hours, thr, measured_only):
    step_min = pd.Timedelta(freq).total_seconds() / 60
    hz = {k: max(1, int(round(v * 30 / step_min))) for k, v in HORIZONS.items()}
    data, measured = prep(_full, freq)
    test_steps = int(test_hours * 60 / step_min); max_steps = max(hz.values()); warm = max_steps - 1; n_roll = test_steps + warm
    test = data.iloc[-test_steps:]; mask = measured.iloc[-test_steps:].values
    raw = {f: _rolling(data[f], order, n_roll, max_steps) for f in FEATS}
    preds = {"ARIMA": {}, "Persistence": {}}
    for name, s in hz.items():
        origins = warm + np.arange(test_steps) - s + 1
        preds["ARIMA"][name] = pd.DataFrame({f: raw[f][origins, s - 1] for f in FEATS}, index=test.index)
        preds["Persistence"][name] = pd.DataFrame({f: data[f].shift(s).loc[test.index] for f in FEATS}, index=test.index)
    use = mask if (measured_only and mask.sum() >= 8) else np.ones(len(test), bool)
    act_imb = imbalance_pct(test); act_flag = act_imb > thr
    rows, det = [], []
    for model, by_h in preds.items():
        for name, fr in by_h.items():
            for f in FEATS:
                e = fr[f].values[use] - test[f].values[use]; yv = test[f].values[use]
                rows.append(dict(model=model, horizon=name, phase=f, MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()), R2=1 - (e ** 2).sum() / max(((yv - yv.mean()) ** 2).sum(), 1e-9)))
            imb = imbalance_pct(fr); flag = (imb > thr).values; a = act_flag.values; u = use
            tp = int((flag & a & u).sum()); pf = int((flag & u).sum()); af = int((a & u).sum())
            det.append(dict(model=model, horizon=name, mean_imbalance=imb.values[u].mean(), flagged=pf, actual_flagged=af, hits=tp,
                            precision=tp / pf if pf else np.nan, recall=tp / af if af else np.nan, MAE_pct=np.abs(imb.values[u] - act_imb.values[u]).mean()))
    return dict(test=test, measured=mask, used=use, preds=preds, metrics=pd.DataFrame(rows), det=pd.DataFrame(det), act_imb=act_imb, hz=hz,
                share_measured=float(measured.mean()), step_min=step_min)


@st.cache_data(show_spinner="Forecasting ahead…")
def outlook(_full, dkey, freq, order, steps):
    data, _ = prep(_full, freq); out = {}
    for f in FEATS:
        fc = ARIMA(data[f], order=order).fit().get_forecast(steps); out[f] = fc.predicted_mean; out[f + "_lo"] = fc.conf_int().iloc[:, 0]; out[f + "_hi"] = fc.conf_int().iloc[:, 1]
    o = pd.DataFrame(out); o["imbalance"] = imbalance_pct(o); return o, data
