import numpy as np, pandas as pd, plotly.graph_objects as go
import streamlit as st
from core import data, ui, forecast as fc

full, d = data.get()
ui.hero("Load Prediction", "Forecast the feeder's active power (kW). Models see electrical signals only — KW and its phase parts are removed from the inputs — and are scored on the newest 15 % of data.",
        ["train 70 % · val 15 % · test 15 %", "20-min input window", "KW is target only"])
c = st.columns([1.1, 2.4])
with c[0]: h = st.segmented_control("Forecast horizon", [1, 5, 15, 30], default=1, format_func=lambda x: f"{x} min", selection_mode="single") or 1
with c[1]: chosen = st.multiselect("Models to train & compare", fc.ALL_MODELS[2:], default=["Extra Trees", "GRU", "RLS (online)", "SARIMA(2,0,1)"], help="Baselines (persistence, 20-min moving average) are always included. SARIMA runs for the 1-minute horizon only.")
if not chosen: st.info("Pick at least one model."); st.stop()
if h != 1: chosen = [m for m in chosen if not m.startswith("SARIMA")]
R = fc.backtest(d, data.key(d), h, tuple(chosen)); M = R["M"]; ml = M.drop(index=fc.BASELINES, errors="ignore"); best = ml.index[0]
b = M.loc[best]; ma = M.loc["Moving avg (20 min)"]
k = st.columns(4)
nx = R["next"][best]
with k[0]: ui.kpi(f"Next {h}-min forecast", f"{nx:,.0f} kW", f"{best} · from {R['last_t']:%d %b %H:%M} (last reading {R['last_kw']:,.0f})", "info")
with k[1]: ui.kpi("Best model", best, f"MAE {b.MAE:.1f} kW", "good")
with k[2]: ui.kpi("Test RMSE · R²", f"{b.RMSE:.1f} · {b.R2:.2f}", f"MA-20 baseline: {ma.RMSE:.1f} · {ma.R2:.2f}")
with k[3]: ui.kpi("vs 20-min moving avg", f"{b['vs_MA20_%']:+.1f} %", "RMSE improvement" if b["vs_MA20_%"] > 0 else "no model beats the baseline", "good" if b["vs_MA20_%"] > 0 else "warn")
st.write("")
t1, t2, t3, t4 = st.tabs(["Forecast vs actual", "Leaderboard", "Error analysis", "Next-step outlook"])
with t1:
    n = len(R["y"]); show = st.slider("Test minutes shown", 60, n, min(400, n), step=20)
    sel = st.multiselect("Show models", list(R["P"]), default=[best, "Moving avg (20 min)"])
    t = R["t"][:show]; f = go.Figure(); tt, ya = data.gapify(t, R["y"][:show]); f.add_scatter(x=tt, y=ya, name="Actual", line=dict(color="#E6EAF2", width=1.6), connectgaps=False)
    for i, m in enumerate(sel):
        tt, pa = data.gapify(t, R["P"][m][:show]); f.add_scatter(x=tt, y=pa, name=m, line=dict(width=2, dash="dash" if "Moving" in m or m == "Persistence" else "solid", color=ui.COLORWAY[i % 8]), connectgaps=False)
    f.update_layout(title=f"Actual vs predicted KW — {h} min ahead", yaxis_title="kW"); f.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(ui.style(f, 470), width="stretch")
    ui.note("Forecasts sit <i>between</i> the HIGH and LOW levels on purpose: when the next dip's timing is uncertain, the probability-weighted average minimises squared error. Perfect dip timing would lift R² to ≈ 0.94 — timing is the whole problem.")
with t2:
    cols = ["#6B778F" if i in fc.BASELINES else ui.AMBER if i == best else ui.CYAN for i in M.index[::-1]]
    f = go.Figure(go.Bar(y=M.index[::-1], x=M.RMSE[::-1], orientation="h", marker_color=cols, text=[f"{v:.1f}" for v in M.RMSE[::-1]], textposition="outside"))
    f.update_layout(title="Test RMSE in kW (lower is better · grey = baselines)", xaxis_range=[0, M.RMSE.max() * 1.15], hovermode="closest"); st.plotly_chart(ui.style(f, 380, legend=False), width="stretch")
    st.dataframe(M.round(2).rename(columns={"MAPE": "MAPE %"}), width="stretch")
    st.caption(f"{R['n_train']:,} training samples · {R['n_test']:,} test samples. Differences of 1–2 kW are within noise on a test set this small.")
with t3:
    m = st.selectbox("Model", list(R["P"]), index=list(R["P"]).index(best)); e = R["P"][m] - R["y"]
    a, b2 = st.columns(2)
    f = go.Figure(go.Histogram(x=e, nbinsx=50, marker_color=ui.VIOLET)); f.add_vline(x=0, line_dash="dash", line_color="#8892A6"); f.update_layout(title=f"Residuals (pred − actual) · bias {e.mean():+.1f} kW", hovermode="closest")
    a.plotly_chart(ui.style(f, 360, legend=False), width="stretch")
    f = go.Figure(go.Scatter(x=R["y"], y=R["P"][m], mode="markers", marker=dict(size=4, color=ui.CYAN, opacity=.5)))
    lo, hi = R["y"].min(), R["y"].max(); f.add_scatter(x=[lo, hi], y=[lo, hi], line=dict(color=ui.AMBER, dash="dash"), name="perfect"); f.update_layout(title="Predicted vs actual", xaxis_title="actual kW", yaxis_title="predicted kW", hovermode="closest")
    b2.plotly_chart(ui.style(f, 360, legend=False), width="stretch")
with t4:
    nxt = pd.Series(R["next"]).rename("forecast kW").to_frame(); nxt["Δ vs last reading"] = nxt["forecast kW"] - R["last_kw"]
    st.write(f"Forecast for **{R['last_t'] + pd.Timedelta(minutes=h):%d %b %Y %H:%M}** (t + {h} min), using the 20 minutes up to the last reading in the data.")
    f = go.Figure(go.Bar(x=nxt.index, y=nxt["forecast kW"], marker_color=ui.CYAN, text=[f"{v:.0f}" for v in nxt["forecast kW"]], textposition="outside"))
    f.add_hline(y=R["last_kw"], line_dash="dash", line_color=ui.AMBER, annotation_text="last reading"); f.update_layout(yaxis_range=[nxt["forecast kW"].min() * .85, nxt["forecast kW"].max() * 1.08], hovermode="closest")
    st.plotly_chart(ui.style(f, 360, legend=False), width="stretch"); st.dataframe(nxt.round(1), width="stretch")
    ui.note("The dataset ends on 17 Oct 2024, so this is the forecast from the <i>latest available</i> reading. Connect a live feed and re-run for true real-time forecasts.")
