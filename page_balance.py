import numpy as np, pandas as pd, plotly.graph_objects as go
import streamlit as st
import data, ui
import balance as bal

full, d = data.get()
ui.hero("Load Balance", "Forecast each phase (R, Y, B), derive the imbalance % — max phase deviation from the 3-phase mean — and flag windows that cross your alert threshold.",
        ["rolling ARIMA per phase", "imbalance = max|phase − mean| / mean", "alert threshold adjustable"])
with st.expander("Settings", expanded=True):
    c = st.columns(5)
    freq = c[0].selectbox("Smoothing", ["30min", "15min", "1h"], index=0)
    p, dd, q = c[1].number_input("AR p", 0, 5, 1), c[2].number_input("Diff d", 0, 2, 1), c[3].number_input("MA q", 0, 5, 1)
    thr = c[4].slider("Alert threshold %", 1.0, 15.0, 5.0, .5)
    c = st.columns(3)
    hrs = c[0].slider("Evaluation window (hours)", 24, 240, 48, 24)
    mo = c[1].toggle("Score on measured bins only", True, help="Only ~9 % of the 30-min bins in this file are real measurements; the rest are interpolated across gaps. Scoring on interpolated bins flatters the model.")
    steps = c[2].slider("Outlook length (steps)", 4, 24, 12)
order = (p, dd, q)
R = bal.run(full, data.key(d), freq, order, hrs, thr, mo); det = R["det"]; test = R["test"]; hz = list(R["hz"])
cur = bal.imbalance_pct(full.tail(1)).iloc[0]; tone, lab = bal.status(cur, thr)
a = det[det.model == "ARIMA"]; best = a.sort_values("MAE_pct").iloc[0]
k = st.columns(4)
with k[0]: ui.kpi("Imbalance now", f"{cur:.1f} %", f"{lab} · latest reading", tone)
with k[1]: ui.kpi("Peak actual in window", f"{R['act_imb'][R['used']].max():.1f} %", f"{int((R['act_imb'] > thr)[R['used']].sum())} bins over {thr:g} %", "warn" if R["act_imb"][R["used"]].max() > thr else "good")
with k[2]: ui.kpi("Best-tracked horizon", best.horizon, f"imbalance error ±{best.MAE_pct:.2f} pp", "info")
with k[3]: ui.kpi("Measured data", f"{R['share_measured']*100:.0f} %", f"{int(R['used'].sum())} of {len(test)} bins scored", "warn" if R["share_measured"] < .5 else "good")
if R["share_measured"] < .5:
    ui.note(f"Only <b>{R['share_measured']*100:.0f} %</b> of {freq} bins are real measurements (the file has recording gaps); the notebook filled the rest by interpolation. Accuracy figures are therefore based on very few points — treat them as indicative.")
st.write("")
t1, t2, t3, t4 = st.tabs(["Phase forecasts", "Imbalance alerts", "Accuracy", "Outlook"])
with t1:
    hsel = st.radio("Horizon", hz, horizontal=True, index=min(1, len(hz) - 1)); P = R["preds"]["ARIMA"][hsel]
    f = go.Figure()
    for ph in bal.FEATS:
        col = ui.PHASE[ph]; f.add_scatter(x=test.index, y=test[ph], name=f"{ph} actual", line=dict(color=col, width=2))
        f.add_scatter(x=P.index, y=P[ph], name=f"{ph} ARIMA", line=dict(color=col, width=1.6, dash="dot"))
    mm = test.index[R["measured"]]; f.add_scatter(x=mm, y=test.loc[mm, "YKW"], mode="markers", name="measured bin", marker=dict(color="#fff", size=5, symbol="diamond"))
    f.update_layout(title=f"Per-phase kW — actual vs ARIMA {hsel} ahead (white ◆ = measured, otherwise interpolated)", yaxis_title="kW"); st.plotly_chart(ui.style(f, 470), width="stretch")
with t2:
    f = go.Figure(); f.add_scatter(x=test.index, y=R["act_imb"], name="Actual", line=dict(color="#E6EAF2", width=2.2))
    for i, hname in enumerate(hz):
        im = bal.imbalance_pct(R["preds"]["ARIMA"][hname]); f.add_scatter(x=im.index, y=im, name=f"ARIMA {hname}", line=dict(width=1.5, color=ui.COLORWAY[i]))
        fl = im[im > thr]; f.add_scatter(x=fl.index, y=fl, mode="markers", marker=dict(symbol="x", size=9, color=ui.COLORWAY[i]), name=f"{hname} flagged", showlegend=False)
    f.add_hrect(y0=thr, y1=max(thr * 2, R["act_imb"].max() * 1.1), fillcolor="rgba(251,113,133,.08)", line_width=0); f.add_hline(y=thr, line_dash="dash", line_color=ui.RED, annotation_text=f"threshold {thr:g} %")
    f.update_layout(title="Load imbalance % — actual vs forecast-derived", yaxis_title="imbalance %"); st.plotly_chart(ui.style(f, 440), width="stretch")
    s = det.copy(); s[["precision", "recall"]] = s[["precision", "recall"]].round(2)
    st.dataframe(s.rename(columns={"mean_imbalance": "mean imb %", "MAE_pct": "imb error (pp)", "actual_flagged": "actual alerts", "flagged": "predicted alerts"}).round(2), width="stretch", hide_index=True)
with t3:
    met = R["metrics"]; m = st.selectbox("Metric", ["R2", "MAE", "RMSE"])
    f = go.Figure()
    for i, mod in enumerate(["ARIMA", "Persistence"]):
        g = met[met.model == mod].groupby("horizon", sort=False)[m].mean(); f.add_bar(x=g.index, y=g.values, name=mod, marker_color=[ui.AMBER, "#6B778F"][i])
    f.update_layout(barmode="group", title=f"{m} averaged across R/Y/B phases", hovermode="closest"); st.plotly_chart(ui.style(f, 360), width="stretch")
    st.dataframe(met.pivot_table(index=["horizon", "model"], columns="phase", values=m, sort=False).round(3), width="stretch")
    st.caption("Persistence = 'phase kW stays what it was at forecast time'. ARIMA should beat it at longer horizons; on this small, gappy dataset it often only ties.")
with t4:
    o, hist = bal.outlook(full, data.key(d), freq, order, steps); fl = o[o.imbalance > thr]
    f = go.Figure()
    for ph in bal.FEATS:
        col = ui.PHASE[ph]; h_ = hist[ph].tail(24)
        f.add_scatter(x=h_.index, y=h_, name=f"{ph} history", line=dict(color=col, width=2))
        f.add_scatter(x=o.index, y=o[ph], name=f"{ph} forecast", line=dict(color=col, dash="dot", width=2))
        f.add_scatter(x=list(o.index) + list(o.index[::-1]), y=list(o[ph + "_hi"]) + list(o[ph + "_lo"][::-1]), fill="toself", fillcolor="rgba(255,255,255,.05)", line=dict(width=0), showlegend=False, hoverinfo="skip")
    f.update_layout(title=f"Next {steps} × {freq} — phase forecast with 95 % interval", yaxis_title="kW"); st.plotly_chart(ui.style(f, 400), width="stretch")
    f = go.Figure(go.Bar(x=o.index, y=o.imbalance, marker_color=[ui.RED if v > thr else ui.GREEN for v in o.imbalance])); f.add_hline(y=thr, line_dash="dash", line_color=ui.RED)
    f.update_layout(title="Forecast imbalance %", hovermode="closest"); st.plotly_chart(ui.style(f, 260, legend=False), width="stretch")
    st.success("No imbalance alert expected in the outlook window.") if fl.empty else st.warning(f"{len(fl)} of {len(o)} future bins exceed {thr:g} % — first at {fl.index[0]:%d %b %H:%M}.")
    ui.note("Per-phase imbalance on a three-phase load is usually corrected by re-distributing single-phase loads. Use the flagged windows to plan switching.")
