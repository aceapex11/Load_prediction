import numpy as np, pandas as pd, plotly.graph_objects as go
import streamlit as st
import data, ui
import balance as bal

full, d = data.get()
last = full.iloc[-1]; imb = float(bal.imbalance_pct(full.tail(1)).iloc[0])
ui.hero("Plant load at a glance", "Short-term kW forecasting and three-phase balance monitoring for one metered feeder, built from your notebooks.",
        [f"Device {int(full.DeviceID.iloc[0])}", f"{len(full):,} readings", f"{full.Timestamps.min():%d %b} → {full.Timestamps.max():%d %b %Y}", f"{d.seg.nunique()} recordings"])
tone, label = bal.status(imb, 5.0)
c = st.columns(4)
with c[0]: ui.kpi("Latest load", f"{last.KW:,.0f} kW", f"{last.Timestamps:%d %b %Y %H:%M}", "info")
with c[1]: ui.kpi("Average load", f"{full.KW.mean():,.0f} kW", f"peak {full.KW.max():,.0f} kW")
with c[2]: ui.kpi("Phase imbalance (latest)", f"{imb:.1f} %", f"{label} · alert at 5 %", tone)
with c[3]: ui.kpi("Power factor", f"{full.PF.mean():.2f}", f"min {full.PF.min():.2f}", "good" if full.PF.mean() > .9 else "warn")
st.write("")
a, b = st.columns([2.2, 1])
with a:
    t, kw, ma = data.gapify(full.Timestamps, full.KW, full.KW.rolling(60, min_periods=10).mean(), max_gap="10min")
    f = go.Figure([go.Scatter(x=t, y=kw, name="KW (1-min)", line=dict(width=.8, color="#4A5B7D"), connectgaps=False),
                   go.Scatter(x=t, y=ma, name="60-min average", line=dict(width=2.4, color=ui.AMBER), connectgaps=False)])
    f.update_layout(title="Load over the whole period (gaps = no recording)", yaxis_title="kW"); f.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(ui.style(f, 400), width="stretch")
with b:
    p = full.tail(1)[bal.FEATS].iloc[0]
    f = go.Figure(go.Bar(x=["R", "Y", "B"], y=p.values, marker_color=[ui.PHASE[k] for k in bal.FEATS], text=[f"{v:.0f}" for v in p.values], textposition="outside"))
    f.add_hline(y=p.mean(), line_dash="dash", line_color="#8892A6", annotation_text="mean"); f.update_layout(title="Phase kW · latest reading", yaxis_range=[0, p.max() * 1.25], hovermode="closest")
    st.plotly_chart(ui.style(f, 400, legend=False), width="stretch")
a, b = st.columns(2)
with a:
    hp = full.assign(h=full.Timestamps.dt.hour).groupby("h").KW.agg(["mean", "min", "max"])
    f = go.Figure([go.Scatter(x=hp.index, y=hp["max"], line=dict(width=0), showlegend=False), go.Scatter(x=hp.index, y=hp["min"], fill="tonexty", fillcolor="rgba(255,181,71,.12)", line=dict(width=0), name="min–max"),
                   go.Scatter(x=hp.index, y=hp["mean"], name="mean", line=dict(color=ui.AMBER, width=3))])
    f.update_layout(title="Typical day (hour of day)", xaxis_title="hour", yaxis_title="kW"); st.plotly_chart(ui.style(f, 320), width="stretch")
with b:
    f = go.Figure(go.Histogram(x=d.KW, nbinsx=70, marker_color=ui.CYAN)); f.update_layout(title="KW distribution: two humps = HIGH / LOW machine states", xaxis_title="kW", hovermode="closest")
    st.plotly_chart(ui.style(f, 320, legend=False), width="stretch")
ui.note("The machine toggles between a HIGH and a LOW state every 3–4 minutes, so a 1-minute-ahead forecast can never be perfect. Expect R² ≈ 0.4–0.5 — see <b>Load Prediction</b>.")
seg = d.groupby("seg").agg(start=("Timestamps", "min"), end=("Timestamps", "max"), minutes=("KW", "size"), avg_kW=("KW", "mean")).query("minutes >= 100").round(0)
with st.expander("Continuous recordings (≥ 100 min)"): st.dataframe(seg, width="stretch")
