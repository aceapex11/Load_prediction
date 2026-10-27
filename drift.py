import numpy as np, pandas as pd, plotly.graph_objects as go
import streamlit as st
from core import data, ui, drift

full, d = data.get()
ui.hero("Drift Monitor", "The machine's normal level shifts from period to period, so a model trained earlier slowly goes stale. Compare periods and watch for level changes.", ["PSI · KS test", "ADWIN level detector"])
split = st.date_input("Compare data before / after", value=pd.Timestamp("2024-10-01").date(), min_value=d.Timestamps.min().date(), max_value=d.Timestamps.max().date())
a, b = d[d.Timestamps < pd.Timestamp(split)], d[d.Timestamps >= pd.Timestamp(split)]
if len(a) < 100 or len(b) < 100: st.warning("Pick a date that leaves at least 100 readings on each side."); st.stop()
F = drift.feature_drift(a, b); al = drift.level_alarms(d)
k = st.columns(4)
with k[0]: ui.kpi("Mean load before → after", f"{a.KW.mean():.0f} → {b.KW.mean():.0f}", f"{b.KW.mean() - a.KW.mean():+.0f} kW", "warn")
with k[1]: ui.kpi("Features with major drift", f"{int((F.PSI > .25).sum())} / {len(F)}", "PSI > 0.25", "bad" if (F.PSI > .25).sum() else "good")
with k[2]: ui.kpi("Largest shift", F.feature.iloc[0], f"PSI {F.PSI.iloc[0]:.2f}", "info")
with k[3]: ui.kpi("Level-change alarms", f"{len(al)}", "ADWIN on 10-min KW level", "warn" if al else "good")
st.write("")
c1, c2 = st.columns([1, 1.4])
with c1:
    f = go.Figure(go.Bar(y=F.feature[::-1], x=F.PSI[::-1], orientation="h", marker_color=[ui.RED if v > .25 else ui.AMBER if v > .1 else ui.GREEN for v in F.PSI[::-1]]))
    f.add_vline(x=.25, line_dash="dash", line_color=ui.RED, annotation_text="major"); f.update_layout(title="Population Stability Index", hovermode="closest"); st.plotly_chart(ui.style(f, 460, legend=False), width="stretch")
with c2:
    t, y = data.gapify(d.Timestamps, d.KW.rolling(30, min_periods=10).mean(), max_gap="10min")[0:2]
    f = go.Figure(go.Scatter(x=t, y=y, name="30-min avg KW", line=dict(color=ui.CYAN, width=1.6), connectgaps=False))
    for i in al: f.add_vline(x=d.Timestamps.iloc[i], line_color=ui.RED, opacity=.35)
    f.add_vline(x=pd.Timestamp(split), line_dash="dot", line_color=ui.AMBER, annotation_text="split"); f.update_layout(title="Level drift — red lines = ADWIN alarms", yaxis_title="kW"); st.plotly_chart(ui.style(f, 460), width="stretch")
st.dataframe(F.round(3).rename(columns={"before": "mean before", "after": "mean after"}), width="stretch", hide_index=True)
ui.note("Notebook finding: continuously updated models (RLS) and GRU fine-tuning on recent data handled drift best; drift-<i>triggered</i> refits added nothing because detectors fire rarely. In production, re-check PSI weekly.")
