import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import ui

st.set_page_config(page_title="GridSense · Load Forecast & Balance", page_icon="⚡", layout="wide")
ui.inject()

with st.sidebar:
    ui.brand()
    pages = [st.Page("page_overview.py", title="Overview", icon=":material/dashboard:", default=True),
             st.Page("page_prediction.py", title="Load Prediction", icon=":material/show_chart:"),
             st.Page("page_balance.py", title="Load Balance", icon=":material/balance:"),
             st.Page("page_drift.py", title="Drift Monitor", icon=":material/monitoring:")]
    nav = st.navigation(pages, position="sidebar")
    st.divider()
    with st.expander("Data source", expanded=False):
        up = st.file_uploader("Upload meter .xlsx", type=["xlsx"], help="Same columns as SecureElite_DeviceID19_KW_gt_1.xlsx (Timestamps, KW, RKW, YKW, BKW, KVA, I1…)")
        if up is not None: st.session_state["upload_bytes"] = up.getvalue()
        elif "upload_bytes" in st.session_state and st.button("Use bundled dataset"): del st.session_state["upload_bytes"]; st.rerun()
        st.caption("Using: " + ("uploaded file" if "upload_bytes" in st.session_state else "bundled SecureElite · Device 19"))
nav.run()
