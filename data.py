"""Data loading + cleaning (mirrors the notebooks)."""
import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

DEFAULT = Path(__file__).resolve().parent.parent / "data" / "SecureElite_DeviceID19_KW_gt_1.xlsx"


@st.cache_data(show_spinner="Reading meter data…")
def load(upload: bytes | None = None):
    """Returns (full, d). full = every valid row (used for load balance);
    d = rows with KW>=250 and a `seg` id that breaks at gaps > 2 min (used for load prediction)."""
    raw = pd.read_excel(io.BytesIO(upload) if upload else DEFAULT)
    raw.columns = raw.columns.astype(str).str.strip()
    raw["Timestamps"] = pd.to_datetime(raw["Timestamps"], errors="coerce")
    raw = raw.dropna(subset=["Timestamps", "KW"]).sort_values("Timestamps").drop_duplicates("Timestamps").reset_index(drop=True)
    d = raw.copy()
    d["valid"] = d.KW >= 250
    gap = d.Timestamps.diff().dt.total_seconds().fillna(0)
    d["seg"] = ((gap > 120) | (~d.valid) | (~d.valid.shift(1, fill_value=True))).cumsum()
    d = d[d.valid].reset_index(drop=True)
    return raw, d


def get():
    up = st.session_state.get("upload_bytes")
    return load(up)


def key(d):
    return f"{len(d)}-{d.Timestamps.iloc[0]}-{d.Timestamps.iloc[-1]}"


def gapify(t, *arrays, max_gap="3min"):
    """Insert NaN where timestamps jump so Plotly lines break across recording gaps."""
    t = pd.DatetimeIndex(t)
    brk = np.where(np.diff(t.values) > np.timedelta64(pd.Timedelta(max_gap).value, "ns"))[0] + 1
    tt = np.insert(t.values, brk, t.values[brk - 1] + np.timedelta64(1, "m")) if len(brk) else t.values
    out = [np.insert(np.asarray(a, float), brk, np.nan) if len(brk) else np.asarray(a, float) for a in arrays]
    return (tt, *out)
