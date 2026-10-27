"""Short-term KW forecasting: baselines, classical ML (no KW inputs), SARIMA, GRU (NumPy), RLS.
Logic follows KW_Load_Forecasting_Drift_GRU_TL_SARIMA_noKW notebook."""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

SEQ = ["KVAR", "KVA", "PF", "I1", "I2", "I3", "V1", "V2", "V3", "F", "RTHDI", "YTHDI", "BTHDI"]  # KW / RKW / YKW / BKW are NOT inputs
W = 20
POWER_CH = [SEQ.index(c) for c in ("KVAR", "KVA")]
KVA_I = SEQ.index("KVA")


def windows(d, h):
    ts = d.Timestamps
    tod = (ts.dt.hour * 60 + ts.dt.minute) / 1440
    F = np.c_[d[SEQ].values, np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod)]
    kw = d.KW.values
    X, y, t, K, tgt = [], [], [], [], []
    for _, g in d.groupby("seg"):
        ix = g.index.values
        for k in range(W - 1, len(ix) - h):
            X.append(F[ix[k - W + 1:k + 1]]); y.append(kw[ix[k + h]]); t.append(ts.values[ix[k + h]])
            K.append(kw[ix[k - W + 1:k + 1]]); tgt.append(ix[k + h])
    return np.array(X), np.array(y), pd.to_datetime(np.array(t)), np.array(K), np.array(tgt)


def to_rel(X):
    Xr = X.copy(); level = X[:, :, KVA_I].mean(1)
    for c in POWER_CH:
        Xr[:, :, c] = X[:, :, c] - X[:, :, c].mean(1, keepdims=True)
    return np.concatenate([Xr, np.repeat(level[:, None, None], X.shape[1], 1)], 2)


def tabular(X):
    nf = len(SEQ); cols = {}
    for j, f in enumerate(SEQ):
        for l in range(3): cols[f"{f}_lag{l}"] = X[:, -1 - l, j]
    for f in ("KVA", "KVAR", "I1"):
        j = SEQ.index(f)
        for w in (4, 8, 20):
            s = X[:, -w:, j]; cols[f"{f}_mean{w}"] = s.mean(1); cols[f"{f}_std{w}"] = s.std(1); cols[f"{f}_min{w}"] = s.min(1); cols[f"{f}_max{w}"] = s.max(1)
    cols["tod_sin"] = X[:, -1, nf]; cols["tod_cos"] = X[:, -1, nf + 1]; cols["level"] = X[:, -1, nf + 2]
    return pd.DataFrame(cols)


def metrics(y, p):
    e = p - y; mse = np.mean(e ** 2)
    return dict(MAE=np.mean(np.abs(e)), RMSE=np.sqrt(mse), R2=1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2), MAPE=100 * np.mean(np.abs(e) / np.abs(y)))


# ---------------- GRU (plain NumPy, verified by gradient check in the notebook) ----------------
def _sig(x): return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


class GRU:
    def __init__(s, n_in, H=16, seed=0):
        r = np.random.RandomState(seed); s.H = H
        s.p = {"W": r.randn(n_in, 3 * H) * np.sqrt(1 / n_in), "U": np.concatenate([np.linalg.qr(r.randn(H, H))[0] for _ in range(3)], 1),
               "b": np.zeros(3 * H), "Wo": r.randn(H, 1) * np.sqrt(1 / H), "bo": np.zeros(1)}
        s.m = {k: np.zeros_like(v) for k, v in s.p.items()}; s.v = {k: np.zeros_like(v) for k, v in s.p.items()}; s.t = 0

    def forward(s, X, cache=False):
        B, T, _ = X.shape; H = s.H; W_, U, b = s.p["W"], s.p["U"], s.p["b"]; h = np.zeros((B, H)); C = []
        for t in range(T):
            x = X[:, t]; xw = x @ W_ + b
            z = _sig(xw[:, :H] + h @ U[:, :H]); r = _sig(xw[:, H:2 * H] + h @ U[:, H:2 * H])
            c = np.tanh(xw[:, 2 * H:] + (r * h) @ U[:, 2 * H:]); hn = (1 - z) * h + z * c
            if cache: C.append((x, h, z, r, c))
            h = hn
        y = (h @ s.p["Wo"] + s.p["bo"])[:, 0]
        return (y, h, C) if cache else y

    def grads(s, X, y):
        B, T, _ = X.shape; H = s.H; U = s.p["U"]; pred, h, C = s.forward(X, True); e = pred - y
        g = {k: np.zeros_like(v) for k, v in s.p.items()}; dy = (2 * e / B)[:, None]
        g["Wo"] = h.T @ dy; g["bo"] = dy.sum(0); dh = dy @ s.p["Wo"].T
        for t in reversed(range(T)):
            x, hp, z, r, c = C[t]; dz = dh * (c - hp); dc = dh * z; dhp = dh * (1 - z)
            dcp = dc * (1 - c ** 2); g["U"][:, 2 * H:] += (r * hp).T @ dcp; drh = dcp @ U[:, 2 * H:].T; dr = drh * hp; dhp += drh * r
            dzp = dz * z * (1 - z); drp = dr * r * (1 - r)
            g["U"][:, :H] += hp.T @ dzp; g["U"][:, H:2 * H] += hp.T @ drp; dhp += dzp @ U[:, :H].T + drp @ U[:, H:2 * H].T
            dg = np.concatenate([dzp, drp, dcp], 1); g["W"] += x.T @ dg; g["b"] += dg.sum(0); dh = dhp
        return g

    def fit(s, X, y, epochs=4, bs=128, lr=3e-3, seed=0, clip=5.0):
        r = np.random.RandomState(seed)
        for _ in range(epochs):
            idx = r.permutation(len(X))
            for i in range(0, len(X), bs):
                j = idx[i:i + bs]; g = s.grads(X[j], y[j]); n = np.sqrt(sum((v ** 2).sum() for v in g.values())); sc = min(1.0, clip / (n + 1e-12)); s.t += 1
                for k in s.p:
                    gk = g[k] * sc; s.m[k] = .9 * s.m[k] + .1 * gk; s.v[k] = .999 * s.v[k] + .001 * gk ** 2
                    s.p[k] -= lr * (s.m[k] / (1 - .9 ** s.t)) / (np.sqrt(s.v[k] / (1 - .999 ** s.t)) + 1e-8)

    def predict(s, X, bs=2048): return np.concatenate([s.forward(X[i:i + bs]) for i in range(0, len(X), bs)])


class RLS:
    """Recursive least squares with forgetting factor = the 'incremental learning' winner in the notebook."""
    def __init__(s, d, lam=0.998, delta=1e3): s.lam = lam; s.w = np.zeros(d); s.P = np.eye(d) * delta
    def predict(s, x): return float(s.w @ x)
    def update(s, x, y):
        Px = s.P @ x; k = Px / (s.lam + x @ Px); s.w = s.w + k * (y - s.w @ x); s.P = (s.P - np.outer(k, Px)) / s.lam


RLS_COLS = ["level"] + [f"{f}_lag{l}" for f in ("KVA", "KVAR", "I1") for l in range(3)] + [f"{f}_lag0" for f in ("F", "I2", "I3", "V1", "V2", "V3", "PF")] + ["KVA_mean4", "KVA_std4", "KVA_mean20"]
BASELINES = ["Persistence", "Moving avg (20 min)"]
ML = {"Ridge": lambda: make_pipeline(StandardScaler(), Ridge(100)),
      "Extra Trees": lambda: ExtraTreesRegressor(200, min_samples_leaf=3, max_features=.5, n_jobs=-1, random_state=0),
      "Random Forest": lambda: RandomForestRegressor(200, min_samples_leaf=3, max_features=.4, n_jobs=-1, random_state=0),
      "Hist Gradient Boosting": lambda: HistGradientBoostingRegressor(learning_rate=.05, max_iter=200, max_leaf_nodes=8, l2_regularization=1, random_state=0)}
ALL_MODELS = BASELINES + list(ML) + ["GRU", "RLS (online)", "SARIMA(2,0,1)"]


def latest_window(d):
    g = d[d.seg == d.seg.iloc[-1]].tail(W)
    ts = g.Timestamps; tod = (ts.dt.hour * 60 + ts.dt.minute) / 1440
    F = np.c_[g[SEQ].values, np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod)]
    return F[None], g.KW.values, g


@st.cache_resource(show_spinner="Training models… (cached after the first run)")
def backtest(_d, dkey, h, models):
    """Chronological 70/15/15 split. Models fit on train+val, scored on the newest 15 % (test)."""
    X, y, t, K, tgt = windows(_d, h); n = len(y); i1, i2 = int(.7 * n), int(.85 * n); te = slice(i2, n)
    Xr = to_rel(X); T = tabular(Xr); P = {}; nxt = {}
    Xl, Kl, gl = latest_window(_d); Tl = tabular(to_rel(Xl)); Xlr = to_rel(Xl)
    P["Persistence"] = K[te, -1]; nxt["Persistence"] = Kl[-1]
    P["Moving avg (20 min)"] = K[te].mean(1); nxt["Moving avg (20 min)"] = Kl.mean()
    for name in models:
        if name in ML:
            m = ML[name]().fit(T.iloc[:i2], y[:i2]); P[name] = m.predict(T.iloc[te]); nxt[name] = float(m.predict(Tl)[0])
        elif name == "GRU":
            mu = Xr[:i2].reshape(-1, Xr.shape[2]).mean(0); sd = Xr[:i2].reshape(-1, Xr.shape[2]).std(0) + 1e-9; ym, ys = y[:i2].mean(), y[:i2].std()
            ps, pl = [], []
            for s in range(2):
                g = GRU(Xr.shape[2], 16, seed=s); g.fit((Xr[:i2] - mu) / sd, (y[:i2] - ym) / ys, epochs=4, seed=s)
                ps.append(g.predict((Xr[te] - mu) / sd) * ys + ym); pl.append(g.predict((Xlr - mu) / sd)[0] * ys + ym)
            P[name] = np.mean(ps, 0); nxt[name] = float(np.mean(pl))
        elif name == "RLS (online)":
            A = T[RLS_COLS].values.astype(float); A = np.c_[(A - A[:800].mean(0)) / (A[:800].std(0) + 1e-9), np.ones(n)]; r = RLS(A.shape[1]); out = np.zeros(n)
            for i in range(n):
                out[i] = r.predict(A[i])
                if i - h >= 0: r.update(A[i - h], y[i - h])          # label of sample i-h is known at time i
            P[name] = out[te]; Al = T.iloc[:0]; al = tabular(Xlr)[RLS_COLS].values.astype(float)
            mu_a = T[RLS_COLS].values[:800].mean(0); sd_a = T[RLS_COLS].values[:800].std(0) + 1e-9
            nxt[name] = r.predict(np.r_[(al[0] - mu_a) / sd_a, 1.0])
        elif name.startswith("SARIMA") and h == 1:
            fit_rows = _d[_d.Timestamps <= t[i2 - 1]].KW.values
            res = SARIMAX(fit_rows, order=(2, 0, 1), trend="c", enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=100)
            pred = {}
            for s_, g in _d.groupby("seg"):
                pred.update(zip(g.index.values, np.asarray(res.apply(g.KW.values, refit=False).get_prediction().predicted_mean)))
            P[name] = np.array([pred[i] for i in tgt[te]]); nxt[name] = float(res.apply(gl.KW.values, refit=False).forecast(1)[0])
    yt = y[te]; rows = {k: metrics(yt, p) for k, p in P.items()}
    M = pd.DataFrame(rows).T; M["vs_MA20_%"] = 100 * (1 - M.RMSE / M.loc["Moving avg (20 min)", "RMSE"]); M = M.sort_values("RMSE")
    return dict(t=t[te], y=yt, P=P, M=M, next=nxt, h=h, last_t=gl.Timestamps.iloc[-1], last_kw=float(gl.KW.iloc[-1]), n_train=i2, n_test=n - i2)
