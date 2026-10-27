"""Drift tools from the notebook: PSI / KS (data drift) and a simplified ADWIN (level drift)."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


class ADWIN:
    def __init__(s, delta=0.002, maxw=1500, minsub=60, stride=5): s.d, s.w, s.maxw, s.minsub, s.stride = delta, [], maxw, minsub, stride
    def update(s, x):
        s.w.append(x); s.w = s.w[-s.maxw:]; n = len(s.w)
        if n < 2 * s.minsub or n % s.stride: return False
        a = np.asarray(s.w); c = np.cumsum(a); tot = c[-1]; var = a.var(); lnd = np.log(2 * np.log(n) / s.d)
        for i in range(s.minsub, n - s.minsub + 1, 3):
            n0, n1 = i, n - i; m = 1 / (1 / n0 + 1 / n1)
            eps = np.sqrt(2 * var * lnd / m) + 2 / 3 * lnd / m * max(a.max() - a.min(), 1e-9)
            if abs(c[i - 1] / n0 - (tot - c[i - 1]) / n1) > eps: s.w = s.w[i:]; return True
        return False


FEATURES = ["KW", "KVA", "KVAR", "PF", "I1", "I2", "I3", "V1", "V2", "V3", "F", "RTHDI", "YTHDI", "BTHDI"]


def feature_drift(a, b):
    rows = []
    for c in FEATURES:
        q = np.quantile(a[c], np.linspace(0, 1, 11)); q[0], q[-1] = -np.inf, np.inf
        pa = np.histogram(a[c], q)[0] / len(a) + 1e-6; pb = np.histogram(b[c], q)[0] / len(b) + 1e-6
        rows.append(dict(feature=c, PSI=float(np.sum((pb - pa) * np.log(pb / pa))), KS=ks_2samp(a[c], b[c]).statistic, before=a[c].mean(), after=b[c].mean()))
    return pd.DataFrame(rows).sort_values("PSI", ascending=False).reset_index(drop=True)


def level_alarms(d):
    lvl = d.KW.rolling(10, min_periods=1).mean().values; ad = ADWIN()
    return [i for i in range(len(lvl)) if ad.update(lvl[i])]
