"""
metrics.py -- pure functions that turn the journal into statistics. No I/O, fully unit-tested.

CONCEPTS (all computed on DAILY returns, annualised with 365 days since crypto never closes):
  * Sharpe  = mean return / volatility. Return per unit of total risk.
  * Sortino = mean return / DOWNSIDE deviation. Only punishes losses, not upside swings.
  * Max drawdown = worst peak-to-trough fall of the equity curve.
  * Beta / alpha vs BTC = regression of the arm's returns on BTC's. Beta is "how much BTC exposure
    you really have"; alpha is the part of the return NOT explained by that exposure.
  * IC (information coefficient) = rank correlation between the AI's forecast (outlook -2..+2)
    and what actually happened next. IC 0 = no skill. This is the PRIMARY test.
  * Newey-West t-stat: a t-stat that stays honest when observations are correlated over time
    (overlapping 24h windows are). A naive t-stat would overstate significance -- the same
    pseudo-replication trap Test 14 fell into.
"""
import math
from datetime import timedelta

import numpy as np
import pandas as pd

from crypto_ai.journal import parse_iso

ANN = 365


def daily_returns(equity_rows, arm):
    rows = [r for r in equity_rows if r["arm"] == arm]
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series({parse_iso(r["ts"]): r["equity"] for r in rows}).sort_index()
    daily = s.groupby(s.index.date).last()
    return daily.pct_change().dropna()


def sharpe(r):
    r = np.asarray(r, float)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return r.mean() / r.std(ddof=1) * math.sqrt(ANN)


def sortino(r):
    r = np.asarray(r, float)
    down = np.minimum(r, 0)
    dd = math.sqrt((down ** 2).mean()) if len(r) else 0
    return float("nan") if dd == 0 else r.mean() / dd * math.sqrt(ANN)


def max_drawdown(equity):
    e = np.asarray(equity, float)
    if len(e) == 0:
        return float("nan")
    return float((e / np.maximum.accumulate(e) - 1).min())


def alpha_beta(r, bench):
    df = pd.concat([pd.Series(r), pd.Series(bench)], axis=1, join="inner").dropna()
    if len(df) < 10:
        return {"alpha_ann": float("nan"), "beta": float("nan"), "alpha_t": float("nan"), "n": len(df)}
    y, x = df.iloc[:, 0].values, df.iloc[:, 1].values
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    s2 = resid @ resid / (len(y) - 2)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return {"alpha_ann": coef[0] * ANN, "beta": coef[1], "alpha_t": coef[0] / se[0], "n": len(df)}


def newey_west_t(x, lag):
    x = np.asarray([v for v in x if not np.isnan(v)], float)
    T = len(x)
    if T < 3:
        return float("nan"), T
    m = x.mean()
    d = x - m
    var = d @ d / T
    for l in range(1, min(lag, T - 1) + 1):
        var += 2 * (1 - l / (lag + 1)) * (d[l:] @ d[:-l]) / T
    return (m / math.sqrt(var / T) if var > 0 else float("nan")), T


def spearman(a, b):
    a, b = pd.Series(a, dtype=float), pd.Series(b, dtype=float)
    if a.nunique() < 2 or b.nunique() < 2:
        return float("nan")                      # all-equal forecast => no ranking => undefined
    return float(a.rank().corr(b.rank()))


def ic_series(decision_rows, snapshot_rows, horizon_hours):
    """Per-cycle cross-sectional IC between AI outlook and forward return over horizon_hours.
    Forward return uses the snapshot EXACTLY horizon_hours later; if that cycle was skipped,
    the observation is dropped (never interpolated)."""
    mids = {s["cycle_id"]: {a: v["mid"] for a, v in s["assets"].items()} for s in snapshot_rows}
    out = []
    for d in decision_rows:
        if d.get("arm") != "ai_pv" or not d.get("outlook"):
            continue
        t0 = parse_iso(d["cycle_id"].replace("Z", ":00Z"))
        c1 = (t0 + timedelta(hours=horizon_hours)).strftime("%Y-%m-%dT%H:00Z")
        if d["cycle_id"] not in mids or c1 not in mids:
            continue
        m0, m1 = mids[d["cycle_id"]], mids[c1]
        assets = [a for a in d["outlook"] if a in m0 and a in m1]
        ic = spearman([d["outlook"][a] for a in assets], [m1[a] / m0[a] - 1 for a in assets])
        out.append({"cycle_id": d["cycle_id"], "date": t0.date(), "ic": ic})
    return out


def daily_ic_stats(ic_rows, nw_lag_days):
    if not ic_rows:
        return {"n_cycles": 0, "n_undefined": 0, "n_days": 0, "mean_ic": float("nan"), "t_nw": float("nan")}
    df = pd.DataFrame(ic_rows)
    undefined = int(df["ic"].isna().sum())
    daily = df.dropna().groupby("date")["ic"].mean()
    t, n = newey_west_t(daily.values, nw_lag_days)
    half = len(daily) // 2
    return {"n_cycles": len(df), "n_undefined": undefined, "n_days": n,
            "mean_ic": float(daily.mean()) if n else float("nan"), "t_nw": t,
            "first_half_ic": float(daily.iloc[:half].mean()) if half else float("nan"),
            "second_half_ic": float(daily.iloc[half:].mean()) if half else float("nan")}


def primary_verdict(ic24, ic72):
    """PREREGISTRATION.md section 7, primary test. Only meaningful AT a decision point."""
    m, t = ic24["mean_ic"], ic24["t_nw"]
    if any(math.isnan(v) for v in (m, t)):
        return "NOT EVALUABLE"
    if (m >= 0.03 and t >= 2.0 and ic72["mean_ic"] > 0
            and ic24["first_half_ic"] > 0 and ic24["second_half_ic"] > 0):
        return "PREDICTIVE"
    if m >= 0.02 and 1.0 <= t < 2.0:
        return "INCONCLUSIVE"
    if m <= -0.03 and t <= -2.0:
        return "HARMFUL"
    return "NO EVIDENCE"
