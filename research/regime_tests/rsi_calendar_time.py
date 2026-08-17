"""
rsi_calendar_time.py -- Test 18

The regime hypothesis is closed (3 runs: conditioning on choppy makes RSI
WORSE, not better). What survived is the UNCONDITIONAL result:

    n=8,896  mean=+0.731%  t=6.21
    fair monkey (random entry, same exit rule) = +0.055%
    entry-signal edge = +0.676pp

This contradicts Test 2, which rejected RSI at t=0.34 on n=71 -- explained
by sample size (13 stocks / 3 years vs 1,345 stocks / 12 years).

REMAINING FATAL FLAW: CLUSTERING. RSI<30 signals are not independent. When
the market sells off, hundreds of stocks cross oversold on the SAME DAY.
8,896 trades represent far fewer independent episodes. This is precisely
what killed Test 14, where t fell from -4.02 to -1.60 after declustering.

FIX: calendar-time aggregation. Collapse all trades entered in the same
MONTH into a single observation, then t-test the monthly series. The
t-statistic then counts independent time periods, not correlated trades.

ALSO REPORTED:
  - trades per month (shows how severe the clustering is)
  - fraction of positive months
  - SPY comparison over the same months (project rule 5)

PRE-REGISTERED PASS CRITERIA (fixed before running):
  1. monthly-series t-statistic > 2.0
  2. mean monthly return exceeds 0.05% cost
  3. positive-month fraction > 55%
  4. beats SPY's mean monthly return over the same window
"""

import numpy as np
import pandas as pd

RSI_PERIOD, RSI_ENTRY, RSI_EXIT, MAX_HOLD = 14, 30, 55, 20
COST = 0.0005
PRICE_MIN, PRICE_MAX, MIN_VOL = 5.0, 100.0, 250_000
PRACTICE_END = "2018-12-31"

print("=" * 68)
print("TEST 18 -- CALENDAR-TIME AGGREGATION (declustering)")
print("=" * 68, flush=True)

raw = pd.read_parquet("data/yf_universe.parquet",
                      columns=["ticker", "date", "open", "close", "volume"])
raw["date"] = pd.to_datetime(raw["date"])
raw = raw[raw["date"] <= PRACTICE_END]
raw = raw[(raw["close"] > 0) & (raw["open"] > 0)]
raw = raw.sort_values(["ticker", "date"]).reset_index(drop=True)
raw["adv20"] = raw.groupby("ticker")["volume"].transform(
    lambda s: s.rolling(20, min_periods=20).mean())
raw["eligible"] = ((raw["close"] >= PRICE_MIN) &
                   (raw["close"] <= PRICE_MAX) &
                   (raw["adv20"] >= MIN_VOL))


def wilder_rsi(close, period=RSI_PERIOD):
    n = len(close); out = np.full(n, np.nan)
    if n < period + 1:
        return out
    d = np.diff(close)
    g = np.where(d > 0, d, 0.0); l = np.where(d < 0, -d, 0.0)
    ag, al = g[:period].mean(), l[:period].mean()
    for i in range(period, n - 1):
        ag = (ag * (period - 1) + g[i]) / period
        al = (al * (period - 1) + l[i]) / period
        rs = ag / al if al > 0 else np.inf
        out[i + 1] = 100.0 - 100.0 / (1.0 + rs)
    return out


print("Computing RSI...", flush=True)
parts = []
for i, (tk, g) in enumerate(raw.groupby("ticker", sort=False)):
    if i % 400 == 0:
        print(f"  {i}...", flush=True)
    parts.append(pd.Series(wilder_rsi(g["close"].values), index=g.index))
raw["rsi"] = pd.concat(parts).sort_index()

print("Generating trades...", flush=True)
trades = []
for tk, g in raw.groupby("ticker", sort=False):
    rsi = g["rsi"].values; opens = g["open"].values
    elig = g["eligible"].values; dates = g["date"].values
    n = len(g); i = RSI_PERIOD + 1
    while i < n - 1:
        if (np.isfinite(rsi[i-1]) and np.isfinite(rsi[i])
                and rsi[i-1] < RSI_ENTRY <= rsi[i] and elig[i]):
            ei = i + 1
            if ei >= n - 1:
                break
            epx = opens[ei]; xi = None
            for j in range(ei, min(ei + MAX_HOLD, n - 1)):
                if np.isfinite(rsi[j]) and rsi[j] > RSI_EXIT:
                    xi = j + 1; break
            if xi is None:
                xi = min(ei + MAX_HOLD, n - 1)
            trades.append((dates[ei], opens[xi]/epx - 1.0 - COST))
            i = xi + 1
        else:
            i += 1

tr = pd.DataFrame(trades, columns=["entry_date", "ret"])
tr["month"] = pd.to_datetime(tr["entry_date"]).dt.to_period("M")
print(f"  {len(tr):,} trades", flush=True)

per_month = tr.groupby("month").size()
print("\n" + "=" * 68)
print("CLUSTERING DIAGNOSTIC")
print("=" * 68)
print(f"  distinct months with trades : {len(per_month)}")
print(f"  trades per month  mean={per_month.mean():.1f}  "
      f"median={per_month.median():.0f}  max={per_month.max()}")
print(f"  -> {len(tr):,} trades collapse to {len(per_month)} months")

monthly = tr.groupby("month")["ret"].mean()
n_m = len(monthly)
m_mean = monthly.mean()
m_t = m_mean / (monthly.std(ddof=1) / np.sqrt(n_m))
pos_frac = (monthly > 0).mean()

print("\n" + "=" * 68)
print("CALENDAR-TIME RESULTS (one observation per month)")
print("=" * 68)
print(f"  months              : {n_m}")
print(f"  mean monthly return : {m_mean*100:+.3f}%")
print(f"  median month        : {monthly.median()*100:+.3f}%")
print(f"  t-statistic         : {m_t:.2f}   <-- the honest number")
print(f"  positive months     : {pos_frac*100:.1f}%")
print("  (trade-level t was 6.21 on n=8,896)")

print("\n" + "=" * 68)
print("SPY COMPARISON (rule 5)")
print("=" * 68)
spy_mean = None
try:
    spy = pd.read_parquet("data/spy_yf.parquet")
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy[spy["date"] <= PRACTICE_END].sort_values("date")
    spy["month"] = spy["date"].dt.to_period("M")
    spy_m = spy.groupby("month")["close"].last().pct_change().dropna()
    common = monthly.index.intersection(spy_m.index)
    if len(common) > 12:
        s = spy_m.loc[common]
        r = monthly.loc[common]
        spy_mean = s.mean()
        diff = r - s
        d_t = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))
        print(f"  overlapping months       : {len(common)}")
        print(f"  strategy mean monthly    : {r.mean()*100:+.3f}%")
        print(f"  SPY mean monthly         : {s.mean()*100:+.3f}%")
        print(f"  difference               : {(r.mean()-s.mean())*100:+.3f}pp")
        print(f"  t-stat on difference     : {d_t:.2f}")
        print(f"  months beating SPY       : {(diff > 0).mean()*100:.1f}%")
    else:
        print("  insufficient overlap")
except FileNotFoundError:
    print("  data/spy_yf.parquet not found -- skipped")

print("\n" + "=" * 68)
print("YEAR BY YEAR")
print("=" * 68)
tr["yr"] = pd.to_datetime(tr["entry_date"]).dt.year
for yr, row in tr.groupby("yr")["ret"].agg(["mean", "count"]).iterrows():
    print(f"  {yr}: {row['mean']*100:+6.2f}%  (n={int(row['count']):>5,})")

print("\n" + "=" * 68)
print("VERDICT vs PRE-REGISTERED CRITERIA")
print("=" * 68)
k1 = m_t > 2.0
k2 = m_mean > COST
k3 = pos_frac > 0.55
k4 = (spy_mean is not None) and (m_mean > spy_mean)
print(f"  {'PASS' if k1 else 'FAIL'}  monthly t > 2.0        (t={m_t:.2f})")
print(f"  {'PASS' if k2 else 'FAIL'}  mean > cost            ({m_mean*100:+.3f}%)")
print(f"  {'PASS' if k3 else 'FAIL'}  positive months > 55%  ({pos_frac*100:.1f}%)")
if spy_mean is not None:
    print(f"  {'PASS' if k4 else 'FAIL'}  beats SPY monthly      "
          f"({m_mean*100:+.3f}% vs {spy_mean*100:+.3f}%)")
print(f"\n  OVERALL: {'PASS' if all([k1, k2, k3, k4]) else 'FAIL'}")
print("=" * 68, flush=True)