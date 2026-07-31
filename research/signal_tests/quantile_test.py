"""
quantile_test.py -- the make-or-break cost test for the short-term
reversal effect the IC diagnostic surfaced (mean IC ~ -0.05 at 1d,
~ -0.10 at 5d, consistent sign across ~11-12 of 13 stocks).

alphalens-reloaded was tried first, per instructions, and rejected: it
installs cleanly but pulls in pandas==2.3.3 as a hard dependency
resolution, silently downgrading this project's pinned pandas 3.0.3.
That is exactly the "doesn't work with pandas 3.0.3" case we were told
not to fight -- pandas 3.0.3 was restored immediately after uninstalling
alphalens-reloaded, and everything below is self-contained (numpy /
pandas only).

Signal (fixed, not tuned): 20-day close z-score, (close - MA20) / std20.

Analysis A -- time-series quantile test: pool every (stock, day) into
one sample per horizon, bucket into quintiles by z-score, compare mean
forward return of the most-oversold quintile (Q1) against the most-
overbought (Q5).

Analysis B -- cross-sectional long/short: each rebalance day, rank the
13 stocks against each other by the same z-score, go long the 3 most
oversold / short the 3 most overbought, equal-weighted, dollar-neutral.
Tested at 1/5/10/20-day rebalance frequencies.

Both analyses report gross returns, net returns at 0.1% / 0.2% / 0.4%
round-trip cost, and the break-even cost where the edge hits zero.

Only the 70% practice window is read for every stock; the 30% holdout
is never touched. No parameters are tuned, no alternate signal is
tried -- the 20-day z-score is used exactly as specified.
"""
import os
import numpy as np
import pandas as pd

UNIVERSE = ["SBRA", "VLY", "FLO", "AROC", "HUN", "WEN",
            "CLF", "MGNI", "KSS", "TROX", "VSH", "UAA", "HL"]
TUNE = 0.70
Z_WINDOW = 20
HORIZONS = [1, 5, 10, 20]
COST_LEVELS = [0.001, 0.002, 0.004]   # 0.1%, 0.2%, 0.4% round-trip


def load_tune_data():
    data = {}
    for ticker in UNIVERSE:
        path = os.path.join("data", f"{ticker}_daily.csv")
        if not os.path.exists(path):
            print(f"{ticker}: SKIPPED (missing CSV at {path})")
            continue
        df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        n_tune = int(len(df) * TUNE)
        data[ticker] = df.iloc[:n_tune].reset_index(drop=True)
    return data


def compute_zscore20(close):
    s = pd.Series(close)
    ma = s.rolling(Z_WINDOW).mean()
    std = s.rolling(Z_WINDOW).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (s - ma) / std
    return z.to_numpy()


def forward_return(opens, horizon):
    """fwd[N] = open[N+1+horizon]/open[N+1] - 1: signal known at day N's
    close predicts the return from day N+1's open to day N+1+horizon's open."""
    n = len(opens)
    fwd = np.full(n, np.nan)
    last_n = n - 1 - horizon
    if last_n <= 0:
        return fwd
    idx = np.arange(0, last_n)
    fwd[idx] = opens[idx + 1 + horizon] / opens[idx + 1] - 1
    return fwd


def find_breakeven_cost(net_total_return_fn, hi=0.05, tol=1e-6, max_iter=60):
    """Bisection for the round-trip cost at which total net return hits
    zero. net_total_return_fn(cost) must be (roughly) monotonically
    decreasing in cost, which holds for realistic cost ranges here."""
    if net_total_return_fn(0.0) <= 0:
        return None   # already unprofitable gross -- no positive breakeven
    if net_total_return_fn(hi) > 0:
        return None   # still profitable at the search ceiling
    lo = 0.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        if net_total_return_fn(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


DATA = load_tune_data()

# ------------------------------------------------------------------
# Shared per-ticker signal / forward-return arrays
# ------------------------------------------------------------------
ticker_z = {t: compute_zscore20(df["close"].to_numpy(dtype=float)) for t, df in DATA.items()}
ticker_opens = {t: df["open"].to_numpy(dtype=float) for t, df in DATA.items()}


# ============================================================
# ANALYSIS A -- pooled time-series quantile test
# ============================================================
print()
print("=" * 92)
print("ANALYSIS A: Time-series quantile test (20d z-score, pooled stock-days, gross)")
print("=" * 92)

quintile_gross = {}   # h -> Series indexed by quintile label 1..5, mean fwd return
quintile_counts = {}
spread_gross = {}      # h -> Q1 - Q5

for h in HORIZONS:
    frames = []
    for ticker in DATA:
        z = ticker_z[ticker]
        fwd = forward_return(ticker_opens[ticker], h)
        mask = ~np.isnan(z) & ~np.isnan(fwd)
        frames.append(pd.DataFrame({"z": z[mask], "fwd": fwd[mask]}))
    pooled = pd.concat(frames, ignore_index=True)
    pooled["quintile"] = pd.qcut(pooled["z"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    grouped = pooled.groupby("quintile", observed=True)["fwd"].agg(["mean", "count"])
    quintile_gross[h] = grouped["mean"]
    quintile_counts[h] = grouped["count"]
    spread_gross[h] = float(grouped["mean"].loc[1] - grouped["mean"].loc[5])

header = f"{'Quintile':<12}" + "".join(f"{('h='+str(h)+'d'):>12}" for h in HORIZONS)
print(header)
print("-" * len(header))
for q in [1, 2, 3, 4, 5]:
    label = f"Q{q}" + (" (oversold)" if q == 1 else " (overbought)" if q == 5 else "")
    print(f"{label:<12}" + "".join(f"{quintile_gross[h].loc[q]*100:>11.3f}%" for h in HORIZONS))
print("-" * len(header))
print(f"{'Spread Q1-Q5':<12}" + "".join(f"{spread_gross[h]*100:>11.3f}%" for h in HORIZONS))
print(f"{'n (pooled)':<12}" + "".join(f"{int(quintile_counts[h].sum()):>12}" for h in HORIZONS))

for cost in COST_LEVELS:
    print()
    print(f"-- Net of {cost*100:.1f}% round-trip cost (per quintile: gross - cost; "
          f"spread: gross - 2x cost, one round trip per leg) --")
    print(header)
    print("-" * len(header))
    for q in [1, 2, 3, 4, 5]:
        label = f"Q{q}" + (" (oversold)" if q == 1 else " (overbought)" if q == 5 else "")
        print(f"{label:<12}" + "".join(f"{(quintile_gross[h].loc[q]-cost)*100:>11.3f}%" for h in HORIZONS))
    print(f"{'Spread':<12}" + "".join(f"{(spread_gross[h]-2*cost)*100:>11.3f}%" for h in HORIZONS))

print()
print("Break-even round-trip cost for the Q1-Q5 spread (cost at which net spread = 0):")
for h in HORIZONS:
    s = spread_gross[h]
    if s <= 0:
        print(f"  h={h:>2}d: spread already <= 0 gross -- no positive break-even cost")
    else:
        print(f"  h={h:>2}d: {s/2*100:.3f}%  (gross spread {s*100:.3f}% / 2 legs)")


# ============================================================
# ANALYSIS B -- cross-sectional long/short
# ============================================================
print()
print("=" * 92)
print("ANALYSIS B: Cross-sectional long/short (long 3 most oversold, short 3 most overbought)")
print("=" * 92)

# Build a common, date-aligned panel across all 13 stocks (inner join on
# date, restricted to each stock's own practice window -- never the holdout).
z_series, open_series = {}, {}
for ticker, df in DATA.items():
    z = pd.Series(ticker_z[ticker], index=df["date"])
    o = pd.Series(df["open"].to_numpy(dtype=float), index=df["date"])
    z_series[ticker] = z
    open_series[ticker] = o

z_wide = pd.DataFrame(z_series)
opens_wide = pd.DataFrame(open_series)
valid = z_wide.notna().all(axis=1) & opens_wide.notna().all(axis=1)
z_wide = z_wide.loc[valid].sort_index()
opens_wide = opens_wide.loc[valid].sort_index()
tickers = list(z_wide.columns)
n_days = len(z_wide)
print(f"Aligned panel: {n_days} common trading days across {len(tickers)} stocks "
      f"(inner-joined on date, practice window only)")

analysisB = {}
for h in HORIZONS:
    last_i = n_days - 2 - h
    positions = list(range(0, last_i + 1, h))

    gross_returns, longs, shorts = [], [], []
    for i in positions:
        z_today = z_wide.iloc[i]
        ranked = z_today.sort_values()
        long_t = ranked.index[:3].tolist()
        short_t = ranked.index[-3:].tolist()
        entry_i, exit_i = i + 1, i + 1 + h
        long_ret = (opens_wide.iloc[exit_i][long_t].to_numpy() /
                    opens_wide.iloc[entry_i][long_t].to_numpy() - 1)
        short_ret = (opens_wide.iloc[exit_i][short_t].to_numpy() /
                     opens_wide.iloc[entry_i][short_t].to_numpy() - 1)
        gross_returns.append(float(long_ret.mean() - short_ret.mean()))
        longs.append(set(long_t))
        shorts.append(set(short_t))

    gross_returns = np.array(gross_returns)

    prev_w = pd.Series(0.0, index=tickers)
    turnovers = []
    for lset, sset in zip(longs, shorts):
        w = pd.Series(0.0, index=tickers)
        for t in lset:
            w[t] = 1 / 3
        for t in sset:
            w[t] = -1 / 3
        turnovers.append(0.5 * float(np.abs(w - prev_w).sum()))
        prev_w = w
    turnovers = np.array(turnovers)

    n_periods = len(gross_returns)
    periods_per_year = 252.0 / h
    n_days_spanned = n_periods * h

    def net_total_return(cost, gross=gross_returns, turn=turnovers):
        net = gross - turn * cost
        return float(np.prod(1 + net) - 1)

    variants = {"gross": 0.0}
    for c in COST_LEVELS:
        variants[f"net_{c*1000:.0f}bp"] = c

    variant_stats = {}
    for label, cost in variants.items():
        net = gross_returns - turnovers * cost
        equity = np.cumprod(1 + net)
        total_return = float(equity[-1] - 1)
        annualized = float(equity[-1] ** (252.0 / n_days_spanned) - 1) if n_days_spanned > 0 else float("nan")
        running_max = np.maximum.accumulate(equity)
        drawdown = equity / running_max - 1
        max_dd = float(drawdown.min())
        sharpe = float(net.mean() / net.std(ddof=1) * np.sqrt(periods_per_year)) if net.std(ddof=1) > 0 else float("nan")
        variant_stats[label] = {
            "total_return": total_return, "annualized": annualized,
            "sharpe": sharpe, "max_dd": max_dd,
        }

    breakeven = find_breakeven_cost(net_total_return)

    analysisB[h] = {
        "n_periods": n_periods,
        "avg_turnover": float(turnovers.mean()),
        "variants": variant_stats,
        "breakeven": breakeven,
    }

for h in HORIZONS:
    res = analysisB[h]
    print()
    print(f"--- Rebalance every {h} trading day(s) -- {res['n_periods']} periods, "
          f"avg turnover {res['avg_turnover']*100:.1f}% per rebalance ---")
    row_header = f"{'Variant':<12}{'TotalRet':>11}{'AnnRet':>10}{'Sharpe':>9}{'MaxDD':>9}"
    print(row_header)
    print("-" * len(row_header))
    for label, cost in [("gross", 0.0)] + [(f"net_{c*1000:.0f}bp", c) for c in COST_LEVELS]:
        v = res["variants"][label]
        print(f"{label:<12}{v['total_return']*100:>10.2f}%{v['annualized']*100:>9.2f}%"
              f"{v['sharpe']:>9.2f}{v['max_dd']*100:>8.2f}%")
    if res["breakeven"] is None:
        print("Break-even round-trip cost: n/a (unprofitable gross, or profitable beyond the 5% search ceiling)")
    else:
        print(f"Break-even round-trip cost: {res['breakeven']*100:.3f}%")


# ============================================================
# PLAIN-ENGLISH SUMMARY
# ============================================================
print()
print("=" * 92)
print("SUMMARY: what survives 0.2% round-trip costs")
print("=" * 92)

REF_COST = 0.002
survivors_A = [h for h in HORIZONS if (spread_gross[h] - 2 * REF_COST) > 0]
survivors_B = [h for h in HORIZONS if analysisB[h]["variants"]["net_2bp"]["total_return"] > 0]

if survivors_A:
    print(f"Analysis A (quintile spread): profitable net of 0.2% cost at horizon(s) "
          f"{', '.join(str(h)+'d' for h in survivors_A)}.")
else:
    print("Analysis A (quintile spread): NOT profitable at any horizon after 0.2% cost.")

if survivors_B:
    print(f"Analysis B (cross-sectional long/short): profitable net of 0.2% cost at rebalance "
          f"frequency(ies) {', '.join(str(h)+'d' for h in survivors_B)}.")
else:
    print("Analysis B (cross-sectional long/short): NOT profitable at any rebalance frequency after 0.2% cost.")

print()
print("This is a cost-survival diagnostic only: no parameters were tuned and the 20-day")
print("z-score was tested exactly as specified.")
