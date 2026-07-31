"""
robustness_test.py -- pre-holdout robustness checks on the one finding
that passed pre-registration in factor_test.py: low-volatility long-only
(net Sharpe 0.88 vs SPY 0.64, annualized alpha +4.23%, t=2.47, beta
0.662). Momentum is already rejected (no significant alpha, t=0.13) and
is NOT re-tested here.

The 2019+ holdout remains sealed. Every check below runs on the
practice window only. Data loading, the universe filters, the monthly
panel construction, and the alpha-regression math are reused verbatim
from factor_test.py -- not reimplemented, not altered.

Check 1: sub-period stability (2008-01..2012-12 crisis+recovery vs
         2013-01..2018-10 calm bull market) -- is the alpha a crisis artifact?
Check 2: crisis exclusion (drop 2008-09..2009-06 entirely) -- does the
         alpha collapse without the crash months?
Check 3: sector concentration of the long decile (fetched once per
         ticker via yfinance Ticker.info, cached to disk) -- is this
         just "utilities and staples" wearing a factor's clothing?
Check 4: position-count sensitivity (top 10 / 20 / 38 lowest-vol stocks
         instead of a full decile) -- does a smaller, tradeable book
         still show the effect?
Check 5: realistic small-account commission drag on the 38- and
         20-stock books ($4,700 account, $1.00/order minimum, IBKR-style).

No parameters are tuned, no alternate volatility lookback is tried, and
nothing here is optimized -- these are diagnostics on an already-fixed,
already-specified strategy.
"""
import os
import json
import time
import numpy as np
import pandas as pd
import yfinance as yf

DATA_PATH = os.path.join("data", "yf_universe.parquet")
SPY_PATH = os.path.join("data", "spy_yf.parquet")
HOLDOUT_START = pd.Timestamp("2019-01-01")

PRICE_MIN, PRICE_MAX = 5.0, 100.0
MIN_ADV = 250_000
MIN_HISTORY_MONTHS = 24
GAP_DAYS_THRESHOLD = 10
MOVE_THRESHOLD = 0.80
MIN_UNIVERSE_FOR_DECILE = 20
SECTOR_CACHE_PATH = os.path.join("data", "sector_cache.json")
COST_02 = 0.002

ACCOUNT_VALUE_USD = 4700.0
COMMISSION_PER_ORDER = 1.00


# ============================================================
# LOAD + HOLDOUT BOUNDARY (identical to factor_test.py)
# ============================================================
raw = pd.read_parquet(DATA_PATH)
raw["date"] = pd.to_datetime(raw["date"])
raw_spy = pd.read_parquet(SPY_PATH)
raw_spy["date"] = pd.to_datetime(raw_spy["date"])

print("=" * 96)
print("HOLDOUT CONFIRMATION")
print("=" * 96)
print(f"Raw stock panel: {raw['date'].min().date()} to {raw['date'].max().date()} "
      f"({len(raw)} rows, {raw['ticker'].nunique()} tickers)")
print(f"Raw SPY panel: {raw_spy['date'].min().date()} to {raw_spy['date'].max().date()} ({len(raw_spy)} rows)")

practice = raw[raw["date"] < HOLDOUT_START].copy()
n_dropped_holdout = len(raw) - len(practice)
del raw
spy_practice = raw_spy[raw_spy["date"] < HOLDOUT_START].copy()
n_dropped_spy = len(raw_spy) - len(spy_practice)
del raw_spy

print(f"Practice window kept: date < {HOLDOUT_START.date()} "
      f"(stocks: {len(practice)} rows, SPY: {len(spy_practice)} rows)")
print(f"Rows excluded as holdout: stocks {n_dropped_holdout}, SPY {n_dropped_spy}")
print(f"Maximum date present anywhere from here on: "
      f"{max(practice['date'].max(), spy_practice['date'].max()).date()}")
print("CONFIRMED: the post-2018 holdout is not read anywhere in this script.")

df = practice.sort_values(["ticker", "date"]).reset_index(drop=True)
spy = spy_practice.sort_values("date").reset_index(drop=True)


# ============================================================
# DATA QUALITY CHECKS (identical to factor_test.py)
# ============================================================
n0 = len(df)
bad_price = (df[["open", "high", "low", "close"]] <= 0).any(axis=1) | df["volume"].lt(0)
df = df[~bad_price].copy()

df["prev_close"] = df.groupby("ticker")["close"].shift(1)
df["daily_ret"] = df["close"] / df["prev_close"] - 1
big_move = df["daily_ret"].abs() > MOVE_THRESHOLD
df = df[~big_move.fillna(False)].copy()
df = df.drop(columns=["prev_close", "daily_ret"])

print(f"\nData quality: {n0} rows before checks, {len(df)} after "
      f"({n0 - len(df)} dropped, {(n0-len(df))/n0*100:.3f}%)")


# ============================================================
# BUILD MONTHLY PANEL (identical to factor_test.py)
# ============================================================
df["month"] = df["date"].dt.to_period("M")

close_last = df.sort_values("date").groupby(["ticker", "month"])["close"].last()
open_first = df.sort_values("date").groupby(["ticker", "month"])["open"].first()
vol_mean = df.groupby(["ticker", "month"])["volume"].mean()

monthly = pd.DataFrame({"close": close_last, "open": open_first, "avg_volume": vol_mean}).reset_index()
monthly = monthly.sort_values(["ticker", "month"]).reset_index(drop=True)

g = monthly.groupby("ticker")
monthly["next_open"] = g["open"].shift(-1)
monthly["next_month"] = g["month"].shift(-1)
monthly["next2_open"] = g["open"].shift(-2)
monthly["next2_month"] = g["month"].shift(-2)
monthly["ret"] = g["close"].pct_change()
monthly["hist_months"] = g.cumcount()

valid_next = monthly["next_month"] == (monthly["month"] + 1)
valid_next2 = monthly["next2_month"] == (monthly["month"] + 2)
monthly.loc[~valid_next, "next_open"] = np.nan
monthly.loc[~valid_next2, "next2_open"] = np.nan

monthly["low_vol_signal"] = g["ret"].transform(lambda s: s.rolling(12).std())

monthly["eligible"] = (
    (monthly["close"] >= PRICE_MIN) & (monthly["close"] <= PRICE_MAX) &
    (monthly["avg_volume"] > MIN_ADV) &
    (monthly["hist_months"] >= MIN_HISTORY_MONTHS)
)

all_months = sorted(monthly["month"].unique())
print(f"Monthly panel: {monthly['ticker'].nunique()} tickers, {len(all_months)} months "
      f"({all_months[0]} to {all_months[-1]})")

# SPY monthly period-return series (identical convention)
spy["month"] = spy["date"].dt.to_period("M")
spy_open_first = spy.groupby("month")["open"].first().reset_index().sort_values("month").reset_index(drop=True)
spy_open_first["next_open"] = spy_open_first["open"].shift(-1)
spy_open_first["next_month"] = spy_open_first["month"].shift(-1)
spy_open_first["next2_open"] = spy_open_first["open"].shift(-2)
spy_open_first["next2_month"] = spy_open_first["month"].shift(-2)
spy_valid_next = spy_open_first["next_month"] == (spy_open_first["month"] + 1)
spy_valid_next2 = spy_open_first["next2_month"] == (spy_open_first["month"] + 2)
spy_open_first.loc[~spy_valid_next, "next_open"] = np.nan
spy_open_first.loc[~spy_valid_next2, "next2_open"] = np.nan
spy_open_first["spy_ret"] = spy_open_first["next2_open"] / spy_open_first["next_open"] - 1
SPY_MONTHLY = spy_open_first.set_index("month")["spy_ret"]


# ============================================================
# LOW-VOL FACTOR ENGINE (generalized: month subset + fixed-N option)
# ============================================================
def run_low_vol(months=None, long_n=None):
    use_months = months if months is not None else all_months
    period_records = []
    prev_w = pd.Series(dtype=float)

    for m in use_months:
        month_df = monthly[(monthly["month"] == m) & monthly["eligible"]]
        month_df = month_df.dropna(subset=["low_vol_signal", "next_open", "next2_open"])
        if len(month_df) < MIN_UNIVERSE_FOR_DECILE:
            continue

        month_df = month_df.sort_values("low_vol_signal")
        if long_n is not None:
            n_long = min(long_n, len(month_df))
        else:
            n_long = max(1, len(month_df) // 10)
        long_df = month_df.iloc[:n_long]

        long_ret = long_df["next2_open"] / long_df["next_open"] - 1
        w = pd.Series(1.0 / len(long_df), index=long_df["ticker"].to_numpy())
        idx = w.index.union(prev_w.index)
        turnover = 0.5 * float((w.reindex(idx, fill_value=0.0) -
                                prev_w.reindex(idx, fill_value=0.0)).abs().sum())
        prev_w = w

        period_records.append({
            "month": m, "n_long": len(long_df),
            "long_ret": float(long_ret.mean()), "turnover": turnover,
            "tickers": long_df["ticker"].tolist(),
        })

    res = pd.DataFrame(period_records)
    if len(res):
        res["spy_ret"] = res["month"].map(SPY_MONTHLY)
    return res


def perf_stats(period_returns):
    period_returns = np.asarray(period_returns, dtype=float)
    equity = np.cumprod(1 + period_returns)
    n = len(period_returns)
    total_return = float(equity[-1] - 1)
    annualized = float(equity[-1] ** (12.0 / n) - 1) if n > 0 else float("nan")
    running_max = np.maximum.accumulate(equity)
    max_dd = float((equity / running_max - 1).min())
    std = period_returns.std(ddof=1)
    sharpe = float(period_returns.mean() / std * np.sqrt(12)) if std > 0 else float("nan")
    return {"total_return": total_return, "annualized": annualized, "sharpe": sharpe, "max_dd": max_dd}


def alpha_regression(port_ret, spy_ret):
    port_ret = np.asarray(port_ret, dtype=float)
    spy_ret = np.asarray(spy_ret, dtype=float)
    mask = ~np.isnan(port_ret) & ~np.isnan(spy_ret)
    X, Y = spy_ret[mask], port_ret[mask]
    n = len(X)
    beta = float(np.cov(X, Y, ddof=1)[0, 1] / np.var(X, ddof=1))
    alpha_m = float(Y.mean() - beta * X.mean())
    resid = Y - (alpha_m + beta * X)
    ssr = float(np.sum(resid ** 2))
    sst = float(np.sum((Y - Y.mean()) ** 2))
    r2 = 1 - ssr / sst if sst > 0 else float("nan")
    sigma2 = ssr / (n - 2) if n > 2 else float("nan")
    sxx = float(np.sum((X - X.mean()) ** 2))
    se_alpha = float(np.sqrt(sigma2 * (1 / n + X.mean() ** 2 / sxx))) if sxx > 0 else float("nan")
    t_alpha = alpha_m / se_alpha if se_alpha and se_alpha > 0 else float("nan")
    alpha_annualized = float((1 + alpha_m) ** 12 - 1)
    return {"beta": beta, "alpha_annualized": alpha_annualized, "r2": r2, "t_alpha": t_alpha, "n": n}


def print_perf(label, s):
    print(f"{label:<24}{s['total_return']*100:>10.2f}%{s['annualized']*100:>9.2f}%"
          f"{s['sharpe']:>9.2f}{s['max_dd']*100:>8.2f}%")


summary = {}   # check_name -> bool (alpha survives) for the final printout


# ============================================================
# CHECK 1 -- sub-period stability
# ============================================================
print()
print("=" * 96)
print("CHECK 1: Sub-period stability")
print("=" * 96)

sub_periods = {
    "2008-01..2012-12 (crisis+recovery)": (pd.Period("2008-01", "M"), pd.Period("2012-12", "M")),
    "2013-01..2018-10 (calm bull)": (pd.Period("2013-01", "M"), pd.Period("2018-10", "M")),
}

header = f"{'Book':<24}{'TotalRet':>11}{'AnnRet':>10}{'Sharpe':>9}{'MaxDD':>9}"
for label, (start_m, end_m) in sub_periods.items():
    sub_months = [m for m in all_months if start_m <= m <= end_m]
    res = run_low_vol(months=sub_months)
    if len(res) == 0:
        print(f"\n{label}: no usable periods")
        continue
    long_ret = res["long_ret"].to_numpy()
    spy_ret = res["spy_ret"].to_numpy()
    reg = alpha_regression(long_ret, spy_ret)

    print(f"\n--- {label} ({len(res)} periods) ---")
    print(header)
    print("-" * len(header))
    print_perf("Low-vol long decile", perf_stats(long_ret))
    print_perf("SPY buy&hold", perf_stats(spy_ret[~np.isnan(spy_ret)]))
    print(f"Beta: {reg['beta']:.3f}   Alpha (ann.): {reg['alpha_annualized']*100:.3f}%   "
          f"R2: {reg['r2']:.3f}   t-stat: {reg['t_alpha']:.2f}   n={reg['n']}")
    survives = (not np.isnan(reg['t_alpha'])) and reg['t_alpha'] > 2 and reg['alpha_annualized'] > 0
    summary[f"Check 1 -- {label}"] = survives
    print(f"Alpha survives in this sub-period: {survives}")


# ============================================================
# CHECK 2 -- crisis exclusion
# ============================================================
print()
print("=" * 96)
print("CHECK 2: Crisis exclusion (2008-09 through 2009-06 dropped entirely)")
print("=" * 96)

crisis_months = {pd.Period(f"2008-{m:02d}", "M") for m in range(9, 13)} | \
                {pd.Period(f"2009-{m:02d}", "M") for m in range(1, 7)}
kept_months = [m for m in all_months if m not in crisis_months]
print(f"Excluded months: {sorted(str(m) for m in crisis_months)}")

res2 = run_low_vol(months=kept_months)
long_ret2 = res2["long_ret"].to_numpy()
spy_ret2 = res2["spy_ret"].to_numpy()
reg2 = alpha_regression(long_ret2, spy_ret2)

print(f"\nPeriods remaining: {len(res2)} (of {len(all_months)} total months, "
      f"{len(all_months) - len(res2)} excluded/unusable)")
print(header)
print("-" * len(header))
print_perf("Low-vol long decile", perf_stats(long_ret2))
print_perf("SPY buy&hold", perf_stats(spy_ret2[~np.isnan(spy_ret2)]))
print(f"Beta: {reg2['beta']:.3f}   Alpha (ann.): {reg2['alpha_annualized']*100:.3f}%   "
      f"R2: {reg2['r2']:.3f}   t-stat: {reg2['t_alpha']:.2f}   n={reg2['n']}")
survives2 = (not np.isnan(reg2['t_alpha'])) and reg2['t_alpha'] > 2 and reg2['alpha_annualized'] > 0
summary["Check 2 -- crisis excluded"] = survives2
print(f"Alpha survives with the crisis excluded: {survives2}")


# ============================================================
# CHECK 3 -- sector concentration
# ============================================================
print()
print("=" * 96)
print("CHECK 3: Sector concentration of the long decile")
print("=" * 96)

res_full = run_low_vol(months=all_months)
all_tickers_in_deciles = sorted(set(t for tickers in res_full["tickers"] for t in tickers))
print(f"Unique tickers ever selected into the long decile across the practice window: "
      f"{len(all_tickers_in_deciles)}")

if os.path.exists(SECTOR_CACHE_PATH):
    with open(SECTOR_CACHE_PATH) as f:
        sector_cache = json.load(f)
else:
    sector_cache = {}

to_fetch = [t for t in all_tickers_in_deciles if t not in sector_cache]
print(f"Fetching sector for {len(to_fetch)} tickers not already cached "
      f"(cache: {SECTOR_CACHE_PATH})...", flush=True)
for i, t in enumerate(to_fetch):
    try:
        info = yf.Ticker(t).info
        sector_cache[t] = info.get("sector") or "Unknown"
    except Exception:
        sector_cache[t] = "Unknown"
    if (i + 1) % 25 == 0:
        print(f"  {i+1}/{len(to_fetch)} fetched, saving checkpoint...", flush=True)
        with open(SECTOR_CACHE_PATH, "w") as f:
            json.dump(sector_cache, f)
    time.sleep(0.3)

with open(SECTOR_CACHE_PATH, "w") as f:
    json.dump(sector_cache, f)
print(f"Sector cache saved: {len(sector_cache)} tickers total", flush=True)

sector_weight_rows = []
for _, row in res_full.iterrows():
    tickers = row["tickers"]
    n = len(tickers)
    sectors = [sector_cache.get(t, "Unknown") for t in tickers]
    vc = pd.Series(sectors).value_counts()
    weights = (vc / n).to_dict()
    weights["month"] = row["month"]
    sector_weight_rows.append(weights)

sector_df = pd.DataFrame(sector_weight_rows).set_index("month").fillna(0.0)
avg_sector_weight = sector_df.mean().sort_values(ascending=False)

print("\nAverage sector composition of the long decile (all months):")
for sector, w in avg_sector_weight.items():
    print(f"  {sector:<28}{w*100:>6.1f}%")

top_sector = avg_sector_weight.index[0]
top_weight = avg_sector_weight.iloc[0]
print(f"\nMost-weighted sector: {top_sector} ({top_weight*100:.1f}% average weight)")
concentrated = top_weight > 0.40
print(f"Concentration flag (>40% in a single sector): {concentrated}")


# ============================================================
# CHECK 4 -- position-count sensitivity
# ============================================================
print()
print("=" * 96)
print("CHECK 4: Position-count sensitivity (fixed top-N lowest-vol stocks)")
print("=" * 96)

check4_results = {}
header4 = f"{'N':<8}{'NetSharpe@20bp':>16}{'AlphaT(net)':>13}{'AlphaAnn(net)':>15}"
print(header4)
print("-" * len(header4))
for N in [10, 20, 38]:
    res_n = run_low_vol(months=all_months, long_n=N)
    gross = res_n["long_ret"].to_numpy()
    turnover = res_n["turnover"].to_numpy()
    net = gross - turnover * COST_02
    spy_ret_n = res_n["spy_ret"].to_numpy()
    net_stats = perf_stats(net)
    reg_n = alpha_regression(net, spy_ret_n)
    check4_results[N] = {"res": res_n, "net": net, "reg": reg_n}
    print(f"{N:<8}{net_stats['sharpe']:>16.2f}{reg_n['t_alpha']:>13.2f}{reg_n['alpha_annualized']*100:>14.2f}%")
    survives_n = (not np.isnan(reg_n['t_alpha'])) and reg_n['t_alpha'] > 2 and reg_n['alpha_annualized'] > 0
    summary[f"Check 4 -- top-{N} net of 0.2%"] = survives_n


# ============================================================
# CHECK 5 -- realistic small-account commission drag
# ============================================================
print()
print("=" * 96)
print(f"CHECK 5: Small-account commission drag (${ACCOUNT_VALUE_USD:,.0f} account, "
      f"${COMMISSION_PER_ORDER:.2f}/order minimum)")
print("=" * 96)

header5 = f"{'N':<8}{'AvgOrders/mo':>13}{'AvgDrag/mo':>12}{'DragAnn':>10}{'NetAnnRet':>11}{'GrossAnnRet':>13}"
print(header5)
print("-" * len(header5))
for N in [38, 20]:
    res_n = check4_results[N]["res"]
    gross = res_n["long_ret"].to_numpy()
    turnover = res_n["turnover"].to_numpy()

    # positions replaced per rebalance = turnover * N; each replacement = 1 sell + 1 buy
    orders_per_period = 2.0 * turnover * N
    commission_dollars = orders_per_period * COMMISSION_PER_ORDER
    commission_pct = commission_dollars / ACCOUNT_VALUE_USD

    net_after_commission = gross - commission_pct
    gross_stats = perf_stats(gross)
    net_stats = perf_stats(net_after_commission)

    print(f"{N:<8}{orders_per_period.mean():>13.1f}{commission_pct.mean()*100:>11.2f}%"
          f"{((1+commission_pct.mean())**12-1)*100:>9.2f}%{net_stats['annualized']*100:>10.2f}%"
          f"{gross_stats['annualized']*100:>12.2f}%")

    survives_commission = net_stats['annualized'] > 0
    summary[f"Check 5 -- top-{N} after commission drag"] = survives_commission


# ============================================================
# FINAL SUMMARY
# ============================================================
print()
print("=" * 96)
print("SUMMARY: does the low-volatility long-only alpha survive each check?")
print("=" * 96)
for check, survives in summary.items():
    print(f"  {'PASS' if survives else 'FAIL':<6} {check}")

print()
print("These are diagnostics only: no parameters were tuned, no alternate volatility")
print("lookback was tried, and nothing was optimized. The 2019+ holdout was not read.")
