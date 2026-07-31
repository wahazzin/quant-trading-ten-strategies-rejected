"""
holdout_test.py -- the ONE holdout run for this whole project. The
2019+ holdout has been sealed since data was first fetched and is being
spent here, once, on the single pre-registered configuration that
passed every practice-window check: low-volatility long-only, full
decile (~38 stocks on average), monthly rebalance, trailing 12-month
return volatility signal -- factor_test.py's factor engine, unmodified.

Reused verbatim from factor_test.py: data loading, data-quality checks,
monthly panel construction (close/open/avg_volume/ret/hist_months/
low_vol_signal/eligible), the run_factor() engine, perf_stats(), and
alpha_regression(). Only momentum is dropped (already rejected, not
re-tested per instructions) and the evaluated month range changes.

Why the full 2006-2026 file is loaded, not just 2019+: the trailing
12-month volatility signal and the 24-month-history eligibility filter
are ROLLING statistics that need genuine pre-2019 price history to be
computed correctly for early holdout months -- exactly as a live
strategy trading in January 2019 would have used 2006-2018 prices as
its lookback window. That is not "touching the holdout"; the signal
formula, decile cut, rebalance timing, and cost assumptions were all
fixed by factor_test.py and robustness_test.py before this script ever
ran, on 2008-2018 data only. What IS being spent for the first time
here is every REALIZED RETURN: every rebalance decision, entry price,
and exit price evaluated and reported below is restricted to
date >= 2019-01-01 -- enforced by only ever calling the factor engine
on holdout-period months, never on the pre-2019 months used solely as
rolling-signal history.

This is the final validation run: one configuration, one window, run
once. No alternate lookback, position count, or rebalance frequency is
tested here.
"""
import os
import numpy as np
import pandas as pd

DATA_PATH = os.path.join("data", "yf_universe.parquet")
SPY_PATH = os.path.join("data", "spy_yf.parquet")
HOLDOUT_START = pd.Timestamp("2019-01-01")
HOLDOUT_START_PERIOD = pd.Period("2019-01", "M")

PRICE_MIN, PRICE_MAX = 5.0, 100.0
MIN_ADV = 250_000
MIN_HISTORY_MONTHS = 24
COST_LEVELS_02 = 0.002
GAP_DAYS_THRESHOLD = 10
MOVE_THRESHOLD = 0.80
MIN_UNIVERSE_FOR_DECILE = 20
MIN_DECILE_SIZE_TRUSTED = 25
ACCOUNT_VALUE_USD = 4700.0
COMMISSION_PER_ORDER = 1.00
ETF_EXPENSE_RATIO_ANNUAL = 0.0030

# practice-window reference figures (already established, for side-by-side comparison)
PRACTICE = {
    "window": "2008-01 to 2018-10 (practice)",
    "sharpe_net02": 0.88,
    "alpha_annualized": 0.0423,
    "t_alpha": 2.47,
    "beta": 0.662,
}


# ============================================================
# LOAD (full range -- see docstring for why no date truncation on load)
# ============================================================
raw = pd.read_parquet(DATA_PATH)
raw["date"] = pd.to_datetime(raw["date"])
raw_spy = pd.read_parquet(SPY_PATH)
raw_spy["date"] = pd.to_datetime(raw_spy["date"])

print("=" * 96)
print("HOLDOUT WINDOW CONFIRMATION")
print("=" * 96)
print(f"Full stock panel loaded: {raw['date'].min().date()} to {raw['date'].max().date()} "
      f"({len(raw)} rows, {raw['ticker'].nunique()} tickers)")
print(f"Full SPY panel loaded: {raw_spy['date'].min().date()} to {raw_spy['date'].max().date()} "
      f"({len(raw_spy)} rows)")
print()
print("The full range is loaded (not truncated to 2019+) only so the rolling 12-month")
print("volatility signal and 24-month history filter have genuine pre-2019 lookback data,")
print("exactly as a live strategy would have had. Every REALIZED RETURN reported below --")
print(f"every rebalance, entry, and exit -- is restricted to date >= {HOLDOUT_START.date()}.")

df = raw.sort_values(["ticker", "date"]).reset_index(drop=True)
spy = raw_spy.sort_values("date").reset_index(drop=True)
del raw, raw_spy


# ============================================================
# DATA QUALITY CHECKS (verbatim from factor_test.py)
# ============================================================
n0 = len(df)
bad_price = (df[["open", "high", "low", "close"]] <= 0).any(axis=1) | df["volume"].lt(0)
n_bad_price = int(bad_price.sum())
df = df[~bad_price].copy()

df["prev_close"] = df.groupby("ticker")["close"].shift(1)
df["daily_ret"] = df["close"] / df["prev_close"] - 1
big_move = df["daily_ret"].abs() > MOVE_THRESHOLD
n_big_move = int(big_move.fillna(False).sum())
df = df[~big_move.fillna(False)].copy()
df = df.drop(columns=["prev_close", "daily_ret"])

print(f"\nData quality (full range): {n0} rows before checks, {len(df)} after "
      f"({n0 - len(df)} dropped: {n_bad_price} bad price, {n_big_move} >80% move)")


# ============================================================
# BUILD MONTHLY PANEL (verbatim from factor_test.py, full date range)
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

all_months_full = sorted(monthly["month"].unique())

# SPY monthly period-return series (identical telescoping convention, full range)
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

# ---- restrict evaluated months to the holdout window ONLY ----
all_months = [m for m in all_months_full if m >= HOLDOUT_START_PERIOD]

avg_universe_size = monthly.loc[monthly["eligible"] & monthly["month"].isin(all_months), "month"].value_counts().mean()

print()
print("=" * 96)
print("UNIVERSE (holdout window)")
print("=" * 96)
print(f"Holdout months available for rebalancing: {len(all_months)} "
      f"({all_months[0]} to {all_months[-1]})")
print(f"Average eligible universe size per month: {avg_universe_size:.1f} "
      f"(implied average decile size: {avg_universe_size/10:.1f})")


# ============================================================
# FACTOR ENGINE (verbatim from factor_test.py)
# ============================================================
def run_factor(signal_col, long_is_high_signal):
    period_records = []
    prev_ls_weights = pd.Series(dtype=float)
    prev_long_weights = pd.Series(dtype=float)

    for m in all_months:
        month_df = monthly[(monthly["month"] == m) & monthly["eligible"]]
        month_df = month_df.dropna(subset=[signal_col, "next_open", "next2_open"])
        if len(month_df) < MIN_UNIVERSE_FOR_DECILE:
            continue

        month_df = month_df.sort_values(signal_col)
        decile_n = max(1, len(month_df) // 10)

        if long_is_high_signal:
            short_df = month_df.iloc[:decile_n]
            long_df = month_df.iloc[-decile_n:]
        else:
            long_df = month_df.iloc[:decile_n]
            short_df = month_df.iloc[-decile_n:]

        long_ret = long_df["next2_open"] / long_df["next_open"] - 1
        short_ret = short_df["next2_open"] / short_df["next_open"] - 1
        bench_ret = month_df["next2_open"] / month_df["next_open"] - 1

        w_ls = pd.Series(0.0, index=month_df["ticker"].to_numpy())
        w_ls.loc[long_df["ticker"]] = 1.0 / len(long_df)
        w_ls.loc[short_df["ticker"]] = -1.0 / len(short_df)
        idx_ls = w_ls.index.union(prev_ls_weights.index)
        turnover_ls = 0.5 * float((w_ls.reindex(idx_ls, fill_value=0.0) -
                                    prev_ls_weights.reindex(idx_ls, fill_value=0.0)).abs().sum())
        prev_ls_weights = w_ls

        w_long = pd.Series(0.0, index=long_df["ticker"].to_numpy())
        w_long.loc[:] = 1.0 / len(long_df)
        idx_l = w_long.index.union(prev_long_weights.index)
        turnover_long = 0.5 * float((w_long.reindex(idx_l, fill_value=0.0) -
                                      prev_long_weights.reindex(idx_l, fill_value=0.0)).abs().sum())
        prev_long_weights = w_long

        period_records.append({
            "month": m,
            "n_universe": len(month_df), "n_long": len(long_df), "n_short": len(short_df),
            "spread": float(long_ret.mean() - short_ret.mean()),
            "long_ret": float(long_ret.mean()), "short_ret": float(short_ret.mean()),
            "bench_ret": float(bench_ret.mean()),
            "turnover_ls": turnover_ls,
            "turnover_long": turnover_long,
        })

    res = pd.DataFrame(period_records)
    if len(res):
        res["spy_ret"] = res["month"].map(SPY_MONTHLY)
    return res


def perf_stats(period_returns, turnovers=None, cost=0.0):
    period_returns = np.asarray(period_returns, dtype=float)
    if turnovers is None:
        turnovers = np.zeros(len(period_returns))
    net = period_returns - turnovers * cost
    equity = np.cumprod(1 + net)
    n = len(net)
    total_return = float(equity[-1] - 1)
    annualized = float(equity[-1] ** (12.0 / n) - 1) if n > 0 else float("nan")
    running_max = np.maximum.accumulate(equity)
    max_dd = float((equity / running_max - 1).min())
    std = net.std(ddof=1)
    sharpe = float(net.mean() / std * np.sqrt(12)) if std > 0 else float("nan")
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
    print(f"{label:<30}{s['total_return']*100:>10.2f}%{s['annualized']*100:>9.2f}%"
          f"{s['sharpe']:>9.2f}{s['max_dd']*100:>8.2f}%")


# ============================================================
# RUN THE PRE-REGISTERED CONFIGURATION: low-vol long-only, full decile
# ============================================================
print()
print("=" * 96)
print("HOLDOUT RESULT: Low volatility, long-only, full decile, monthly rebalance")
print("=" * 96)

res = run_factor("low_vol_signal", long_is_high_signal=False)
n_periods = len(res)

if n_periods == 0:
    raise SystemExit("No usable holdout rebalance periods -- cannot report a result.")

avg_decile = res["n_long"].mean()
print(f"Monthly rebalance periods: {n_periods} ({res['month'].iloc[0]} to {res['month'].iloc[-1]})")
print(f"*** AVERAGE STOCKS PER DECILE: {avg_decile:.1f} ***")
if avg_decile < MIN_DECILE_SIZE_TRUSTED:
    print(f"*** WARNING: {avg_decile:.1f} is BELOW the {MIN_DECILE_SIZE_TRUSTED}-stock trust threshold. ***")
else:
    print(f"Decile size clears the {MIN_DECILE_SIZE_TRUSTED}-stock bar.")

long_ret = res["long_ret"].to_numpy()
bench_ret = res["bench_ret"].to_numpy()
spy_ret = res["spy_ret"].to_numpy()
turnover_long = res["turnover_long"].to_numpy()

# commission drag: orders = 2 * turnover * n_long, $1/order, on a $4,700 account
orders = 2.0 * turnover_long * res["n_long"].to_numpy()
commission_pct = orders * COMMISSION_PER_ORDER / ACCOUNT_VALUE_USD
net02 = long_ret - turnover_long * COST_LEVELS_02
net_full = net02 - commission_pct

print()
header = f"{'Variant':<30}{'TotalRet':>11}{'AnnRet':>10}{'Sharpe':>9}{'MaxDD':>9}"
print(header)
print("-" * len(header))
print_perf("Long decile -- gross", perf_stats(long_ret))
print_perf("Long decile -- net 0.2% RT", perf_stats(net02))
print_perf("Long decile -- net 0.2%+commission", perf_stats(net_full))

spy_valid = spy_ret[~np.isnan(spy_ret)]
etf_monthly_drag = (1 + ETF_EXPENSE_RATIO_ANNUAL) ** (1 / 12) - 1
etf_proxy = long_ret - etf_monthly_drag

print()
print_perf("SPY buy&hold", perf_stats(spy_valid))
print_perf("Equal-weighted universe", perf_stats(bench_ret))
print_perf("Min-vol ETF proxy (0.30% ER)", perf_stats(etf_proxy))

print()
print("--- Formal alpha test: net-of-0.2% long decile returns regressed on SPY monthly returns ---")
reg = alpha_regression(net02, spy_ret)
print(f"Beta:                 {reg['beta']:.3f}")
print(f"Alpha (annualized):   {reg['alpha_annualized']*100:.3f}%")
print(f"R-squared:            {reg['r2']:.3f}")
print(f"t-stat on alpha:      {reg['t_alpha']:.2f}   (n={reg['n']} monthly observations)")


# ============================================================
# PRACTICE VS HOLDOUT SIDE BY SIDE
# ============================================================
print()
print("=" * 96)
print("PRACTICE WINDOW vs HOLDOUT WINDOW -- side by side")
print("=" * 96)
cmp_header = f"{'Window':<32}{'Sharpe (net 0.2%)':>18}{'Alpha (ann.)':>14}{'t-stat':>9}{'Beta':>8}"
print(cmp_header)
print("-" * len(cmp_header))
holdout_sharpe_net02 = perf_stats(net02)["sharpe"]
print(f"{PRACTICE['window']:<32}{PRACTICE['sharpe_net02']:>18.2f}"
      f"{PRACTICE['alpha_annualized']*100:>13.2f}%{PRACTICE['t_alpha']:>9.2f}{PRACTICE['beta']:>8.3f}")
print(f"{'2019-01 to ' + str(res['month'].iloc[-1]) + ' (holdout)':<32}{holdout_sharpe_net02:>18.2f}"
      f"{reg['alpha_annualized']*100:>13.2f}%{reg['t_alpha']:>9.2f}{reg['beta']:>8.3f}")


# ============================================================
# PLAIN-ENGLISH CONCLUSION
# ============================================================
print()
print("=" * 96)
print("CONCLUSION")
print("=" * 96)

practice_significant = PRACTICE["t_alpha"] > 2 and PRACTICE["alpha_annualized"] > 0
holdout_significant = (not np.isnan(reg["t_alpha"])) and reg["t_alpha"] > 2 and reg["alpha_annualized"] > 0

if holdout_significant and reg["alpha_annualized"] >= PRACTICE["alpha_annualized"] * 0.5:
    verdict = "PERSISTED"
elif reg["alpha_annualized"] > 0 and reg["t_alpha"] > 0:
    verdict = "WEAKENED"
else:
    verdict = "DISAPPEARED"

print(f"Practice window: alpha {PRACTICE['alpha_annualized']*100:.2f}%/yr, t={PRACTICE['t_alpha']:.2f}, "
      f"beta {PRACTICE['beta']:.3f}, net Sharpe {PRACTICE['sharpe_net02']:.2f} "
      f"({'significant' if practice_significant else 'not significant'}).")
print(f"Holdout window:  alpha {reg['alpha_annualized']*100:.2f}%/yr, t={reg['t_alpha']:.2f}, "
      f"beta {reg['beta']:.3f}, net Sharpe {holdout_sharpe_net02:.2f} "
      f"({'significant' if holdout_significant else 'not significant'}).")
print()
print(f"VERDICT: the effect {verdict} out of sample.")

print()
print("=" * 96)
print("This is the final validation run on the single pre-registered configuration.")
print("No alternate lookback, position count, or rebalance frequency was tested.")
print("No further iteration follows this result.")
