"""
factor_test.py -- momentum 12-1 and low-volatility factor tests on the
yfinance-sourced, randomly-sampled US common-stock panel
(data/yf_universe.parquet: 2,500 tickers drawn via random.Random(42)
from the current NASDAQ + otherlisted common-stock universe -- see
fetch_yf_universe.py). This is a re-run of an earlier 600-ticker pilot
that fetched cleanly but averaged only 9.3 stocks per decile after
filters -- too few to trust. 2,500 tickers exists specifically to fix
that diversification problem.

Pre-registered spec (declared before this data was fetched, per chat):
for BOTH factors, report both the long/short dollar-neutral formulation
AND the long-only formulation, plus an SPY benchmark and a formal alpha
test. This is also a deliberate fix to a second flaw the pilot exposed:
a dollar-neutral spread with ~9% monthly return std suffers real
volatility drag that can make a genuine long-only effect (low
volatility, in the literature harvested long-only) look like a losing
strategy when forced into a long/short spread. Reporting both
formulations side by side is how that gets resolved instead of guessed at.

Known limitation, accepted deliberately (see chat): this universe is
CURRENTLY LISTED tickers only, so it carries survivorship bias. For
momentum this is conservative (the worst historical losers we would
short are absent from the data entirely, understating the long/short
edge). For low volatility the effect is believed to be minimal. This is
exactly why long-term reversal is NOT tested here -- survivorship bias
would fatally inflate that factor (past losers that recovered survive
into the currently-listed universe; past losers that went to zero do not).

Evidence boundary: practice window = earliest available date through
2018-12-31. Holdout = 2019-01-01 onward. The raw daily panel (both the
stock universe and SPY) is truncated to date < 2019-01-01 IMMEDIATELY
after loading, before any other computation, and the untruncated
reference is deleted -- nothing past that boundary can enter any
calculation below.

Execution: signal from month-end close, position entered at next
month's open, exited at the following month's open (telescoping monthly
rebalance -- each month's exit price is the next signal's entry price).
Equal-weighted deciles; SPY's own monthly return is computed on the
identical entry/exit convention so equity curves and the alpha
regression compare like with like.
"""
import os
import numpy as np
import pandas as pd

DATA_PATH = os.path.join("data", "yf_universe.parquet")
SPY_PATH = os.path.join("data", "spy_yf.parquet")
HOLDOUT_START = pd.Timestamp("2019-01-01")

PRICE_MIN, PRICE_MAX = 5.0, 100.0
MIN_ADV = 250_000
MIN_HISTORY_MONTHS = 24
COST_LEVELS = [0.001, 0.002, 0.004]
GAP_DAYS_THRESHOLD = 10   # calendar days between consecutive rows -> flagged as a long gap
MOVE_THRESHOLD = 0.80     # abs daily return beyond this -> dropped as a likely bad/unadjusted row
MIN_UNIVERSE_FOR_DECILE = 20   # need at least this many eligible stocks to form 10 deciles sensibly
MIN_DECILE_SIZE_TRUSTED = 25   # below this, say so loudly


# ============================================================
# LOAD + HOLDOUT BOUNDARY
# ============================================================
raw = pd.read_parquet(DATA_PATH)
raw["date"] = pd.to_datetime(raw["date"])
raw_spy = pd.read_parquet(SPY_PATH)
raw_spy["date"] = pd.to_datetime(raw_spy["date"])

print("=" * 96)
print("HOLDOUT CONFIRMATION")
print("=" * 96)
print(f"Raw stock panel on disk spans: {raw['date'].min().date()} to {raw['date'].max().date()} "
      f"({len(raw)} rows, {raw['ticker'].nunique()} tickers)")
print(f"Raw SPY panel on disk spans: {raw_spy['date'].min().date()} to {raw_spy['date'].max().date()} "
      f"({len(raw_spy)} rows)")

practice = raw[raw["date"] < HOLDOUT_START].copy()
n_dropped_holdout = len(raw) - len(practice)
del raw
spy_practice = raw_spy[raw_spy["date"] < HOLDOUT_START].copy()
n_dropped_spy = len(raw_spy) - len(spy_practice)
del raw_spy   # both untruncated references are gone; nothing below can read post-2018 data

print(f"Practice window kept: date < {HOLDOUT_START.date()}  "
      f"(stocks: {len(practice)} rows, SPY: {len(spy_practice)} rows)")
print(f"Rows excluded as holdout: stocks {n_dropped_holdout}, SPY {n_dropped_spy}")
print(f"Maximum date present in any DataFrame from this point forward: "
      f"{max(practice['date'].max(), spy_practice['date'].max()).date()}")
print("CONFIRMED: the post-2018 holdout was not read by this script beyond the initial date check above.")

df = practice.sort_values(["ticker", "date"]).reset_index(drop=True)
spy = spy_practice.sort_values("date").reset_index(drop=True)


# ============================================================
# DATA QUALITY CHECKS (practice window only)
# ============================================================
print()
print("=" * 96)
print("DATA QUALITY CHECKS (practice window only)")
print("=" * 96)

n0 = len(df)
bad_price = (df[["open", "high", "low", "close"]] <= 0).any(axis=1) | df["volume"].lt(0)
n_bad_price = int(bad_price.sum())
df = df[~bad_price].copy()

df["prev_close"] = df.groupby("ticker")["close"].shift(1)
df["daily_ret"] = df["close"] / df["prev_close"] - 1
big_move = df["daily_ret"].abs() > MOVE_THRESHOLD
n_big_move = int(big_move.fillna(False).sum())
df = df[~big_move.fillna(False)].copy()

df["prev_date"] = df.groupby("ticker")["date"].shift(1)
df["gap_days"] = (df["date"] - df["prev_date"]).dt.days
n_long_gaps = int((df["gap_days"] > GAP_DAYS_THRESHOLD).sum())

print(f"Rows before quality checks: {n0}")
print(f"Dropped -- zero/negative price or negative volume: {n_bad_price}")
print(f"Dropped -- single-day move > {MOVE_THRESHOLD*100:.0f}% (possible unadjusted split / bad tick): {n_big_move}")
print(f"Flagged, not dropped -- gaps > {GAP_DAYS_THRESHOLD} calendar days between consecutive rows: {n_long_gaps}")
print(f"Rows remaining: {len(df)} ({n0 - len(df)} dropped total, "
      f"{(n0 - len(df)) / n0 * 100:.3f}% of pre-check rows)")

df = df.drop(columns=["prev_close", "daily_ret", "prev_date", "gap_days"])


# ============================================================
# BUILD MONTHLY PANEL (stocks)
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

monthly["close_lag1"] = g["close"].shift(1)
monthly["close_lag12"] = g["close"].shift(12)
valid_lag1 = g["month"].shift(1) == (monthly["month"] - 1)
valid_lag12 = g["month"].shift(12) == (monthly["month"] - 12)
mom_12_1 = monthly["close_lag1"] / monthly["close_lag12"] - 1
monthly["mom_12_1"] = np.where(valid_lag1 & valid_lag12, mom_12_1, np.nan)

monthly["low_vol_signal"] = g["ret"].transform(lambda s: s.rolling(12).std())

monthly["eligible"] = (
    (monthly["close"] >= PRICE_MIN) & (monthly["close"] <= PRICE_MAX) &
    (monthly["avg_volume"] > MIN_ADV) &
    (monthly["hist_months"] >= MIN_HISTORY_MONTHS)
)

all_months = sorted(monthly["month"].unique())
avg_universe_size = monthly.loc[monthly["eligible"], "month"].value_counts().mean()

print()
print("=" * 96)
print("UNIVERSE")
print("=" * 96)
print(f"Monthly panel: {len(monthly)} (ticker, month) rows, {monthly['ticker'].nunique()} tickers, "
      f"{len(all_months)} months ({all_months[0]} to {all_months[-1]})")
print(f"Filters: close ${PRICE_MIN}-${PRICE_MAX}, monthly avg daily volume > {MIN_ADV:,}, "
      f"{MIN_HISTORY_MONTHS}+ months prior history")
print(f"Average eligible universe size per month: {avg_universe_size:.1f} "
      f"(implied average decile size: {avg_universe_size/10:.1f})")


# ============================================================
# BUILD SPY MONTHLY PERIOD-RETURN SERIES (same telescoping convention)
# ============================================================
spy["month"] = spy["date"].dt.to_period("M")
spy_open_first = spy.groupby("month")["open"].first().reset_index()
spy_open_first = spy_open_first.sort_values("month").reset_index(drop=True)
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
# FACTOR ENGINE
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

        # dollar-neutral long/short turnover
        w_ls = pd.Series(0.0, index=month_df["ticker"].to_numpy())
        w_ls.loc[long_df["ticker"]] = 1.0 / len(long_df)
        w_ls.loc[short_df["ticker"]] = -1.0 / len(short_df)
        idx_ls = w_ls.index.union(prev_ls_weights.index)
        turnover_ls = 0.5 * float((w_ls.reindex(idx_ls, fill_value=0.0) -
                                    prev_ls_weights.reindex(idx_ls, fill_value=0.0)).abs().sum())
        prev_ls_weights = w_ls

        # long-only turnover (separate book, separate churn)
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
    mask = ~np.isnan(port_ret) & ~np.isnan(spy_ret)
    X = spy_ret[mask]
    Y = port_ret[mask]
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


def print_stats_table(rows, header_label="Variant"):
    header = f"{header_label:<14}{'TotalRet':>11}{'AnnRet':>10}{'Sharpe':>9}{'MaxDD':>9}"
    print(header)
    print("-" * len(header))
    for label, s in rows:
        print(f"{label:<14}{s['total_return']*100:>10.2f}%{s['annualized']*100:>9.2f}%"
              f"{s['sharpe']:>9.2f}{s['max_dd']*100:>8.2f}%")


def report_factor(name, signal_col, long_is_high_signal):
    print()
    print("=" * 96)
    print(f"FACTOR: {name}")
    print("=" * 96)

    res = run_factor(signal_col, long_is_high_signal)
    n_periods = len(res)
    if n_periods == 0:
        print("No usable rebalance periods -- skipping.")
        return

    spread = res["spread"].to_numpy()
    turnover_ls = res["turnover_ls"].to_numpy()
    turnover_long = res["turnover_long"].to_numpy()
    long_ret = res["long_ret"].to_numpy()
    short_ret = res["short_ret"].to_numpy()
    bench_ret = res["bench_ret"].to_numpy()
    spy_ret = res["spy_ret"].to_numpy()

    avg_decile = res["n_long"].mean()
    print(f"Monthly rebalance periods: {n_periods} ({res['month'].iloc[0]} to {res['month'].iloc[-1]})")
    print(f"*** AVERAGE STOCKS PER DECILE: {avg_decile:.1f} ***")
    if avg_decile < MIN_DECILE_SIZE_TRUSTED:
        print(f"*** WARNING: {avg_decile:.1f} is BELOW the {MIN_DECILE_SIZE_TRUSTED}-stock trust threshold "
              f"set for this run -- treat this factor's result with corresponding caution. ***")
    else:
        print(f"Decile size clears the {MIN_DECILE_SIZE_TRUSTED}-stock bar set for this run.")

    mean_spread = float(spread.mean())
    std_spread = float(spread.std(ddof=1))
    t_stat = mean_spread / (std_spread / np.sqrt(n_periods)) if std_spread > 0 else float("nan")

    print()
    print("--- Long/short dollar-neutral ---")
    print(f"Average turnover per rebalance: {turnover_ls.mean()*100:.1f}%")
    print(f"Pooled spread t-stat: {t_stat:.2f}  (mean {mean_spread*100:.3f}%, "
          f"std {std_spread*100:.3f}%, n={n_periods})")
    print()
    print_stats_table([(label, perf_stats(spread, turnover_ls, cost)) for label, cost in
                        [("gross", 0.0)] + [(f"net_{c*1000:.0f}bp", c) for c in COST_LEVELS]])

    print()
    print("--- Long-only decile ---")
    print(f"Average turnover per rebalance: {turnover_long.mean()*100:.1f}%")
    print()
    print_stats_table([(label, perf_stats(long_ret, turnover_long, cost)) for label, cost in
                        [("gross", 0.0)] + [(f"net_{c*1000:.0f}bp", c) for c in COST_LEVELS]])

    print()
    print("--- Benchmark comparison: long-only decile vs equal-weighted universe vs SPY (gross) ---")
    print_stats_table([
        ("Long decile", perf_stats(long_ret)),
        ("Universe EW", perf_stats(bench_ret)),
        ("SPY buy&hold", perf_stats(spy_ret[~np.isnan(spy_ret)])),
    ], header_label="Book")

    print()
    print("--- Formal alpha test: long-only decile monthly returns regressed on SPY monthly returns ---")
    reg = alpha_regression(long_ret, spy_ret)
    print(f"Beta:                 {reg['beta']:.3f}")
    print(f"Alpha (annualized):   {reg['alpha_annualized']*100:.3f}%")
    print(f"R-squared:            {reg['r2']:.3f}")
    print(f"t-stat on alpha:      {reg['t_alpha']:.2f}   (n={reg['n']} monthly observations)")
    if not np.isnan(reg['t_alpha']) and reg['t_alpha'] > 2 and reg['alpha_annualized'] > 0:
        print("-> Alpha is positive and statistically significant (t>2): not fully explained by SPY beta.")
    else:
        print("-> Alpha is NOT both positive and statistically significant: consistent with this being")
        print("   leveraged/adjusted market exposure rather than demonstrated skill.")


report_factor("Momentum 12-1 (long top decile, short bottom decile)", "mom_12_1", long_is_high_signal=True)
report_factor("Low volatility (long bottom decile / lowest vol, short top decile)", "low_vol_signal", long_is_high_signal=False)

print()
print("=" * 96)
print("This is a diagnostic run of two pre-registered specs only: no parameters were tuned,")
print("no lookbacks or decile cuts were varied, and no configuration was selected from results.")
print("Long-term reversal was deliberately NOT tested -- survivorship bias in this")
print("currently-listed-only universe would fatally inflate that factor.")
