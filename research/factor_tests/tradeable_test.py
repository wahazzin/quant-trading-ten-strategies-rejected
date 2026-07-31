"""
tradeable_test.py -- does cutting rebalance frequency fix the low-vol
strategy's economics? robustness_test.py found the effect real (crisis-
excluded t=3.18) but commission-heavy at monthly rebalance: 4.04%/yr
drag on a $4,700 account at 38 positions, netting 6.20% vs SPY's 8.97%.

Pre-registered spec (one configuration, no variants): same trailing
12-month volatility signal, 20 lowest-vol eligible stocks, equal-
weighted, but rebalanced QUARTERLY (Jan/Apr/Jul/Oct only) instead of
monthly -- selection changes 4x/year instead of 12x/year, cutting
trade-driven costs roughly 3x, while still measuring and reporting
MONTHLY returns (not quarterly) so the t-test keeps ~130 observations.

Data loading, the universe filters, and the monthly panel construction
are reused verbatim from robustness_test.py (same columns: close, open,
avg_volume, low_vol_signal, eligible, hist_months). The quarterly
selection/holding engine below is new by necessity -- robustness_test.py's
run_low_vol() was built for non-overlapping MONTHLY telescoping only, and
a genuinely different rebalance frequency needs different simulation
logic: between rebalances, the same 20 names are held with weights that
DRIFT (no monthly re-equalization -- that would just be monthly trading
again, defeating the point), and trading costs apply ONLY in the
rebalance month itself, not the two hold-only months of each quarter.
perf_stats() and alpha_regression() are reused verbatim.

Holdout (2019-01-01 onward) stays sealed -- truncated immediately after
loading, exactly as in factor_test.py and robustness_test.py.
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
MOVE_THRESHOLD = 0.80
MIN_UNIVERSE_FOR_DECILE = 20

N_STOCKS = 20
REBALANCE_ENTRY_MONTHS = {1, 4, 7, 10}   # January, April, July, October
COST_ROUNDTRIP = 0.002
ACCOUNT_VALUE_USD = 4700.0
COMMISSION_PER_ORDER = 1.00
ETF_EXPENSE_RATIO_ANNUAL = 0.0030


# ============================================================
# LOAD + HOLDOUT BOUNDARY (verbatim from robustness_test.py)
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
# DATA QUALITY CHECKS (verbatim from robustness_test.py)
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
# BUILD MONTHLY PANEL (verbatim from robustness_test.py)
# ============================================================
df["month"] = df["date"].dt.to_period("M")

close_last = df.sort_values("date").groupby(["ticker", "month"])["close"].last()
open_first = df.sort_values("date").groupby(["ticker", "month"])["open"].first()
vol_mean = df.groupby(["ticker", "month"])["volume"].mean()

monthly = pd.DataFrame({"close": close_last, "open": open_first, "avg_volume": vol_mean}).reset_index()
monthly = monthly.sort_values(["ticker", "month"]).reset_index(drop=True)

g = monthly.groupby("ticker")
monthly["ret"] = g["close"].pct_change()
monthly["hist_months"] = g.cumcount()

monthly["low_vol_signal"] = g["ret"].transform(lambda s: s.rolling(12).std())

monthly["eligible"] = (
    (monthly["close"] >= PRICE_MIN) & (monthly["close"] <= PRICE_MAX) &
    (monthly["avg_volume"] > MIN_ADV) &
    (monthly["hist_months"] >= MIN_HISTORY_MONTHS)
)

all_months = sorted(monthly["month"].unique())
print(f"Monthly panel: {monthly['ticker'].nunique()} tickers, {len(all_months)} months "
      f"({all_months[0]} to {all_months[-1]})")

# fast (ticker, month) -> open lookup for the quarterly engine
open_lookup = monthly.set_index(["ticker", "month"])["open"]


# ============================================================
# SPY simple monthly return series: r(m) = open(m+1)/open(m) - 1,
# labeled at m (the month whose open STARTS the one-month hold) --
# a fresh, single-step series distinct from robustness_test.py's
# SPY_MONTHLY (which used the 2-step signal->entry->exit telescoping
# convention for the monthly-rebalanced engine; this script needs
# every individual month's own return, not that).
# ============================================================
spy["month"] = spy["date"].dt.to_period("M")
spy_open = spy.groupby("month")["open"].first().reset_index().sort_values("month").reset_index(drop=True)
spy_open["next_open"] = spy_open["open"].shift(-1)
spy_open["next_month"] = spy_open["month"].shift(-1)
spy_valid = spy_open["next_month"] == (spy_open["month"] + 1)
spy_open.loc[~spy_valid, "next_open"] = np.nan
spy_open["simple_ret"] = spy_open["next_open"] / spy_open["open"] - 1
SPY_SIMPLE_MONTHLY = spy_open.set_index("month")["simple_ret"]


# ============================================================
# QUARTERLY LOW-VOL ENGINE
# ============================================================
def drifting_quarter_returns(stock_rets):
    """stock_rets: list of 1D arrays, one per month in the quarter, each
    holding the selected stocks' simple monthly returns (same stocks,
    same order). Weights start equal and drift (no monthly rebalancing).
    Returns the portfolio's return for each month plus the final
    (drifted) weight vector, needed as the next quarter's pre-trade
    weights for turnover."""
    n = len(stock_rets[0])
    weights = np.full(n, 1.0 / n)
    port_rets = []
    for r in stock_rets:
        port_rets.append(float(np.sum(weights * r)))
        grown = weights * (1 + r)
        weights = grown / grown.sum()
    return port_rets, weights


signal_months = [m for m in all_months if (m + 1).month in REBALANCE_ENTRY_MONTHS]

quarter_records = []
prev_weights = pd.Series(dtype=float)   # ticker -> weight, drifted from the prior quarter

for m_signal in signal_months:
    m1, m2, m3, m4 = m_signal + 1, m_signal + 2, m_signal + 3, m_signal + 4

    sig_df = monthly[(monthly["month"] == m_signal) & monthly["eligible"]]
    sig_df = sig_df.dropna(subset=["low_vol_signal"])
    if len(sig_df) < MIN_UNIVERSE_FOR_DECILE:
        continue

    picks = sig_df.sort_values("low_vol_signal").iloc[:N_STOCKS]["ticker"].tolist()

    opens = {}
    usable = []
    for t in picks:
        vals = [open_lookup.get((t, mm), np.nan) for mm in (m1, m2, m3, m4)]
        if all(pd.notna(v) for v in vals):
            opens[t] = vals
            usable.append(t)
    if len(usable) < MIN_UNIVERSE_FOR_DECILE // 2:   # data-availability guard, not a selection criterion
        continue

    o = np.array([opens[t] for t in usable])   # shape (n_usable, 4): [o1,o2,o3,o4] per stock
    r1 = o[:, 1] / o[:, 0] - 1
    r2 = o[:, 2] / o[:, 1] - 1
    r3 = o[:, 3] / o[:, 2] - 1

    port_rets, end_weights = drifting_quarter_returns([r1, r2, r3])

    new_target = pd.Series(1.0 / len(usable), index=usable)
    idx = new_target.index.union(prev_weights.index)
    turnover = 0.5 * float((new_target.reindex(idx, fill_value=0.0) -
                            prev_weights.reindex(idx, fill_value=0.0)).abs().sum())
    prev_weights = pd.Series(end_weights, index=usable)

    orders = 2.0 * turnover * len(usable)
    commission_pct = orders * COMMISSION_PER_ORDER / ACCOUNT_VALUE_USD
    cost_pct_02 = turnover * COST_ROUNDTRIP

    quarter_records.append({
        "m1": m1, "m2": m2, "m3": m3,
        "gross": port_rets, "n_stocks": len(usable),
        "turnover": turnover, "orders": orders,
        "cost_02": cost_pct_02, "commission_pct": commission_pct,
    })

print(f"\nQuarters processed: {len(quarter_records)} "
      f"({quarter_records[0]['m1']} to {quarter_records[-1]['m3']})")


# ============================================================
# ASSEMBLE MONTHLY SERIES: gross, net-of-0.2%, fully-net (0.2% + commission)
# ============================================================
months_out, gross_out, net02_out, netfull_out = [], [], [], []
turnovers_out, orders_out, commission_pct_out = [], [], []

for q in quarter_records:
    g1, g2, g3 = q["gross"]
    cost02 = q["cost_02"]
    costfull = q["cost_02"] + q["commission_pct"]

    months_out.extend([q["m1"], q["m2"], q["m3"]])
    gross_out.extend([g1, g2, g3])
    net02_out.extend([g1 - cost02, g2, g3])
    netfull_out.extend([g1 - costfull, g2, g3])

    turnovers_out.append(q["turnover"])
    orders_out.append(q["orders"])
    commission_pct_out.append(q["commission_pct"])

gross_arr = np.array(gross_out)
net02_arr = np.array(net02_out)
netfull_arr = np.array(netfull_out)
spy_aligned = np.array([SPY_SIMPLE_MONTHLY.get(m, np.nan) for m in months_out])


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
    print(f"{label:<28}{s['total_return']*100:>10.2f}%{s['annualized']*100:>9.2f}%"
          f"{s['sharpe']:>9.2f}{s['max_dd']*100:>8.2f}%")


# ============================================================
# REPORT
# ============================================================
print()
print("=" * 96)
print(f"QUARTERLY-REBALANCED LOW-VOL, {N_STOCKS} STOCKS -- monthly returns, "
      f"costs applied only at rebalance")
print("=" * 96)

n_periods = len(gross_out)
avg_turnover = float(np.mean(turnovers_out))
avg_orders_per_quarter = float(np.mean(orders_out))
avg_orders_per_year = avg_orders_per_quarter * 4
avg_commission_pct_per_quarter = float(np.mean(commission_pct_out))
commission_drag_annualized = (1 + avg_commission_pct_per_quarter) ** 4 - 1

print(f"Monthly observations: {n_periods}  |  Quarters rebalanced: {len(quarter_records)}")
print(f"Average turnover per rebalance: {avg_turnover*100:.1f}%")
print(f"Average orders per year: {avg_orders_per_year:.1f}")
print(f"Annualized commission drag: {commission_drag_annualized*100:.2f}%")

print()
header = f"{'Variant':<28}{'TotalRet':>11}{'AnnRet':>10}{'Sharpe':>9}{'MaxDD':>9}"
print(header)
print("-" * len(header))
print_perf("Gross", perf_stats(gross_arr))
print_perf("Net of 0.2% round-trip", perf_stats(net02_arr))
print_perf("Fully net (0.2% + commissions)", perf_stats(netfull_arr))

print()
print("--- Formal alpha test on the FULLY-NET monthly return series vs SPY ---")
reg = alpha_regression(netfull_arr, spy_aligned)
print(f"Beta:                 {reg['beta']:.3f}")
print(f"Alpha (annualized):   {reg['alpha_annualized']*100:.3f}%")
print(f"R-squared:            {reg['r2']:.3f}")
print(f"t-stat on alpha:      {reg['t_alpha']:.2f}   (n={reg['n']} monthly observations)")
survives = (not np.isnan(reg['t_alpha'])) and reg['t_alpha'] > 2 and reg['alpha_annualized'] > 0
print(f"Alpha survives (t>2 and positive): {survives}")


# ============================================================
# COMPARISON: fully-net strategy vs SPY vs hypothetical min-vol ETF
# ============================================================
print()
print("=" * 96)
print("COMPARISON: fully-net DIY bot vs SPY buy&hold vs hypothetical min-vol ETF proxy")
print("=" * 96)
print(f"ETF proxy = identical 20-stock quarterly low-vol book, but a flat "
      f"{ETF_EXPENSE_RATIO_ANNUAL*100:.2f}%/yr expense ratio replaces all turnover/commission costs "
      f"(zero trading costs assumed, as in a real fund wrapper).")

monthly_expense_drag = (1 + ETF_EXPENSE_RATIO_ANNUAL) ** (1 / 12) - 1
etf_arr = gross_arr - monthly_expense_drag

spy_valid_mask = ~np.isnan(spy_aligned)
print()
print(header)
print("-" * len(header))
print_perf("DIY bot (fully net)", perf_stats(netfull_arr))
print_perf("SPY buy&hold", perf_stats(spy_aligned[spy_valid_mask]))
print_perf("Min-vol ETF proxy (0.30% ER)", perf_stats(etf_arr))

diy_stats = perf_stats(netfull_arr)
etf_stats = perf_stats(etf_arr)
print()
if diy_stats["annualized"] > etf_stats["annualized"]:
    print(f"The DIY bot beats the hypothetical min-vol ETF proxy on annualized return "
          f"({diy_stats['annualized']*100:.2f}% vs {etf_stats['annualized']*100:.2f}%) "
          f"despite its own trading costs.")
else:
    print(f"The DIY bot does NOT beat the hypothetical min-vol ETF proxy on annualized return "
          f"({diy_stats['annualized']*100:.2f}% vs {etf_stats['annualized']*100:.2f}%) -- "
          f"a real min-vol ETF's economies of scale on trading costs would likely win here.")

print()
print("=" * 96)
print("This is a single pre-registered configuration, run once: no other rebalance")
print("frequency, position count, or volatility lookback was tested or compared.")
print("The 2019+ holdout was not read anywhere in this script.")
