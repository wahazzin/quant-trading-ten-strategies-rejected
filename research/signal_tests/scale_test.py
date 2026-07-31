"""
scale_test.py -- re-run the decisive market-neutral cross-sectional
test (quantile_test.py's Analysis B, which failed gross on the
13-stock universe: Sharpe 0.04/-0.46/-0.17/-0.29 across rebalance
frequencies) at much larger scale, on the universe expand_universe.py
built. Diagnosis under test: was the earlier failure a genuine null
result, or just an underpowered sample (13 stocks, as few as 25-500
rebalance periods)?

Methodology is reused exactly from quantile_test.py's Analysis B:
20-day close z-score (fixed, not tuned), rank cross-sectionally each
rebalance day, long the bottom decile / short the top decile,
equal-weighted, dollar-neutral, non-overlapping rebalances at
1/5/10/20-day frequencies. Gross and net (0.1%/0.2%/0.4% round-trip)
total return, annualized return, Sharpe, max drawdown, turnover -- plus
the pooled t-statistic on the period-level long/short spread returns
(new here; quantile_test.py's Analysis B didn't compute one).

The only change from the 13-stock version, other than universe size, is
book sizing: with ~13 stocks "3 most oversold / 3 most overbought" was
already close to a decile cut; at 100+ stocks a literal decile (~10%)
replaces the fixed count of 3, exactly as instructed.

Universe = the union of every ticker marked status == "PASS" in
data/universe_verification.csv (the original 13-stock verification run)
and data/universe_expanded.csv (expand_universe.py's mechanical filter
pass), restricted to tickers that actually have a CSV on disk. This is
NOT a blind glob of data/*_daily.csv: both verification scripts save a
stock's CSV as soon as it has enough bars, before checking price/ADV --
so data/ also holds leftover files for tickers that FAILED verification
(e.g. TDS, KMT, MTDR, RCUS, JBLU, PLAB, SWX, RIGL from the original run).
Globbing the directory would silently smuggle previously-rejected stocks
back into "the expanded universe," which is exactly the kind of
uncontrolled inclusion this project is trying to avoid. Only the 70%
practice window is read for every stock; the 30% holdout is never
touched. No parameters are tuned, no alternate signal is tried, and no
configuration is selected based on these results.
"""
import os
import numpy as np
import pandas as pd

TUNE = 0.70
Z_WINDOW = 20
HORIZONS = [1, 5, 10, 20]
COST_LEVELS = [0.001, 0.002, 0.004]
DECILE_FRACTION = 0.10


def discover_universe():
    passing = set()
    for fname in ["universe_verification.csv", "universe_expanded.csv"]:
        path = os.path.join("data", fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        passing.update(df.loc[df["status"] == "PASS", "ticker"].tolist())
    universe = sorted(t for t in passing if os.path.exists(os.path.join("data", f"{t}_daily.csv")))
    return universe


def load_tune_data(universe):
    data = {}
    for ticker in universe:
        path = os.path.join("data", f"{ticker}_daily.csv")
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


def find_breakeven_cost(net_total_return_fn, hi=0.05, tol=1e-6, max_iter=60):
    if net_total_return_fn(0.0) <= 0:
        return None
    if net_total_return_fn(hi) > 0:
        return None
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


UNIVERSE = discover_universe()
print(f"Expanded universe discovered: {len(UNIVERSE)} tickers (union of PASS records in "
      f"universe_verification.csv + universe_expanded.csv)")
print(", ".join(UNIVERSE))

DATA = load_tune_data(UNIVERSE)
ticker_z = {t: compute_zscore20(df["close"].to_numpy(dtype=float)) for t, df in DATA.items()}

# Aligned cross-sectional panel (inner join on date, practice window only)
z_series, open_series = {}, {}
for ticker, df in DATA.items():
    z_series[ticker] = pd.Series(ticker_z[ticker], index=df["date"])
    open_series[ticker] = pd.Series(df["open"].to_numpy(dtype=float), index=df["date"])

z_wide = pd.DataFrame(z_series)
opens_wide = pd.DataFrame(open_series)
valid = z_wide.notna().all(axis=1) & opens_wide.notna().all(axis=1)
z_wide = z_wide.loc[valid].sort_index()
opens_wide = opens_wide.loc[valid].sort_index()
tickers = list(z_wide.columns)
n_days = len(z_wide)
n_stocks = len(tickers)
decile_n = max(1, round(n_stocks * DECILE_FRACTION))

print(f"Aligned panel: {n_days} common trading days across {n_stocks} stocks "
      f"(inner-joined on date, practice window only)")
print(f"Decile size: {decile_n} stocks per side (long bottom decile / short top decile)")

print()
print("=" * 96)
print("SCALE TEST: Cross-sectional long/short at decile granularity, expanded universe")
print("=" * 96)

analysis = {}
for h in HORIZONS:
    last_i = n_days - 2 - h
    positions = list(range(0, last_i + 1, h))

    gross_returns, longs, shorts = [], [], []
    for i in positions:
        z_today = z_wide.iloc[i]
        ranked = z_today.sort_values()
        long_t = ranked.index[:decile_n].tolist()
        short_t = ranked.index[-decile_n:].tolist()
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
            w[t] = 1 / decile_n
        for t in sset:
            w[t] = -1 / decile_n
        turnovers.append(0.5 * float(np.abs(w - prev_w).sum()))
        prev_w = w
    turnovers = np.array(turnovers)

    n_periods = len(gross_returns)
    periods_per_year = 252.0 / h
    n_days_spanned = n_periods * h

    # pooled t-stat on the period-level spread returns (new vs quantile_test.py)
    mean_spread = float(gross_returns.mean())
    std_spread = float(gross_returns.std(ddof=1))
    t_stat = mean_spread / (std_spread / np.sqrt(n_periods)) if std_spread > 0 else float("nan")

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
        max_dd = float((equity / running_max - 1).min())
        std = net.std(ddof=1)
        sharpe = float(net.mean() / std * np.sqrt(periods_per_year)) if std > 0 else float("nan")
        variant_stats[label] = {
            "total_return": total_return, "annualized": annualized,
            "sharpe": sharpe, "max_dd": max_dd,
        }

    breakeven = find_breakeven_cost(net_total_return)

    analysis[h] = {
        "n_periods": n_periods, "avg_turnover": float(turnovers.mean()),
        "variants": variant_stats, "breakeven": breakeven,
        "t_stat": t_stat, "mean_spread": mean_spread, "std_spread": std_spread,
    }

for h in HORIZONS:
    res = analysis[h]
    print()
    print(f"--- Rebalance every {h} trading day(s) -- {res['n_periods']} periods, "
          f"avg turnover {res['avg_turnover']*100:.1f}% per rebalance ---")
    print(f"Pooled spread t-stat: {res['t_stat']:.2f}  "
          f"(mean {res['mean_spread']*100:.3f}%, std {res['std_spread']*100:.3f}%, n={res['n_periods']})")
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

print()
print("=" * 96)
print("SUMMARY: what survives 0.2% round-trip costs, at scale")
print("=" * 96)

survivors = [h for h in HORIZONS if analysis[h]["variants"]["net_2bp"]["total_return"] > 0]
significant = [h for h in HORIZONS if not np.isnan(analysis[h]["t_stat"]) and abs(analysis[h]["t_stat"]) > 2]

if survivors:
    print(f"Profitable net of 0.2% cost at rebalance frequency(ies): {', '.join(str(h)+'d' for h in survivors)}.")
else:
    print("NOT profitable at any rebalance frequency after 0.2% cost.")

if significant:
    print(f"Pooled spread t-stat exceeds |t|>2 at: {', '.join(str(h)+'d' for h in significant)}.")
else:
    print("Pooled spread t-stat does not exceed |t|>2 at any rebalance frequency.")

print()
print(f"Universe size: {n_stocks} stocks (vs 13 previously) -- {n_stocks/13:.1f}x the original sample.")
print("This is a diagnostic re-run at scale only: no parameters were tuned, no alternate")
print("signal was tried, and no configuration was selected based on these results.")
