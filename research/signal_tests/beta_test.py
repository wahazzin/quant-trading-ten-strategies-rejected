"""
beta_test.py -- confirms or refutes the market-timing hypothesis raised
by quantile_test.py's split verdict: the pooled time-series Q1-Q5
spread survived costs, but the market-neutral cross-sectional long/short
failed gross. That pattern is consistent with the "edge" actually being
market-wide dip-buying (beta), not stock selection (alpha).

Analysis 1: is the Q1 edge concentrated on days when many stocks are
            simultaneously oversold (pooled bottom quintile)?
Analysis 2: a realistic long-only Q1 strategy vs SPY buy-and-hold and
            an equal-weighted 13-stock buy-and-hold, over the identical
            window (project rule 5's required benchmark comparison).
Analysis 3: regress the strategy's daily returns on SPY's daily returns
            for beta / alpha / R^2 / alpha t-stat.

Signal (fixed, not tuned): 20-day close z-score, (close - MA20) / std20.
"Pooled bottom quintile" = same concept as quantile_test.py Analysis A:
one global threshold from the pooled distribution of all (stock, day)
z-scores in the practice window, not a per-day cross-sectional rank.

Only the 70% practice window is read for the 13 stocks; the 30% holdout
is never touched. SPY is a benchmark, not part of that split -- it is
just sliced to the identical aligned date range. No parameters are
tuned and no alternate signal is tried.
"""
import os
import numpy as np
import pandas as pd

UNIVERSE = ["SBRA", "VLY", "FLO", "AROC", "HUN", "WEN",
            "CLF", "MGNI", "KSS", "TROX", "VSH", "UAA", "HL"]
TUNE = 0.70
Z_WINDOW = 20
HOLD = 10
COST = 0.002


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
    n = len(opens)
    fwd = np.full(n, np.nan)
    last_n = n - 1 - horizon
    if last_n <= 0:
        return fwd
    idx = np.arange(0, last_n)
    fwd[idx] = opens[idx + 1 + horizon] / opens[idx + 1] - 1
    return fwd


DATA = load_tune_data()

# ------------------------------------------------------------------
# Aligned cross-sectional panel across the 13 stocks (inner join on
# date, practice window only -- same construction as quantile_test.py)
# ------------------------------------------------------------------
z_series, open_series, close_series = {}, {}, {}
for ticker, df in DATA.items():
    z = pd.Series(compute_zscore20(df["close"].to_numpy(dtype=float)), index=df["date"])
    z_series[ticker] = z
    open_series[ticker] = pd.Series(df["open"].to_numpy(dtype=float), index=df["date"])
    close_series[ticker] = pd.Series(df["close"].to_numpy(dtype=float), index=df["date"])

z_wide = pd.DataFrame(z_series)
opens_wide = pd.DataFrame(open_series)
closes_wide = pd.DataFrame(close_series)
valid = z_wide.notna().all(axis=1) & opens_wide.notna().all(axis=1) & closes_wide.notna().all(axis=1)
z_wide = z_wide.loc[valid].sort_index()
opens_wide = opens_wide.loc[valid].sort_index()
closes_wide = closes_wide.loc[valid].sort_index()
tickers = list(z_wide.columns)
dates = z_wide.index
n = len(dates)

# Pooled bottom-quintile threshold: one global cutoff from every (stock,
# day) z-score in the aligned window, matching quantile_test.py Analysis A.
all_z = z_wide.to_numpy().ravel()
q1_threshold = float(np.nanpercentile(all_z, 20))
is_q1 = z_wide <= q1_threshold

print(f"Aligned panel: {n} common trading days across {len(tickers)} stocks")
print(f"Pooled bottom-quintile threshold: z <= {q1_threshold:.3f}")

fwd10_wide = pd.DataFrame(
    {t: forward_return(opens_wide[t].to_numpy(), HOLD) for t in tickers}, index=dates
)


# ============================================================
# ANALYSIS 1 -- is the Q1 edge concentrated on broad-dip days?
# ============================================================
print()
print("=" * 90)
print("ANALYSIS 1: Is the Q1 edge concentrated on days when many stocks are")
print("            simultaneously oversold (market-wide dip) vs a few (stock-specific)?")
print("=" * 90)

last_valid = n - 1 - HOLD   # matches forward_return's own validity bound
buckets_order = ["0", "1-3", "4-6", "7-9", "10+"]


def bucket_count(c):
    if c == 0:
        return "0"
    elif c <= 3:
        return "1-3"
    elif c <= 6:
        return "4-6"
    elif c <= 9:
        return "7-9"
    else:
        return "10+"


count_per_day = is_q1.iloc[:last_valid].sum(axis=1)
dist = count_per_day.apply(bucket_count).value_counts().reindex(buckets_order, fill_value=0)

records = []
for i in range(last_valid):
    q1_today = [t for t in tickers if is_q1.iloc[i][t]]
    if not q1_today:
        continue
    frs = fwd10_wide.iloc[i][q1_today].dropna()
    if frs.empty:
        continue
    records.append((int(count_per_day.iloc[i]), bucket_count(int(count_per_day.iloc[i])), float(frs.mean())))

detail = pd.DataFrame(records, columns=["count", "bucket", "fwd_ret"])
cond = detail.groupby("bucket")["fwd_ret"].agg(["mean", "count"]).reindex(buckets_order)

header = f"{'#simultaneous Q1':<18}{'Days':>7}{'AvgFwd10dRet':>15}{'n(withFwd)':>12}"
print(header)
print("-" * len(header))
for b in buckets_order:
    d = int(dist.loc[b])
    if b in cond.index and not pd.isna(cond.loc[b, "mean"]):
        m = cond.loc[b, "mean"] * 100
        c = int(cond.loc[b, "count"])
        print(f"{b:<18}{d:>7}{m:>14.3f}%{c:>12}")
    else:
        print(f"{b:<18}{d:>7}{'n/a':>15}{0:>12}")

valid_cond = cond.dropna()
if len(valid_cond) >= 2:
    lo_bucket = valid_cond.index[0]
    hi_bucket = valid_cond.index[-1]
    lo_ret = valid_cond.loc[lo_bucket, "mean"]
    hi_ret = valid_cond.loc[hi_bucket, "mean"]
    print()
    print(f"Lowest-concurrency bucket with data ('{lo_bucket}'): {lo_ret*100:.3f}% avg fwd 10d return")
    print(f"Highest-concurrency bucket with data ('{hi_bucket}'): {hi_ret*100:.3f}% avg fwd 10d return")
    if hi_ret > lo_ret:
        print("-> Forward returns are HIGHER when more stocks are simultaneously oversold: "
              "consistent with market-wide dip-buying, not stock-specific selection.")
    else:
        print("-> Forward returns are NOT higher when more stocks are simultaneously oversold: "
              "does not support the market-timing explanation on its own.")


# ============================================================
# ANALYSIS 2 -- long-only Q1 strategy vs benchmarks
# ============================================================
print()
print("=" * 90)
print("ANALYSIS 2: Long-only Q1 strategy vs SPY and equal-weighted 13-stock buy-and-hold")
print("=" * 90)

closes_np = closes_wide.to_numpy()
opens_np = opens_wide.to_numpy()
is_q1_np = is_q1.to_numpy()

# 10 overlapping sub-portfolios ("cohorts"), one per day-of-cycle offset,
# each independently entering/exiting on its own 10-day tiling -- the
# standard construction for a daily-entry, N-day-hold systematic strategy.
bucket_daily_return = np.zeros((HOLD, n))
bucket_invested = np.zeros((HOLD, n), dtype=bool)

for b in range(HOLD):
    k = 0
    while True:
        i_k = b + k * HOLD
        entry_day = i_k + 1
        exit_day = i_k + 1 + HOLD
        if exit_day > n - 1:
            break
        q1_idx = np.where(is_q1_np[i_k])[0]
        if len(q1_idx) > 0:
            bucket_daily_return[b, entry_day] = (closes_np[entry_day, q1_idx] / opens_np[entry_day, q1_idx] - 1).mean()
            bucket_invested[b, entry_day] = True
            for j in range(entry_day + 1, exit_day):
                bucket_daily_return[b, j] = (closes_np[j, q1_idx] / closes_np[j - 1, q1_idx] - 1).mean()
                bucket_invested[b, j] = True
            bucket_daily_return[b, exit_day] = (opens_np[exit_day, q1_idx] / closes_np[exit_day - 1, q1_idx] - 1).mean() - COST
            bucket_invested[b, exit_day] = True
        k += 1

strategy_daily_ret = bucket_daily_return.mean(axis=0)
pct_invested = float(bucket_invested.mean() * 100)

strategy_equity = np.cumprod(1 + strategy_daily_ret)

spy_df = pd.read_csv(os.path.join("data", "SPY_daily.csv"), parse_dates=["date"]).sort_values("date").reset_index(drop=True)
spy_close = spy_df.set_index("date")["close"].reindex(dates)
if spy_close.isna().any():
    missing = int(spy_close.isna().sum())
    print(f"NOTE: {missing} of {n} dates missing from SPY data after alignment -- forward-filling.")
    spy_close = spy_close.ffill().bfill()
spy_equity = (spy_close / spy_close.iloc[0]).to_numpy()

ew_equity = (closes_wide.to_numpy() / closes_wide.iloc[0].to_numpy()).mean(axis=1)


def perf_stats(equity, daily_returns=None):
    if daily_returns is None:
        daily_returns = np.diff(equity) / equity[:-1]
    total_return = float(equity[-1] - 1)
    annualized = float(equity[-1] ** (252.0 / len(equity)) - 1)
    running_max = np.maximum.accumulate(equity)
    max_dd = float((equity / running_max - 1).min())
    std = daily_returns.std(ddof=1)
    sharpe = float(daily_returns.mean() / std * np.sqrt(252)) if std > 0 else float("nan")
    return total_return, annualized, sharpe, max_dd


strat_stats = perf_stats(strategy_equity, strategy_daily_ret)
spy_stats = perf_stats(spy_equity)
ew_stats = perf_stats(ew_equity)

header = f"{'':<20}{'TotalRet':>11}{'AnnRet':>10}{'Sharpe':>9}{'MaxDD':>9}{'PctInvested':>13}"
print(header)
print("-" * len(header))
for label, stats, pct in [
    ("Q1 strategy (net)", strat_stats, pct_invested),
    ("SPY buy&hold", spy_stats, 100.0),
    ("EW 13-stock buy&hold", ew_stats, 100.0),
]:
    tr, ar, sh, dd = stats
    print(f"{label:<20}{tr*100:>10.2f}%{ar*100:>9.2f}%{sh:>9.2f}{dd*100:>8.2f}%{pct:>12.1f}%")


# ============================================================
# ANALYSIS 3 -- beta decomposition
# ============================================================
print()
print("=" * 90)
print("ANALYSIS 3: Beta decomposition (Q1 strategy daily returns regressed on SPY daily returns)")
print("=" * 90)

spy_daily_ret = spy_close.pct_change().to_numpy()
mask = ~np.isnan(spy_daily_ret) & (np.arange(n) > 0)
X = spy_daily_ret[mask]
Y = strategy_daily_ret[mask]
n_obs = len(X)

beta = float(np.cov(X, Y, ddof=1)[0, 1] / np.var(X, ddof=1))
alpha_daily = float(Y.mean() - beta * X.mean())
pred = alpha_daily + beta * X
resid = Y - pred
ssr = float(np.sum(resid ** 2))
sst = float(np.sum((Y - Y.mean()) ** 2))
r2 = 1 - ssr / sst if sst > 0 else float("nan")
sigma2 = ssr / (n_obs - 2)
sxx = float(np.sum((X - X.mean()) ** 2))
se_alpha = float(np.sqrt(sigma2 * (1 / n_obs + X.mean() ** 2 / sxx)))
t_alpha = alpha_daily / se_alpha if se_alpha > 0 else float("nan")
alpha_annualized = float((1 + alpha_daily) ** 252 - 1)

print(f"Beta:                 {beta:.3f}")
print(f"Alpha (annualized):   {alpha_annualized*100:.3f}%")
print(f"R-squared:            {r2:.3f}")
print(f"t-stat on alpha:      {t_alpha:.2f}   (n={n_obs} daily observations)")


# ============================================================
# PLAIN-ENGLISH CONCLUSION
# ============================================================
print()
print("=" * 90)
print("CONCLUSION")
print("=" * 90)

beats_spy_total = strat_stats[0] > spy_stats[0]
beats_spy_sharpe = strat_stats[2] > spy_stats[2]
alpha_significant = (not np.isnan(t_alpha)) and (t_alpha > 2) and (alpha_annualized > 0)

if beats_spy_total and beats_spy_sharpe:
    print(f"The Q1 strategy beats SPY buy-and-hold on both total return "
          f"({strat_stats[0]*100:.1f}% vs {spy_stats[0]*100:.1f}%) and Sharpe "
          f"({strat_stats[2]:.2f} vs {spy_stats[2]:.2f}) after 0.2% costs.")
else:
    print(f"The Q1 strategy does NOT beat SPY buy-and-hold after 0.2% costs "
          f"(total return {strat_stats[0]*100:.1f}% vs {spy_stats[0]*100:.1f}%, "
          f"Sharpe {strat_stats[2]:.2f} vs {spy_stats[2]:.2f}).")

if alpha_significant:
    print(f"Alpha is statistically significant and positive (t={t_alpha:.2f}, "
          f"annualized alpha {alpha_annualized*100:.2f}%) -- some return is not explained by SPY beta.")
else:
    print(f"Alpha is NOT statistically significant and positive (t={t_alpha:.2f}, "
          f"annualized alpha {alpha_annualized*100:.2f}%, beta {beta:.2f}): the strategy's "
          f"return looks like leveraged/adjusted SPY exposure with extra steps, not "
          f"stock-picking skill.")

print()
print("This is a diagnostic test only: the 20-day z-score, 0.2% cost, and 10-day hold")
print("were used exactly as specified -- nothing here was tuned or optimized.")
