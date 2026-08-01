"""
vol_target_test.py -- Test 15: volatility targeting (Moreira & Muir,
Journal of Finance 2017), the one genuinely untested hypothesis class
remaining after fourteen rejections. Every prior test asked "can we
predict which stocks go up?" This asks a different question: given
that we hold the market, can we time HOW MUCH of it we hold? Scale
exposure down when realized volatility is high, up when it's low --
this predicts nothing about direction, it manages risk. Disputed:
Cederburg et al. found the effect fails out-of-sample for most
factors. Prior is low going in; the project's record is 0/14.

Pre-registered spec (fixed, one configuration, no variants):
  - Each month-end, realized vol = std(daily returns, trailing 21
    trading days) * sqrt(252) (annualized).
  - Next month's exposure = target_vol / realized_vol, target_vol =
    15%, clipped to [0, 1.5].
  - Rebalanced monthly at the open -- same telescoping open-to-open
    monthly-return convention used by every other factor test in this
    project (signal known at month-end close, position entered at the
    next month's open, held to the following month's open).
  - Exposure above 1.0 is margin: 5%/year financing cost on the
    borrowed portion, charged monthly (5%/12).
  - 0.05% round-trip cost on the CHANGE in exposure each month
    (ETF-scale -- this trades one instrument, not a multi-name
    portfolio, hence the much smaller cost than the 20-100bp costs
    used in the cross-sectional factor tests).

Evidence boundary: practice = SPY inception (1993-01-29) through
2009-12. Holdout = 2010-01 onward, sealed, never read below.

Honest comparison, because this is the crux of what Moreira & Muir vs.
Cederburg et al. actually disagree about: Sharpe ratio is
leverage-invariant in the textbook case -- scale both the mean return
and the volatility of a strategy by the same constant and the ratio
doesn't move. So if vol-targeting "works," the improvement has to come
specifically from the TIME-VARYING exposure, not merely from carrying
leverage. A constant-leverage benchmark is built here, set to the
vol-targeting strategy's own realized AVERAGE exposure over the
practice window (a quantity DERIVED from the strategy's own behavior,
not chosen or tuned for a favorable result) and costed identically. If
vol-targeting doesn't beat constant leverage at the same average
exposure, the timing added nothing beyond what plain leverage would
have.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yfinance as yf

SPY_PATH = os.path.join("data", "spy_full_history.parquet")
FETCH_START = "1993-01-01"
HOLDOUT_START = pd.Timestamp("2010-01-01")

VOL_LOOKBACK_DAYS = 21
TARGET_VOL = 0.15
EXPOSURE_MIN, EXPOSURE_MAX = 0.0, 1.5
MARGIN_RATE_ANNUAL = 0.05
ROUNDTRIP_COST = 0.0005
TRADING_DAYS = 252

pd.set_option("display.width", 140)


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
    """Verbatim math from fundamental_test.py / fugazzi_retest.py's alpha_regression()."""
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


def print_stats_row(label, s):
    print(f"{label:<28}{s['total_return']*100:>10.2f}%{s['annualized']*100:>9.2f}%"
          f"{s['sharpe']:>9.2f}{s['max_dd']*100:>9.2f}%")


# ============================================================
# FETCH (cached) -- SPY from inception, deeper history than anything used so far
# ============================================================
print("=" * 100)
print("FETCH")
print("=" * 100)
if os.path.exists(SPY_PATH):
    raw = pd.read_parquet(SPY_PATH)
    print(f"Loaded cached {SPY_PATH}")
else:
    print(f"Fetching SPY daily history from {FETCH_START} (inception) via yfinance...")
    data = yf.download("SPY", start=FETCH_START, auto_adjust=True, progress=False)
    data.columns = data.columns.get_level_values(0)
    raw = data.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
    raw.columns = ["date", "open", "high", "low", "close", "volume"]
    raw.to_parquet(SPY_PATH, index=False)
    print(f"Saved to {SPY_PATH}")

raw["date"] = pd.to_datetime(raw["date"])
print(f"Raw SPY history on disk: {raw['date'].min().date()} to {raw['date'].max().date()} ({len(raw)} rows)")

# ============================================================
# EVIDENCE BOUNDARY -- truncate immediately, delete the untruncated reference
# ============================================================
print()
print("=" * 100)
print("EVIDENCE BOUNDARY CONFIRMATION")
print("=" * 100)
practice = raw[raw["date"] < HOLDOUT_START].sort_values("date").reset_index(drop=True)
n_dropped = len(raw) - len(practice)
del raw

print(f"Practice window kept: date < {HOLDOUT_START.date()} ({len(practice)} rows, "
      f"{practice['date'].min().date()} to {practice['date'].max().date()})")
print(f"Rows excluded as holdout: {n_dropped}")
print(f"Maximum date present in any DataFrame from this point forward: {practice['date'].max().date()}")
print("CONFIRMED: 2010-01-01 onward was not read by this script beyond the initial date check above.")

# ============================================================
# DAILY RETURNS + ROLLING REALIZED VOLATILITY
# ============================================================
practice["daily_ret"] = practice["close"].pct_change()
practice["realized_vol"] = practice["daily_ret"].rolling(VOL_LOOKBACK_DAYS).std() * np.sqrt(TRADING_DAYS)

# ============================================================
# MONTHLY PANEL
# ============================================================
practice["month"] = practice["date"].dt.to_period("M")
monthly_open = practice.groupby("month")["open"].first()
month_end_vol = practice.groupby("month")["realized_vol"].last()

monthly = pd.DataFrame({"open": monthly_open, "vol_asof_month_end": month_end_vol}).sort_index()
monthly["next_open"] = monthly["open"].shift(-1)
monthly["next_month"] = monthly.index.to_series().shift(-1)
valid_next = monthly["next_month"] == (monthly.index.to_series() + 1)
monthly.loc[~valid_next, "next_open"] = np.nan
monthly["spy_ret"] = monthly["next_open"] / monthly["open"] - 1

# Exposure for month m uses the vol measured at the END of month m-1 (no lookahead:
# the signal that sets a month's exposure is fully known before that month begins).
monthly["vol_for_exposure"] = monthly["vol_asof_month_end"].shift(1)
monthly["exposure"] = (TARGET_VOL / monthly["vol_for_exposure"]).clip(EXPOSURE_MIN, EXPOSURE_MAX)

n_before_dropna = len(monthly)
usable = monthly.dropna(subset=["exposure", "spy_ret"]).copy()
print()
print("=" * 100)
print("MONTHLY PANEL")
print("=" * 100)
print(f"Total months in practice window: {n_before_dropna}")
print(f"Months dropped (21-day vol warmup at the start, or the final month with no next_open): "
      f"{n_before_dropna - len(usable)}")
print(f"Usable months: {len(usable)} ({usable.index.min()} to {usable.index.max()})")

# ============================================================
# VOL-TARGETED PORTFOLIO: exposure, financing cost, turnover cost
# ============================================================
exposure = usable["exposure"].to_numpy()
spy_ret = usable["spy_ret"].to_numpy()

prev_exposure = np.empty_like(exposure)
prev_exposure[0] = 0.0  # starting from cash
prev_exposure[1:] = exposure[:-1]
turnover = np.abs(exposure - prev_exposure)

margin_cost = np.maximum(0.0, exposure - 1.0) * (MARGIN_RATE_ANNUAL / 12.0)
turnover_cost = turnover * ROUNDTRIP_COST
net_ret_targeted = exposure * spy_ret - margin_cost - turnover_cost

avg_exposure = float(exposure.mean())
avg_turnover = float(turnover.mean())

# ============================================================
# BENCHMARK 1: plain SPY buy-and-hold (exposure fixed at 1.0, uncosted)
# ============================================================
net_ret_spy = spy_ret.copy()

# ============================================================
# BENCHMARK 2: constant leverage at the strategy's OWN realized average exposure
# -- isolates "did the TIMING help" from "did carrying leverage help"
# ============================================================
const_exposure = np.full_like(exposure, avg_exposure)
const_prev = np.empty_like(const_exposure)
const_prev[0] = 0.0
const_prev[1:] = const_exposure[:-1]  # constant after month 1 -> turnover is 0 from month 2 on
const_turnover = np.abs(const_exposure - const_prev)
const_margin_cost = np.maximum(0.0, const_exposure - 1.0) * (MARGIN_RATE_ANNUAL / 12.0)
const_turnover_cost = const_turnover * ROUNDTRIP_COST
net_ret_const_leverage = const_exposure * spy_ret - const_margin_cost - const_turnover_cost

# ============================================================
# REPORT
# ============================================================
print()
print("=" * 100)
print("RESULTS -- practice window only (1993 to 2009-12)")
print("=" * 100)
stats_targeted = perf_stats(net_ret_targeted)
stats_spy = perf_stats(net_ret_spy)
stats_const = perf_stats(net_ret_const_leverage)

header = f"{'Portfolio':<28}{'TotalRet':>11}{'AnnRet':>10}{'Sharpe':>9}{'MaxDD':>9}"
print(header)
print("-" * len(header))
print_stats_row("Vol-targeted", stats_targeted)
print_stats_row("SPY buy & hold", stats_spy)
print_stats_row("Constant leverage (same avg)", stats_const)

print()
print(f"Average exposure (vol-targeted):        {avg_exposure*100:.1f}%")
print(f"Average monthly turnover (exposure Δ):   {avg_turnover*100:.1f}pp")
print(f"SPY buy & hold: exposure fixed at 100%, turnover 0% by construction.")

print()
print("=" * 100)
print("ALPHA REGRESSION -- vol-targeted portfolio (net of costs) vs SPY, monthly")
print("=" * 100)
reg = alpha_regression(net_ret_targeted, spy_ret)
print(f"Beta:                 {reg['beta']:.3f}")
print(f"Alpha (annualized):   {reg['alpha_annualized']*100:+.3f}%")
print(f"R-squared:            {reg['r2']:.3f}")
print(f"t-stat on alpha:      {reg['t_alpha']:.2f}   (n={reg['n']} monthly observations)")

print()
print("=" * 100)
print("HONEST COMPARISON -- is this risk-adjusted improvement, and does it beat leverage alone?")
print("=" * 100)
sharpe_ratio_vs_spy = stats_targeted["sharpe"] / stats_spy["sharpe"] if stats_spy["sharpe"] else float("nan")
sharpe_ratio_vs_const = stats_targeted["sharpe"] / stats_const["sharpe"] if stats_const["sharpe"] else float("nan")
print(f"Sharpe(vol-targeted) / Sharpe(SPY):                {sharpe_ratio_vs_spy:.3f}")
print(f"Sharpe(vol-targeted) / Sharpe(constant leverage):  {sharpe_ratio_vs_const:.3f}")
print()
print(f"Vol-targeted total/annualized return vs SPY: {stats_targeted['total_return']*100:.2f}% / "
      f"{stats_targeted['annualized']*100:.2f}%  vs  {stats_spy['total_return']*100:.2f}% / "
      f"{stats_spy['annualized']*100:.2f}%")
print(f"-> Raw-return framing ({'beats' if stats_targeted['total_return'] > stats_spy['total_return'] else 'does NOT beat'} "
      f"SPY on total return) is not the claim being tested here -- Sharpe ratio is.")
print()
if sharpe_ratio_vs_const > 1.0 and stats_targeted["sharpe"] > stats_spy["sharpe"]:
    verdict = ("Vol-targeting improves Sharpe over BOTH plain SPY and constant leverage at the same "
               "average exposure -- the improvement is attributable to the time-varying exposure "
               "itself, not merely to carrying leverage.")
elif stats_targeted["sharpe"] > stats_spy["sharpe"] and sharpe_ratio_vs_const <= 1.0:
    verdict = ("Vol-targeting improves Sharpe over plain SPY, but NOT over constant leverage at the "
               "same average exposure -- the apparent improvement is explained by carrying leverage, "
               "not by timing it. This does not support the Moreira & Muir timing claim specifically.")
else:
    verdict = "Vol-targeting does not improve Sharpe over plain SPY at all. Rejected outright."
print(f"VERDICT: {verdict}")

print()
print("=" * 100)
print("This is a single pre-registered specification: target_vol=15%, 21-day lookback, [0, 1.5x] cap, "
      "5%/yr margin cost, 0.05% round-trip cost. None of these were tuned or varied. The 2010+ holdout "
      "was not read anywhere above.")
