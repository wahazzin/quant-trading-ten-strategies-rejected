"""
vol_target_holdout_test.py -- Test 15's ONE-SHOT holdout run. This is
the final validation step for volatility targeting (Moreira & Muir
2017): the practice window (1993-2009, RESEARCH_LOG.md Test 15) showed
Sharpe 0.66 vs SPY's 0.54 (ratio 1.217) and 1.244 vs constant leverage
at matched average exposure, with alpha t=1.59 (positive, not
significant). This script spends the sealed 2010+ holdout to see if
that persists.

Reused VERBATIM from vol_target_test.py, unchanged: the 21-day
realized-vol lookback, 15% target vol, [0, 1.5x] exposure cap, 5%/yr
margin cost on the borrowed portion, 0.05% round-trip cost on the
exposure change, monthly rebalance at the open, perf_stats(), and
alpha_regression(). Nothing was tuned or varied after seeing the
holdout -- this file was written before running it, mirrors the
practice-window script's logic exactly, and is run exactly once.

Pre-registered interpretation (binding, fixed before this ran):
  - Sharpe ratio >= 1.15x SPY AND alpha t > 2       -> SURVIVED
  - Sharpe ratio >= 1.15x SPY BUT t between 1 and 2  -> CONSISTENT, not proven
  - Sharpe ratio < 1.10x SPY, OR negative alpha      -> DEAD

Practice-window figures below are hardcoded from RESEARCH_LOG.md's
already-published Test 15 entry -- NOT recomputed here, since
recomputing them would mean this "holdout-only" script reading
pre-2010 data again. They are historical record, not re-derived.

This is the final validation run for this hypothesis. No iteration,
no anomaly investigation, no follow-up variants -- report what this
spec produces, once.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

SPY_PATH = os.path.join("data", "spy_full_history.parquet")
HOLDOUT_START = pd.Timestamp("2010-01-01")

VOL_LOOKBACK_DAYS = 21
TARGET_VOL = 0.15
EXPOSURE_MIN, EXPOSURE_MAX = 0.0, 1.5
MARGIN_RATE_ANNUAL = 0.05
ROUNDTRIP_COST = 0.0005
TRADING_DAYS = 252

# Pre-registered decision thresholds (fixed before this script ran)
SHARPE_RATIO_SURVIVE = 1.15
SHARPE_RATIO_DEAD = 1.10

# Already-published Test 15 practice-window results (RESEARCH_LOG.md) -- hardcoded record, not recomputed
PRACTICE = {
    "total_return": 2.6815, "annualized": 0.0813, "sharpe": 0.66, "max_dd": -0.3903,
    "avg_exposure": 1.036, "avg_turnover": 0.187,
    "alpha_beta": 0.771, "alpha_annualized": 0.02262, "alpha_r2": 0.820, "alpha_t": 1.59, "alpha_n": 200,
}
PRACTICE_SHARPE_RATIO_VS_SPY = 1.217
PRACTICE_SPY_SHARPE = 0.54

pd.set_option("display.width", 140)


def perf_stats(period_returns):
    """Verbatim from vol_target_test.py."""
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
    """Verbatim from vol_target_test.py."""
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
# LOAD + HOLDOUT BOUNDARY
# ============================================================
print("=" * 100)
print("HOLDOUT CONFIRMATION")
print("=" * 100)
raw = pd.read_parquet(SPY_PATH)
raw["date"] = pd.to_datetime(raw["date"])
print(f"Raw SPY history on disk: {raw['date'].min().date()} to {raw['date'].max().date()} ({len(raw)} rows)")

holdout = raw[raw["date"] >= HOLDOUT_START].sort_values("date").reset_index(drop=True)
n_excluded_pre2010 = len(raw) - len(holdout)
del raw

print(f"Holdout window kept: date >= {HOLDOUT_START.date()} ({len(holdout)} rows, "
      f"{holdout['date'].min().date()} to {holdout['date'].max().date()})")
print(f"Rows excluded (pre-2010 practice window): {n_excluded_pre2010}")
print(f"Minimum date present in any DataFrame from this point forward: {holdout['date'].min().date()}")
print("CONFIRMED: only 2010-01-01-onward data was evaluated below. Pre-2010 rows were dropped "
      "immediately after load and never used in any computation in this script.")

# NOTE: the first ~21-22 trading days of 2010 (into the daily-return/rolling-vol computation
# just below) are used ONLY to seed the 21-day rolling realized-vol window for exposure signals
# generated from January 2010 onward -- this is standard warmup for a trailing indicator, not a
# read of practice-window data; the practice window itself (pre-2010) is never loaded above.

# ============================================================
# DAILY RETURNS + ROLLING REALIZED VOLATILITY (verbatim methodology)
# ============================================================
holdout["daily_ret"] = holdout["close"].pct_change()
holdout["realized_vol"] = holdout["daily_ret"].rolling(VOL_LOOKBACK_DAYS).std() * np.sqrt(TRADING_DAYS)

# ============================================================
# MONTHLY PANEL (verbatim methodology)
# ============================================================
holdout["month"] = holdout["date"].dt.to_period("M")
monthly_open = holdout.groupby("month")["open"].first()
month_end_vol = holdout.groupby("month")["realized_vol"].last()

monthly = pd.DataFrame({"open": monthly_open, "vol_asof_month_end": month_end_vol}).sort_index()
monthly["next_open"] = monthly["open"].shift(-1)
monthly["next_month"] = monthly.index.to_series().shift(-1)
valid_next = monthly["next_month"] == (monthly.index.to_series() + 1)
monthly.loc[~valid_next, "next_open"] = np.nan
monthly["spy_ret"] = monthly["next_open"] / monthly["open"] - 1

monthly["vol_for_exposure"] = monthly["vol_asof_month_end"].shift(1)
monthly["exposure"] = (TARGET_VOL / monthly["vol_for_exposure"]).clip(EXPOSURE_MIN, EXPOSURE_MAX)

n_before_dropna = len(monthly)
usable = monthly.dropna(subset=["exposure", "spy_ret"]).copy()
print()
print("=" * 100)
print("MONTHLY PANEL")
print("=" * 100)
print(f"Total months in holdout window: {n_before_dropna}")
print(f"Months dropped (vol warmup at the start, or final month with no next_open): "
      f"{n_before_dropna - len(usable)}")
print(f"Usable months: {len(usable)} ({usable.index.min()} to {usable.index.max()})")

# ============================================================
# VOL-TARGETED PORTFOLIO (verbatim methodology)
# ============================================================
exposure = usable["exposure"].to_numpy()
spy_ret = usable["spy_ret"].to_numpy()

prev_exposure = np.empty_like(exposure)
prev_exposure[0] = 0.0
prev_exposure[1:] = exposure[:-1]
turnover = np.abs(exposure - prev_exposure)

margin_cost = np.maximum(0.0, exposure - 1.0) * (MARGIN_RATE_ANNUAL / 12.0)
turnover_cost = turnover * ROUNDTRIP_COST
net_ret_targeted = exposure * spy_ret - margin_cost - turnover_cost

avg_exposure = float(exposure.mean())
avg_turnover = float(turnover.mean())

net_ret_spy = spy_ret.copy()

const_exposure = np.full_like(exposure, avg_exposure)
const_prev = np.empty_like(const_exposure)
const_prev[0] = 0.0
const_prev[1:] = const_exposure[:-1]
const_turnover = np.abs(const_exposure - const_prev)
const_margin_cost = np.maximum(0.0, const_exposure - 1.0) * (MARGIN_RATE_ANNUAL / 12.0)
const_turnover_cost = const_turnover * ROUNDTRIP_COST
net_ret_const_leverage = const_exposure * spy_ret - const_margin_cost - const_turnover_cost

# ============================================================
# REPORT
# ============================================================
print()
print("=" * 100)
print("RESULTS -- HOLDOUT WINDOW ONLY (2010-01 to present)")
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
print(f"Average monthly turnover (exposure delta): {avg_turnover*100:.1f}pp")

print()
print("=" * 100)
print("ALPHA REGRESSION -- vol-targeted portfolio (net of costs) vs SPY, monthly, HOLDOUT")
print("=" * 100)
reg = alpha_regression(net_ret_targeted, spy_ret)
print(f"Beta:                 {reg['beta']:.3f}")
print(f"Alpha (annualized):   {reg['alpha_annualized']*100:+.3f}%")
print(f"R-squared:            {reg['r2']:.3f}")
print(f"t-stat on alpha:      {reg['t_alpha']:.2f}   (n={reg['n']} monthly observations)")

# ============================================================
# PRACTICE vs HOLDOUT SIDE-BY-SIDE
# ============================================================
print()
print("=" * 100)
print("PRACTICE (1993-2009) vs HOLDOUT (2010-present) -- side by side")
print("=" * 100)
compare_header = f"{'Metric':<28}{'Practice':>14}{'Holdout':>14}"
print(compare_header)
print("-" * len(compare_header))
print(f"{'Total return':<28}{PRACTICE['total_return']*100:>13.2f}%{stats_targeted['total_return']*100:>13.2f}%")
print(f"{'Annualized return':<28}{PRACTICE['annualized']*100:>13.2f}%{stats_targeted['annualized']*100:>13.2f}%")
print(f"{'Sharpe':<28}{PRACTICE['sharpe']:>14.2f}{stats_targeted['sharpe']:>14.2f}")
print(f"{'Max drawdown':<28}{PRACTICE['max_dd']*100:>13.2f}%{stats_targeted['max_dd']*100:>13.2f}%")
print(f"{'Avg exposure':<28}{PRACTICE['avg_exposure']*100:>13.1f}%{avg_exposure*100:>13.1f}%")
print(f"{'Avg monthly turnover':<28}{PRACTICE['avg_turnover']*100:>12.1f}pp{avg_turnover*100:>12.1f}pp")
print(f"{'Alpha (annualized)':<28}{PRACTICE['alpha_annualized']*100:>13.2f}%{reg['alpha_annualized']*100:>13.2f}%")
print(f"{'Alpha t-stat':<28}{PRACTICE['alpha_t']:>14.2f}{reg['t_alpha']:>14.2f}")
print(f"{'Alpha R-squared':<28}{PRACTICE['alpha_r2']:>14.3f}{reg['r2']:>14.3f}")
print(f"{'n (monthly obs)':<28}{PRACTICE['alpha_n']:>14d}{reg['n']:>14d}")

holdout_sharpe_ratio_vs_spy = stats_targeted["sharpe"] / stats_spy["sharpe"] if stats_spy["sharpe"] else float("nan")
holdout_sharpe_ratio_vs_const = stats_targeted["sharpe"] / stats_const["sharpe"] if stats_const["sharpe"] else float("nan")
print()
print(f"Sharpe ratio vs SPY -- practice: {PRACTICE_SHARPE_RATIO_VS_SPY:.3f}   holdout: {holdout_sharpe_ratio_vs_spy:.3f}")
print(f"Sharpe ratio vs constant leverage -- holdout: {holdout_sharpe_ratio_vs_const:.3f}")

# ============================================================
# COVID SUB-PERIOD (2020-02 to 2020-06)
# ============================================================
print()
print("=" * 100)
print("COVID SUB-PERIOD: 2020-02 through 2020-06 -- exposure and return by month")
print("=" * 100)
covid_months = pd.period_range("2020-02", "2020-06", freq="M")
covid_rows = []
for m in covid_months:
    if m not in usable.index:
        covid_rows.append({"month": str(m), "exposure": np.nan, "vol_targeted_ret": np.nan, "spy_ret": np.nan})
        continue
    row = usable.loc[m]
    exp = row["exposure"]
    sret = row["spy_ret"]
    mc = max(0.0, exp - 1.0) * (MARGIN_RATE_ANNUAL / 12.0)
    # turnover cost for this specific month, consistent with the vectorized calc above
    pos = usable.index.get_loc(m)
    prev_exp = usable["exposure"].iloc[pos - 1] if pos > 0 else 0.0
    tc = abs(exp - prev_exp) * ROUNDTRIP_COST
    vt_ret = exp * sret - mc - tc
    covid_rows.append({"month": str(m), "exposure": exp, "vol_targeted_ret": vt_ret, "spy_ret": sret})

covid_df = pd.DataFrame(covid_rows)
print(f"{'Month':<10}{'Exposure':>10}{'VolTgt Ret':>13}{'SPY Ret':>11}{'Difference':>13}")
print("-" * 57)
for r in covid_df.itertuples(index=False):
    if pd.isna(r.exposure):
        print(f"{r.month:<10}{'n/a':>10}{'n/a':>13}{'n/a':>11}{'n/a':>13}")
        continue
    diff = r.vol_targeted_ret - r.spy_ret
    print(f"{r.month:<10}{r.exposure*100:>9.1f}%{r.vol_targeted_ret*100:>12.2f}%{r.spy_ret*100:>10.2f}%"
          f"{diff*100:>+12.2f}pp")

valid_covid = covid_df.dropna(subset=["exposure"])
if len(valid_covid):
    covid_vt_total = float((1 + valid_covid["vol_targeted_ret"]).prod() - 1)
    covid_spy_total = float((1 + valid_covid["spy_ret"]).prod() - 1)
    print(f"\nCumulative Feb-Jun 2020: vol-targeted {covid_vt_total*100:+.2f}%  vs  SPY {covid_spy_total*100:+.2f}%  "
          f"(difference {covid_vt_total*100 - covid_spy_total*100:+.2f}pp)")
print("Note: exposure each month was set using the PRIOR month's trailing-21-day realized vol -- "
      "February's exposure reflects January 2020's (pre-crash, low-vol) reading, so the 21-day "
      "lookback structurally could not react until the vol spike itself showed up in a month-end "
      "reading, one month later.")

# ============================================================
# PRE-REGISTERED VERDICT (mechanical, per the fixed decision rule)
# ============================================================
print()
print("=" * 100)
print("PRE-REGISTERED VERDICT")
print("=" * 100)
print(f"Rule: Sharpe ratio >= {SHARPE_RATIO_SURVIVE} AND alpha t > 2       -> SURVIVED")
print(f"      Sharpe ratio >= {SHARPE_RATIO_SURVIVE} BUT t between 1 and 2  -> CONSISTENT, not proven")
print(f"      Sharpe ratio < {SHARPE_RATIO_DEAD}, OR negative alpha        -> DEAD")
print(f"\nHoldout Sharpe ratio vs SPY: {holdout_sharpe_ratio_vs_spy:.3f}")
print(f"Holdout alpha t-stat: {reg['t_alpha']:.2f}")
print(f"Holdout alpha (annualized): {reg['alpha_annualized']*100:+.3f}%")

if reg["alpha_annualized"] < 0 or holdout_sharpe_ratio_vs_spy < SHARPE_RATIO_DEAD:
    verdict = "DEAD"
elif holdout_sharpe_ratio_vs_spy >= SHARPE_RATIO_SURVIVE and reg["t_alpha"] > 2:
    verdict = "SURVIVED"
elif holdout_sharpe_ratio_vs_spy >= SHARPE_RATIO_SURVIVE and 1 <= reg["t_alpha"] <= 2:
    verdict = "CONSISTENT, not proven"
else:
    verdict = "DEAD (falls in neither the SURVIVED nor CONSISTENT band)"
print(f"\nMECHANICAL VERDICT: {verdict}")

print()
print("=" * 100)
print("PLAIN-ENGLISH CONCLUSION")
print("=" * 100)
if verdict == "SURVIVED":
    print("The effect PERSISTED out of sample. Practice-window Sharpe improvement and statistical "
          "significance both held up in the holdout. This is the first genuine finding in 15 tests.")
elif verdict.startswith("CONSISTENT"):
    print("The effect is CONSISTENT with persisting but not statistically proven in the holdout -- "
          "the Sharpe improvement held up, but the alpha t-stat did not clear 2. Worth forward "
          "testing as a risk-management layer; not worth claiming as a demonstrated edge.")
else:
    print("The effect DIED out of sample, the same outcome as low-volatility (Test 8). Whatever "
          "produced the practice-window Sharpe improvement did not persist into 2010-present.")

print()
print("Final validation run -- no iteration, no anomaly investigation, no follow-up variants. "
      "Logged and stopped per instruction.")
