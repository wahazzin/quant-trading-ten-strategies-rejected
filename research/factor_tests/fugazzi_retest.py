"""
fugazzi_retest.py -- retest of "Project Fugazzi" (a prior CAPM/Jensen's
alpha stock-picking exercise that was never benchmarked against SPY).
Three of the four original test scripts had a START_DATE bug that
accidentally tested on the 2021-2023 TRAINING period instead of the
intended 2024-2025 walk-forward window -- this rebuilds all four tests
correctly, against the identical out-of-sample window, with SPY finally
in the picture.

Conventions matched EXACTLY to the original scripts (per instruction --
not "corrected" to a more standard method, since the point is to retest
the original methodology honestly, only fixing the date-range bug):
  - Daily returns; annualized return = mean(daily_ret) * 252,
    annualized vol = std(daily_ret) * sqrt(252) (simple scaling, NOT
    geometric compounding).
  - No rebalancing: each stock gets BUDGET * weight at day one, then
    (1 + pct_change).cumprod() * weight * BUDGET per stock, summed --
    true fixed-share buy-and-hold, weights drift with relative
    performance over the holding period.
  - Beta: from 2-year weekly returns vs ^GSPC (the trailing 2 years of
    the 2021-2023 training window, i.e. 2022-01-01 to 2023-12-31 --
    the training window itself is 3 calendar years, but the beta
    convention is specifically "2-year weekly", so the beta lookback
    and the Jensen's-alpha return lookback are deliberately different
    windows, both entirely inside the training period, no lookahead
    either way).
  - risk_free = 0.045, market_ret = 0.10 (fixed assumptions, not fitted
    or fetched -- this is the original CAPM screening convention).
  - Jensen's alpha (Test 3 selection) = realized annualized return -
    (risk_free + beta * (market_ret - risk_free)).

Test 1: the original 6-stock portfolio (fixed weights, chosen with
        hindsight through 2025) over the actual 2024-2025 OOS window.
Test 2: SPY over the identical window -- the benchmark that was never
        run -- plus a full OLS alpha regression of Test 1's portfolio
        against SPY (beta, annualized alpha, R^2, t-stat), using the
        same daily-return / mean*252 annualization convention.
Test 3: the honest reselection -- Jensen's alpha computed on EACH of
        the 30 candidates using ONLY 2021-2023 data, top 6 picked as of
        end-2023, equal-weighted, run over 2024-2025. The gap between
        this and Test 1 is the size of the hindsight advantage baked
        into the original selection.
Test 4: equal-weighted all 30 candidates over 2024-2025 -- the no-skill
        baseline. If Tests 1/3 don't clear this, stock-picking wasn't
        the source of any edge.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yfinance as yf

ORIGINAL_TICKERS = ["NVDA", "AVGO", "LLY", "XOM", "WMT", "GOOGL"]
ORIGINAL_WEIGHTS = [0.19, 0.205, 0.276, 0.211, 0.019, 0.10]

UNIVERSE_30 = [
    "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "TSM", "LLY",
    "AVGO", "V", "JPM", "WMT", "XOM", "UNH", "MA", "PG", "COST", "JNJ",
    "HD", "MRK", "ABBV", "BAC", "NFLX", "AMD", "KO", "PEP", "CVX", "ORCL",
]

MARKET_INDEX = "^GSPC"
BENCHMARK_ETF = "SPY"

TRAIN_START = "2021-01-01"
TRAIN_END = "2023-12-31"
BETA_WINDOW_START = "2022-01-01"  # trailing 2 years of the training window, per convention
TEST_START = "2024-01-01"
TEST_END = "2025-12-31"

RISK_FREE = 0.045
MARKET_RET_ASSUMED = 0.10
TRADING_DAYS = 252
BUDGET = 10_000.0  # arbitrary notional capital -- cancels out of every % figure reported

pd.set_option("display.width", 140)


def annualized_return(daily_ret):
    return float(daily_ret.mean() * TRADING_DAYS)


def annualized_vol(daily_ret):
    return float(daily_ret.std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(daily_ret):
    vol = annualized_vol(daily_ret)
    return float(annualized_return(daily_ret) / vol) if vol > 0 else float("nan")


def max_drawdown(value_path):
    running_max = value_path.cummax()
    dd = value_path / running_max - 1
    return float(dd.min())


def buy_and_hold_path(daily_prices, tickers, weights):
    """No-rebalance, fixed-share buy-and-hold: BUDGET*weight invested in
    each stock at day one, then (1+pct_change).cumprod() per stock,
    summed -- verbatim reproduction of the original construction."""
    px = daily_prices[tickers]
    pct = px.pct_change().fillna(0.0)
    growth = (1 + pct).cumprod()
    dollar_paths = growth * (pd.Series(weights, index=tickers) * BUDGET)
    portfolio_value = dollar_paths.sum(axis=1)
    portfolio_ret = portfolio_value.pct_change().dropna()
    return portfolio_value, portfolio_ret


def report_row(label, value_path, ret):
    total_ret = float(value_path.iloc[-1] / value_path.iloc[0] - 1)
    return {
        "portfolio": label,
        "total_return_pct": total_ret * 100,
        "annualized_return_pct": annualized_return(ret) * 100,
        "sharpe": sharpe_ratio(ret),
        "max_dd_pct": max_drawdown(value_path) * 100,
    }


def alpha_regression_daily(port_ret, bench_ret):
    """Same OLS math as this project's other alpha_regression() helpers
    (factor_test.py / fundamental_test.py), adapted to daily data and
    the mean*252 annualization convention specified for this retest."""
    aligned = pd.concat([port_ret.rename("port"), bench_ret.rename("bench")], axis=1).dropna()
    X = aligned["bench"].to_numpy()
    Y = aligned["port"].to_numpy()
    n = len(X)
    beta = float(np.cov(X, Y, ddof=1)[0, 1] / np.var(X, ddof=1))
    alpha_daily = float(Y.mean() - beta * X.mean())
    resid = Y - (alpha_daily + beta * X)
    ssr = float(np.sum(resid ** 2))
    sst = float(np.sum((Y - Y.mean()) ** 2))
    r2 = 1 - ssr / sst if sst > 0 else float("nan")
    sigma2 = ssr / (n - 2) if n > 2 else float("nan")
    sxx = float(np.sum((X - X.mean()) ** 2))
    se_alpha = float(np.sqrt(sigma2 * (1 / n + X.mean() ** 2 / sxx))) if sxx > 0 else float("nan")
    t_alpha = alpha_daily / se_alpha if se_alpha and se_alpha > 0 else float("nan")
    alpha_annualized = alpha_daily * TRADING_DAYS  # mean*252 convention, not compounded
    return {"beta": beta, "alpha_annualized_pct": alpha_annualized * 100, "r2": r2, "t_alpha": t_alpha, "n": n}


# ============================================================
# FETCH
# ============================================================
print("=" * 100)
print("FETCH")
print("=" * 100)
all_tickers = sorted(set(UNIVERSE_30) | set(ORIGINAL_TICKERS) | {BENCHMARK_ETF})
print(f"Requesting {len(all_tickers)} equity tickers + {MARKET_INDEX} from yfinance, "
      f"{TRAIN_START} to {TEST_END} (daily)...")

daily_raw = yf.download(all_tickers + [MARKET_INDEX], start=TRAIN_START, end=TEST_END,
                          auto_adjust=True, progress=False)["Close"]

missing = [t for t in all_tickers + [MARKET_INDEX] if t not in daily_raw.columns or daily_raw[t].dropna().empty]
if missing:
    print(f"TICKERS THAT FAILED TO FETCH (excluded from every test below): {missing}")
else:
    print("All requested tickers fetched successfully.")

usable_30 = [t for t in UNIVERSE_30 if t not in missing]
usable_original = [t for t in ORIGINAL_TICKERS if t not in missing]
if len(usable_original) < len(ORIGINAL_TICKERS):
    raise SystemExit(f"One or more of the original 6 tickers failed to fetch ({ORIGINAL_TICKERS}) -- "
                      f"cannot run Tests 1/2 without all of them. Missing: {missing}")

print(f"Usable 30-stock universe: {len(usable_30)}/{len(UNIVERSE_30)}")

print(f"Requesting weekly closes for beta ({BETA_WINDOW_START} to {TRAIN_END})...")
weekly_raw = yf.download(usable_30 + [MARKET_INDEX], start=BETA_WINDOW_START, end=TRAIN_END,
                           interval="1wk", auto_adjust=True, progress=False)["Close"]

# ============================================================
# HOLDOUT / WINDOW CONFIRMATION
# ============================================================
print()
print("=" * 100)
print("WINDOW CONFIRMATION")
print("=" * 100)
print(f"Daily panel spans {daily_raw.index.min().date()} to {daily_raw.index.max().date()}")
print(f"Training window (Jensen's-alpha selection, Test 3):  {TRAIN_START} to {TRAIN_END}")
print(f"Beta window (trailing 2yr weekly, Test 3):           {BETA_WINDOW_START} to {TRAIN_END}")
print(f"Walk-forward / OOS test window (Tests 1-4):          {TEST_START} to {TEST_END}")
print("This is the fix for the original START_DATE bug -- Tests 1, 3, and 4 previously ran "
      "on the training window by mistake. All four tests below use ONLY the OOS window for "
      "performance measurement; the training window is read only to compute Test 3's selection.")

test_prices = daily_raw.loc[TEST_START:TEST_END]
train_prices = daily_raw.loc[TRAIN_START:TRAIN_END]

# ============================================================
# TEST 1 -- original portfolio, walk-forward as designed
# ============================================================
print()
print("=" * 100)
print("TEST 1 -- Original portfolio (hindsight-selected, fixed weights), 2024-2025 OOS")
print("=" * 100)
print(f"Tickers: {ORIGINAL_TICKERS}")
print(f"Weights: {ORIGINAL_WEIGHTS}")

t1_value, t1_ret = buy_and_hold_path(test_prices, ORIGINAL_TICKERS, ORIGINAL_WEIGHTS)
t1_stats = report_row("Test 1: Original 6 (hindsight)", t1_value, t1_ret)
print(pd.DataFrame([t1_stats]).to_string(index=False))

# ============================================================
# TEST 2 -- SPY benchmark + alpha regression
# ============================================================
print()
print("=" * 100)
print("TEST 2 -- SPY benchmark (never run originally) + alpha regression")
print("=" * 100)

spy_value, spy_ret = buy_and_hold_path(test_prices, [BENCHMARK_ETF], [1.0])
t2_stats = report_row("Test 2: SPY buy&hold", spy_value, spy_ret)
print(pd.DataFrame([t2_stats]).to_string(index=False))

diff_total = t1_stats["total_return_pct"] - t2_stats["total_return_pct"]
diff_ann = t1_stats["annualized_return_pct"] - t2_stats["annualized_return_pct"]
diff_sharpe = t1_stats["sharpe"] - t2_stats["sharpe"]
print()
print(f"Difference (Original portfolio - SPY): total return {diff_total:+.2f}pp, "
      f"annualized {diff_ann:+.2f}pp, Sharpe {diff_sharpe:+.2f}")

print()
print("--- Alpha regression: Test 1 portfolio daily returns vs SPY daily returns (2024-2025) ---")
reg = alpha_regression_daily(t1_ret, spy_ret)
print(f"Beta:                 {reg['beta']:.3f}")
print(f"Alpha (annualized):   {reg['alpha_annualized_pct']:+.3f}%")
print(f"R-squared:            {reg['r2']:.3f}")
print(f"t-stat on alpha:      {reg['t_alpha']:.2f}   (n={reg['n']} daily observations)")

# ============================================================
# TEST 3 -- honest reselection via Jensen's alpha (2021-2023 only)
# ============================================================
print()
print("=" * 100)
print("TEST 3 -- Honest reselection: Jensen's alpha computed on 2021-2023 data only")
print("=" * 100)

market_train_daily = train_prices[MARKET_INDEX].pct_change().dropna()
market_train_weekly = weekly_raw[MARKET_INDEX].pct_change().dropna()

selection_rows = []
for ticker in usable_30:
    stock_daily = train_prices[ticker].pct_change().dropna()
    ann_ret = annualized_return(stock_daily)

    stock_weekly = weekly_raw[ticker].pct_change().dropna()
    aligned = pd.concat([stock_weekly.rename("s"), market_train_weekly.rename("m")], axis=1).dropna()
    beta = float(np.cov(aligned["m"], aligned["s"], ddof=1)[0, 1] / np.var(aligned["m"], ddof=1))

    jensens_alpha = ann_ret - (RISK_FREE + beta * (MARKET_RET_ASSUMED - RISK_FREE))
    selection_rows.append({"ticker": ticker, "ann_return_2021_2023_pct": ann_ret * 100,
                            "beta_2yr_weekly": beta, "jensens_alpha_pct": jensens_alpha * 100})

sel_df = pd.DataFrame(selection_rows).sort_values("jensens_alpha_pct", ascending=False).reset_index(drop=True)
print("Full 30-stock ranking by Jensen's alpha (2021-2023 data only, no lookahead):")
print(sel_df.to_string(index=False))

top6_honest = sel_df.head(6)["ticker"].tolist()
print()
print(f"Top 6 by Jensen's alpha as of end-2023 (honest, equal-weighted): {top6_honest}")
print(f"Original 6 (chosen using data through 2025 -- hindsight):        {ORIGINAL_TICKERS}")
overlap = sorted(set(top6_honest) & set(ORIGINAL_TICKERS))
print(f"Overlap between the two lists: {overlap if overlap else '(none)'}")

t3_value, t3_ret = buy_and_hold_path(test_prices, top6_honest, [1 / 6] * 6)
t3_stats = report_row("Test 3: Honest top-6 (2021-2023 selection)", t3_value, t3_ret)
print()
print(pd.DataFrame([t3_stats]).to_string(index=False))

hindsight_gap_total = t1_stats["total_return_pct"] - t3_stats["total_return_pct"]
hindsight_gap_ann = t1_stats["annualized_return_pct"] - t3_stats["annualized_return_pct"]
hindsight_gap_sharpe = t1_stats["sharpe"] - t3_stats["sharpe"]
print()
print(f"Hindsight advantage (Original - Honest reselection): total return {hindsight_gap_total:+.2f}pp, "
      f"annualized {hindsight_gap_ann:+.2f}pp, Sharpe {hindsight_gap_sharpe:+.2f}")

# ============================================================
# TEST 4 -- equal-weighted all 30, the no-skill baseline
# ============================================================
print()
print("=" * 100)
print("TEST 4 -- Equal-weighted all 30 candidates, 2024-2025 (no-skill baseline)")
print("=" * 100)

t4_value, t4_ret = buy_and_hold_path(test_prices, usable_30, [1 / len(usable_30)] * len(usable_30))
t4_stats = report_row(f"Test 4: Equal-weight all {len(usable_30)}", t4_value, t4_ret)
print(pd.DataFrame([t4_stats]).to_string(index=False))

# ============================================================
# SIDE-BY-SIDE SUMMARY
# ============================================================
print()
print("=" * 100)
print("SIDE-BY-SIDE SUMMARY -- all four tests + SPY, identical 2024-2025 window")
print("=" * 100)
summary = pd.DataFrame([t1_stats, t2_stats, t3_stats, t4_stats])
print(summary.to_string(index=False))

print()
print("=" * 100)
print("CONCLUSION")
print("=" * 100)
beat_spy = t1_stats["total_return_pct"] > t2_stats["total_return_pct"]
print(f"1) Did the original portfolio beat SPY out-of-sample? "
      f"{'YES' if beat_spy else 'NO'} "
      f"({t1_stats['total_return_pct']:+.2f}% vs {t2_stats['total_return_pct']:+.2f}%, "
      f"alpha t-stat {reg['t_alpha']:.2f}).")
print(f"2) Hindsight/selection-bias advantage baked into the original 6 vs an honest, "
      f"2021-2023-only reselection: {hindsight_gap_total:+.2f}pp total return, "
      f"{hindsight_gap_sharpe:+.2f} Sharpe.")
beat_baseline_orig = t1_stats["total_return_pct"] > t4_stats["total_return_pct"]
beat_baseline_honest = t3_stats["total_return_pct"] > t4_stats["total_return_pct"]
print(f"3) Vs. the honest no-skill baseline (equal-weight all 30): original portfolio "
      f"{'beat' if beat_baseline_orig else 'did NOT beat'} it "
      f"({t1_stats['total_return_pct']:+.2f}% vs {t4_stats['total_return_pct']:+.2f}%); "
      f"the honestly-reselected top-6 {'beat' if beat_baseline_honest else 'did NOT beat'} it "
      f"({t3_stats['total_return_pct']:+.2f}% vs {t4_stats['total_return_pct']:+.2f}%).")
print()
print("Weights and ticker list were not tuned or changed from the original specification; "
      "only the test window (START_DATE bug) and the missing SPY benchmark were fixed.")
