"""
swedish_retest.py -- retest of the original swedish_capm.py watchlist
(SPOT, ERIC, AZN, ALV, OTLY -- all US-listed ADRs of Nordic-connected
companies), using the identical conventions established in
fugazzi_retest.py so the two retests are directly comparable:
  - Daily returns; annualized return = mean(daily_ret) * 252,
    annualized vol = std(daily_ret) * sqrt(252) (simple scaling, not
    geometric compounding).
  - No rebalancing: BUDGET * weight invested per stock at day one, then
    (1 + pct_change).cumprod() * weight * BUDGET per stock, summed --
    fixed-share buy-and-hold, weights drift with relative performance.
  - Beta: 2-year weekly returns vs ^GSPC, computed over the window
    immediately PRECEDING the test period (2022-01-01 to 2023-12-31) --
    the CAPM weights must be computable before 2024-01-01 without
    looking into the test window itself, exactly as a pre-registered
    watchlist script would have had to do it live.

Three variants, all over the 2024-01-01 to 2025-12-31 test window:
  (a) Equal-weighted (1/5 each) -- the no-opinion baseline.
  (b) CAPM-weighted, the original formula:
        expected_return = 0.045 + beta * 0.055
        weight_i = expected_return_i / sum(expected_return)
      Beta and expected_return are both computed ONLY from the
      pre-test window (2022-2023) -- the weights are fixed before the
      test period starts, not fit to it.
  (c) SPY buy-and-hold -- the benchmark.

Full alpha regression (beta, annualized alpha, R^2, t-stat) vs SPY is
reported for (a) and (b), using the same daily / mean*252 convention as
fugazzi_retest.py's regression.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = ["SPOT", "ERIC", "AZN", "ALV", "OTLY"]
MARKET_INDEX = "^GSPC"
BENCHMARK_ETF = "SPY"

BETA_START = "2022-01-01"
BETA_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2025-12-31"

RISK_FREE = 0.045
RISK_PREMIUM = 0.055  # market_ret(0.10) - risk_free(0.045), matching fugazzi_retest.py's constants
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
    alpha_annualized = alpha_daily * TRADING_DAYS
    return {"beta": beta, "alpha_annualized_pct": alpha_annualized * 100, "r2": r2, "t_alpha": t_alpha, "n": n}


# ============================================================
# FETCH
# ============================================================
print("=" * 100)
print("FETCH")
print("=" * 100)
all_tickers = TICKERS + [BENCHMARK_ETF, MARKET_INDEX]
print(f"Requesting {all_tickers} from yfinance, {BETA_START} to {TEST_END} (daily)...")

daily_raw = yf.download(all_tickers, start=BETA_START, end=TEST_END, auto_adjust=True, progress=False)["Close"]
missing = [t for t in all_tickers if t not in daily_raw.columns or daily_raw[t].dropna().empty]
if missing:
    raise SystemExit(f"TICKERS THAT FAILED TO FETCH: {missing} -- cannot proceed without all of them.")
print("All requested tickers fetched successfully.")

print(f"Requesting weekly closes for beta ({BETA_START} to {BETA_END})...")
weekly_raw = yf.download(TICKERS + [MARKET_INDEX], start=BETA_START, end=BETA_END,
                           interval="1wk", auto_adjust=True, progress=False)["Close"]

print()
print("=" * 100)
print("WINDOW CONFIRMATION")
print("=" * 100)
print(f"Beta window (pre-test, 2-year weekly): {BETA_START} to {BETA_END}")
print(f"Test window (all 3 variants):          {TEST_START} to {TEST_END}")
print("CAPM weights are computed ONLY from the beta window, before the test window begins -- "
      "no data from 2024-2025 enters variant (b)'s weight calculation.")

test_prices = daily_raw.loc[TEST_START:TEST_END]

# ============================================================
# CAPM WEIGHTS (variant b)
# ============================================================
print()
print("=" * 100)
print("CAPM WEIGHT CALCULATION (pre-test window only)")
print("=" * 100)
market_weekly = weekly_raw[MARKET_INDEX].pct_change().dropna()

capm_rows = []
for ticker in TICKERS:
    stock_weekly = weekly_raw[ticker].pct_change().dropna()
    aligned = pd.concat([stock_weekly.rename("s"), market_weekly.rename("m")], axis=1).dropna()
    beta = float(np.cov(aligned["m"], aligned["s"], ddof=1)[0, 1] / np.var(aligned["m"], ddof=1))
    expected_return = RISK_FREE + beta * RISK_PREMIUM
    capm_rows.append({"ticker": ticker, "beta_2yr_weekly": beta, "expected_return_pct": expected_return * 100})

capm_df = pd.DataFrame(capm_rows)
total_expected = capm_df["expected_return_pct"].sum()
capm_df["capm_weight"] = capm_df["expected_return_pct"] / total_expected
print(capm_df.to_string(index=False))
if (capm_df["expected_return_pct"] < 0).any():
    print("\nNOTE: at least one expected_return is negative -- this produces a negative or distorted "
          "weight under the literal weight_i = expected_return_i / sum(expected_return) formula. "
          "Reported as-is, per the original formula, not patched.")

capm_weights = capm_df.set_index("ticker")["capm_weight"].reindex(TICKERS).tolist()

# ============================================================
# VARIANT (a) -- Equal-weighted
# ============================================================
print()
print("=" * 100)
print("VARIANT (a) -- Equal-weighted, 2024-2025")
print("=" * 100)
eq_value, eq_ret = buy_and_hold_path(test_prices, TICKERS, [1 / len(TICKERS)] * len(TICKERS))
eq_stats = report_row("(a) Equal-weighted", eq_value, eq_ret)
print(pd.DataFrame([eq_stats]).to_string(index=False))

# ============================================================
# VARIANT (b) -- CAPM-weighted
# ============================================================
print()
print("=" * 100)
print("VARIANT (b) -- CAPM-weighted (pre-test beta), 2024-2025")
print("=" * 100)
print(f"Weights: {dict(zip(TICKERS, [round(w, 4) for w in capm_weights]))}")
capm_value, capm_ret = buy_and_hold_path(test_prices, TICKERS, capm_weights)
capm_stats = report_row("(b) CAPM-weighted", capm_value, capm_ret)
print(pd.DataFrame([capm_stats]).to_string(index=False))

# ============================================================
# VARIANT (c) -- SPY benchmark
# ============================================================
print()
print("=" * 100)
print("VARIANT (c) -- SPY benchmark, 2024-2025")
print("=" * 100)
spy_value, spy_ret = buy_and_hold_path(test_prices, [BENCHMARK_ETF], [1.0])
spy_stats = report_row("(c) SPY buy&hold", spy_value, spy_ret)
print(pd.DataFrame([spy_stats]).to_string(index=False))

# ============================================================
# ALPHA REGRESSIONS vs SPY
# ============================================================
print()
print("=" * 100)
print("ALPHA REGRESSIONS vs SPY")
print("=" * 100)
for label, ret in [("(a) Equal-weighted", eq_ret), ("(b) CAPM-weighted", capm_ret)]:
    reg = alpha_regression_daily(ret, spy_ret)
    print(f"--- {label} ---")
    print(f"Beta:                 {reg['beta']:.3f}")
    print(f"Alpha (annualized):   {reg['alpha_annualized_pct']:+.3f}%")
    print(f"R-squared:            {reg['r2']:.3f}")
    print(f"t-stat on alpha:      {reg['t_alpha']:.2f}   (n={reg['n']} daily observations)")
    print()

# ============================================================
# SIDE-BY-SIDE SUMMARY
# ============================================================
print("=" * 100)
print("SIDE-BY-SIDE SUMMARY -- all three variants, identical 2024-2025 window")
print("=" * 100)
summary = pd.DataFrame([eq_stats, capm_stats, spy_stats])
print(summary.to_string(index=False))

print()
print("Weights and ticker list were not tuned or changed from the original specification.")
