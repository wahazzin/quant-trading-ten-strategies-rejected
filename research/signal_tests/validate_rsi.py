"""
validate_rsi.py -- three validation checks on the RSI(14) mean-reversion
signal (backtest_rsi.py) before committing to it. All three run ONLY on
each stock's 70% practice window; the 30% locked holdout is never read.

Check 1: pool every trade return from all 13 stocks and test the pooled
         mean against zero (t-stat + bootstrap CI), then run a pooled
         monkey test (2,000 simulated universes) for a percentile rank.
Check 2: sweep RSI period x oversold x exit thresholds (27 combos) and
         report how the average monkey-beat % is distributed across the
         neighborhood. This does NOT change the registered parameters
         (14/30/55) and makes no recommendation -- it only answers
         "is the result stable, or a knife-edge fluke?"
Check 3: build the equal-weighted (1/13 each), independently-traded
         portfolio equity curve at the registered 14/30/55 parameters
         and report return, max drawdown, longest drawdown, and Sharpe.
"""
import os
import io
import contextlib
import numpy as np
import pandas as pd

# Reuse the RSI computation and backtest/monkey-test logic from
# backtest_rsi.py verbatim -- nothing here re-derives the strategy.
# (Import side effects -- the original script's own printout -- are
# suppressed so this script's output isn't cluttered by a re-run of it.)
with contextlib.redirect_stdout(io.StringIO()):
    import backtest_rsi as rsi_mod

# save the registered parameters so Check 2's sweep can restore them
REG_PERIOD, REG_OVERSOLD, REG_EXIT = rsi_mod.RSI_PERIOD, rsi_mod.RSI_OVERSOLD, rsi_mod.RSI_EXIT

rng = np.random.default_rng(42)


def load_tune_data():
    data = {}
    for ticker in rsi_mod.UNIVERSE:
        path = os.path.join("data", f"{ticker}_daily.csv")
        if not os.path.exists(path):
            print(f"{ticker}: SKIPPED (missing CSV at {path})")
            continue
        df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        n_tune = int(len(df) * rsi_mod.TUNE)
        data[ticker] = df.iloc[:n_tune].reset_index(drop=True)
    return data


DATA = load_tune_data()


# ============================================================
# CHECK 1 -- pooled statistical test (registered 14/30/55 params)
# ============================================================
print()
print("=" * 70)
print("CHECK 1: Pooled statistical test (14/30/55, practice window only)")
print("=" * 70)

pooled_rets = []
stock_sim_inputs = []   # (opens, holding_lengths) per stock, for the pooled monkey test
for ticker, tune_df in DATA.items():
    trades, rets, opens = rsi_mod.backtest(tune_df)
    if len(trades) == 0:
        print(f"{ticker}: SKIPPED (zero trades)")
        continue
    pooled_rets.extend(rets.tolist())
    stock_sim_inputs.append((opens, [x - e for e, x in trades]))

pooled_rets = np.array(pooled_rets)
n_trades = len(pooled_rets)
mean_ret = float(pooled_rets.mean())
std_ret = float(pooled_rets.std(ddof=1))
t_stat = mean_ret / (std_ret / np.sqrt(n_trades))

BOOT_N = 10_000
idx = rng.integers(0, n_trades, size=(BOOT_N, n_trades))
boot_means = pooled_rets[idx].mean(axis=1)
ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

print(f"Total trades pooled:        {n_trades}")
print(f"Pooled mean return/trade:   {mean_ret*100:.3f}%")
print(f"Std dev of trade returns:   {std_ret*100:.3f}%")
print(f"t-statistic vs zero:        {t_stat:.3f}  (df={n_trades-1})")
print(f"Bootstrap 95% CI on mean:   [{ci_low*100:.3f}%, {ci_high*100:.3f}%]  ({BOOT_N:,} resamples)")

# pooled monkey test: 2,000 simulated universes, same trade counts &
# holding lengths per stock, random entry dates
N_POOLED_SIMS = 2000


def pooled_monkey_sim(stock_data, n_sims, rng, cost):
    sim_means = np.empty(n_sims)
    for s in range(n_sims):
        sim_rets = []
        for opens, holding_lengths in stock_data:
            n_bars = len(opens)
            for hl in holding_lengths:
                max_start = n_bars - 1 - hl
                if max_start < 0:
                    continue
                e_r = rng.integers(0, max_start + 1)
                x_r = e_r + hl
                sim_rets.append(opens[x_r] / opens[e_r] - 1 - cost)
        sim_means[s] = np.mean(sim_rets) if sim_rets else np.nan
    return sim_means


sim_means = pooled_monkey_sim(stock_sim_inputs, N_POOLED_SIMS, rng, rsi_mod.COST)
percentile = float(np.mean(sim_means < mean_ret) * 100)
print(f"Pooled monkey test:         {N_POOLED_SIMS:,} simulated universes")
print(f"  -> actual pooled mean beats {percentile:.1f}% of simulated universes")


# ============================================================
# CHECK 2 -- parameter stability grid (report only, no selection)
# ============================================================
print()
print("=" * 70)
print("CHECK 2: Parameter stability grid (report only -- registered params")
print("          remain 14/30/55; nothing here is selected or recommended)")
print("=" * 70)

PERIODS = [10, 14, 21]
OVERSOLDS = [25, 30, 35]
EXITS = [50, 55, 60]

grid_results = []
for p in PERIODS:
    for ov in OVERSOLDS:
        for ex in EXITS:
            rsi_mod.RSI_PERIOD, rsi_mod.RSI_OVERSOLD, rsi_mod.RSI_EXIT = p, ov, ex
            beat_pcts = []
            for ticker, tune_df in DATA.items():
                trades, rets, opens = rsi_mod.backtest(tune_df)
                if len(trades) == 0:
                    continue
                strat_return = float(np.prod(1 + rets) - 1)
                holding_lengths = [x - e for e, x in trades]
                monkey_returns = rsi_mod.monkey_test(opens, holding_lengths, rsi_mod.N_MONKEY, rng)
                beat_pcts.append(float(np.mean(monkey_returns < strat_return) * 100))
            avg_beat = float(np.mean(beat_pcts)) if beat_pcts else float("nan")
            grid_results.append({
                "period": p, "oversold": ov, "exit": ex,
                "avg_monkey_beat": avg_beat, "n_stocks": len(beat_pcts),
            })

# restore the registered parameters -- Check 3 depends on this
rsi_mod.RSI_PERIOD, rsi_mod.RSI_OVERSOLD, rsi_mod.RSI_EXIT = REG_PERIOD, REG_OVERSOLD, REG_EXIT

print(f"{'Period':>7}{'Oversold':>10}{'Exit':>7}{'AvgMonkey%':>13}{'nStocks':>9}")
for g in grid_results:
    tag = "  <- registered" if (g["period"], g["oversold"], g["exit"]) == (REG_PERIOD, REG_OVERSOLD, REG_EXIT) else ""
    print(f"{g['period']:>7}{g['oversold']:>10}{g['exit']:>7}{g['avg_monkey_beat']:>12.1f}%{g['n_stocks']:>9}{tag}")

beats = np.array([g["avg_monkey_beat"] for g in grid_results])
print()
print(f"Distribution across {len(grid_results)} combos:")
print(f"  min:    {beats.min():.1f}%")
print(f"  median: {np.median(beats):.1f}%")
print(f"  max:    {beats.max():.1f}%")
print(f"  combos exceeding 55%: {int(np.sum(beats > 55))} / {len(grid_results)}")


# ============================================================
# CHECK 3 -- portfolio drawdown at registered 14/30/55 parameters
# ============================================================
print()
print("=" * 70)
print("CHECK 3: Equal-weighted portfolio drawdown (14/30/55, practice window)")
print("=" * 70)

common_n = min(len(tune_df) for tune_df in DATA.values())
equity_curves = []
for ticker, tune_df in DATA.items():
    trades, rets, opens = rsi_mod.backtest(tune_df)
    n_bars = len(tune_df)
    equity = np.ones(n_bars)
    cur, last_exit = 1.0, 0
    for e, x in trades:
        equity[last_exit:x] = cur
        cur *= (1 + (opens[x] / opens[e] - 1 - rsi_mod.COST))
        last_exit = x
    equity[last_exit:] = cur
    equity_curves.append(equity[:common_n])

portfolio_equity = np.mean(equity_curves, axis=0)
total_return = float(portfolio_equity[-1] - 1)

running_max = np.maximum.accumulate(portfolio_equity)
drawdown = portfolio_equity / running_max - 1
max_dd = float(drawdown.min())

underwater = drawdown < -1e-12
longest_dd, cur_run = 0, 0
for u in underwater:
    if u:
        cur_run += 1
        longest_dd = max(longest_dd, cur_run)
    else:
        cur_run = 0

daily_rets = np.diff(portfolio_equity) / portfolio_equity[:-1]
sharpe = float(daily_rets.mean() / daily_rets.std() * np.sqrt(252)) if daily_rets.std() > 0 else float("nan")

print(f"Stocks in portfolio:        {len(equity_curves)} (equal-weighted, 1/{len(equity_curves)} each)")
print(f"Portfolio window length:    {common_n} trading days")
print(f"Total portfolio return:     {total_return*100:.2f}%")
print(f"Maximum drawdown:           {max_dd*100:.2f}%")
print(f"Longest drawdown duration:  {longest_dd} trading days")
print(f"Sharpe ratio (annualized):  {sharpe:.3f}")
