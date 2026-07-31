"""
backtest_rsi.py -- run the SAME monkey-tested methodology as
backtest_universe.py, but with signal #2 of the pre-registered budget
of 3: an RSI(14) mean-reversion strategy, across ALL verified stocks.

Hypothesis: our universe is volatility-heavy, where trend-following
(signal #1, SMA 20/50) gets whipsawed and mean-reversion is
theoretically better suited.

For every stock:
  - split 70% practice / 30% locked holdout (we only touch practice here)
  - RSI(14) mean-reversion, trade next-day OPEN (no lookahead), 0.2% cost
  - compare strategy vs buy-and-hold
  - run 500 random monkeys, report what % we beat
"""
import os
import pandas as pd
import numpy as np

UNIVERSE = ["SBRA", "VLY", "FLO", "AROC", "HUN", "WEN",
            "CLF", "MGNI", "KSS", "TROX", "VSH", "UAA", "HL"]
RSI_PERIOD = 14
RSI_OVERSOLD = 30   # entry: RSI crosses up through this level
RSI_EXIT = 55       # exit: RSI closes above this level
MAX_HOLD = 20       # exit: forced out after this many trading days
COST = 0.002
TUNE = 0.70
N_MONKEY = 500
rng = np.random.default_rng(42)


def compute_rsi(close, period=RSI_PERIOD):
    """Standard Wilder RSI: seed with a simple average of the first
    `period` gains/losses, then smooth recursively (Wilder's method)."""
    n = len(close)
    rsi = np.full(n, np.nan)
    if n <= period:
        return rsi
    delta = np.diff(close)  # delta[i] = close[i+1] - close[i]
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = gain[:period].mean()
    avg_loss = loss[:period].mean()
    rsi[period] = _rsi_from_avg(avg_gain, avg_loss)

    for i in range(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        rsi[i + 1] = _rsi_from_avg(avg_gain, avg_loss)
    return rsi


def _rsi_from_avg(avg_gain, avg_loss):
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def backtest(df):
    df = df.copy()
    close = df["close"].to_numpy()
    opens = df["open"].to_numpy()
    rsi = compute_rsi(close, RSI_PERIOD)
    in_pos, entry_i, trades = False, None, []
    for i in range(RSI_PERIOD + 1, len(df) - 1):
        if in_pos:
            days_held = i - entry_i
            if rsi[i] > RSI_EXIT or days_held >= MAX_HOLD:
                trades.append((entry_i, i + 1))
                in_pos = False
                continue
        if not in_pos and rsi[i-1] <= RSI_OVERSOLD and rsi[i] > RSI_OVERSOLD:
            in_pos, entry_i = True, i + 1
    if in_pos:
        trades.append((entry_i, len(df) - 1))
    rets = np.array([opens[x]/opens[e] - 1 - COST for e, x in trades])
    return trades, rets, opens


def monkey_test(opens, holding_lengths, n_monkey, rng):
    """500 random traders: same number of trades, same holding lengths
    as the real strategy, but entered at random dates. Returns each
    monkey's total compounded return so we can see how many the
    strategy actually beat."""
    n = len(opens)
    monkey_returns = np.empty(n_monkey)
    for m in range(n_monkey):
        total = 1.0
        for hl in holding_lengths:
            max_start = n - 1 - hl
            if max_start < 0:
                continue
            e_rand = rng.integers(0, max_start + 1)
            x_rand = e_rand + hl
            r = opens[x_rand] / opens[e_rand] - 1 - COST
            total *= (1 + r)
        monkey_returns[m] = total - 1
    return monkey_returns


results = []
skipped = []

for ticker in UNIVERSE:
    path = os.path.join("data", f"{ticker}_daily.csv")
    if not os.path.exists(path):
        print(f"{ticker}: SKIPPED (missing CSV at {path})")
        skipped.append((ticker, "missing CSV"))
        continue

    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    n_tune = int(len(df) * TUNE)
    tune_df = df.iloc[:n_tune].reset_index(drop=True)

    trades, rets, opens = backtest(tune_df)

    if len(trades) == 0:
        print(f"{ticker}: SKIPPED (zero trades generated on tune window)")
        skipped.append((ticker, "zero trades"))
        continue

    num_trades = len(trades)
    win_rate = float(np.mean(rets > 0))
    expectancy = float(np.mean(rets))
    strat_return = float(np.prod(1 + rets) - 1)

    # buy-and-hold over the same window the strategy actually trades
    # (from the first bar it could act on, to the last bar available),
    # with the same round-trip cost assumption.
    bh_entry, bh_exit = RSI_PERIOD, len(opens) - 1
    buy_hold_return = float(opens[bh_exit] / opens[bh_entry] - 1 - COST)

    holding_lengths = [x - e for e, x in trades]
    monkey_returns = monkey_test(opens, holding_lengths, N_MONKEY, rng)
    monkey_beat_pct = float(np.mean(monkey_returns < strat_return) * 100)

    results.append({
        "ticker": ticker,
        "trades": num_trades,
        "win_rate": win_rate,
        "expectancy": expectancy,
        "strat_return": strat_return,
        "buy_hold_return": buy_hold_return,
        "monkey_beat_pct": monkey_beat_pct,
    })

header = f"{'Ticker':<8}{'Trades':>7}{'WinRate':>10}{'Expectncy':>12}{'StratRet':>11}{'BuyHold':>10}{'Monkey%':>10}"
print()
print(header)
print("-" * len(header))
for r in results:
    print(
        f"{r['ticker']:<8}{r['trades']:>7}"
        f"{r['win_rate']*100:>9.1f}%"
        f"{r['expectancy']*100:>11.2f}%"
        f"{r['strat_return']*100:>10.1f}%"
        f"{r['buy_hold_return']*100:>9.1f}%"
        f"{r['monkey_beat_pct']:>9.1f}%"
    )

if skipped:
    print()
    print("Skipped:")
    for ticker, reason in skipped:
        print(f"  {ticker}: {reason}")

n = len(results)
beat_bh = sum(1 for r in results if r["strat_return"] > r["buy_hold_return"])
beat_monkey60 = sum(1 for r in results if r["monkey_beat_pct"] > 60)
avg_expectancy = float(np.mean([r["expectancy"] for r in results])) if n else float("nan")
avg_monkey_beat = float(np.mean([r["monkey_beat_pct"] for r in results])) if n else float("nan")

print()
print("=== Summary across universe ===")
print(f"Stocks tested:                 {n} / {len(UNIVERSE)}")
print(f"Beat buy-and-hold:             {beat_bh} / {n}")
print(f"Beat >60% of monkeys:          {beat_monkey60} / {n}")
print(f"Average expectancy per trade:  {avg_expectancy*100:.3f}%")
print(f"Average monkey-beat %:         {avg_monkey_beat:.1f}%")
