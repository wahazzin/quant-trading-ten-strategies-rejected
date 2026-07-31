"""
backtest_universe.py -- run the SAME monkey-tested SMA crossover across
ALL verified stocks at once. This is the first test that can tell a
REAL edge (works broadly) from LUCK (works on one lucky stock).

For every stock:
  - split 70% practice / 30% locked holdout (we only touch practice here)
  - SMA 20/50 crossover, trade next-day OPEN (no lookahead), 0.2% cost
  - compare strategy vs buy-and-hold
  - run 500 random monkeys, report what % we beat
"""
import os
import pandas as pd
import numpy as np

UNIVERSE = ["SBRA", "VLY", "FLO", "AROC", "HUN", "WEN",
            "CLF", "MGNI", "KSS", "TROX", "VSH", "UAA", "HL"]
FAST, SLOW = 20, 50
COST = 0.002
TUNE = 0.70
N_MONKEY = 500
rng = np.random.default_rng(42)


def backtest(df):
    df = df.copy()
    df["f"] = df["close"].rolling(FAST).mean()
    df["s"] = df["close"].rolling(SLOW).mean()
    opens = df["open"].to_numpy()
    in_pos, entry_i, trades = False, None, []
    for i in range(SLOW, len(df) - 1):
        fp, sp = df["f"].iloc[i-1], df["s"].iloc[i-1]
        fn, sn = df["f"].iloc[i], df["s"].iloc[i]
        if not in_pos and fp <= sp and fn > sn:
            in_pos, entry_i = True, i + 1
        elif in_pos and fp >= sp and fn < sn:
            trades.append((entry_i, i + 1))
            in_pos = False
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
    bh_entry, bh_exit = SLOW, len(opens) - 1
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
