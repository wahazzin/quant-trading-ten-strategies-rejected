"""
backtest_sma.py -- FIRST REAL SIGNAL TEST (rule 3 + rule 9).

Tests ONE thing in isolation: does the SMA 20/50 crossover on Ford
daily bars have any edge, or is it indistinguishable from luck?

Honesty features:
 1. Evidence boundary: last 30% of data is LOCKED (holdout). We only
    compute on the first 70%. The holdout is the final exam we take
    exactly once, much later, after we stop tweaking.
 2. No lookahead: a crossover confirmed at day N's CLOSE trades at
    day N+1's OPEN. You cannot act on a price before it exists.
 3. Costs: 0.2% per round trip (spread + commission estimate).
 4. Null test: 2000 simulated random traders, same number of trades,
    same holding lengths, random timing. If we can't beat them,
    the signal is noise.
"""
import pandas as pd
import numpy as np

TUNE_FRACTION = 0.70
FAST, SLOW = 20, 50
COST_PCT = 0.002          # per round trip
N_RANDOM = 2000
rng = np.random.default_rng(42)

df = pd.read_csv("data/F_daily.csv", parse_dates=["date"])
df = df.iloc[:-1].reset_index(drop=True)   # drop 2026-07-20 partial bar

split = int(len(df) * TUNE_FRACTION)
tune = df.iloc[:split].reset_index(drop=True)
holdout_first_day = df["date"].iloc[split].date()

print(f"Total clean bars: {len(df)}")
print(f"Practice field:   {len(tune)} bars "
      f"({tune['date'].iloc[0].date()} -> {tune['date'].iloc[-1].date()})")
print(f"FINAL EXAM locked from {holdout_first_day} onward -- untouched.\n")

tune["sma_f"] = tune["close"].rolling(FAST).mean()
tune["sma_s"] = tune["close"].rolling(SLOW).mean()

in_pos = False
entry_i = None
entry_px = None
trades = []   # (entry_index, exit_index, net_return)

for i in range(SLOW, len(tune) - 1):
    f_prev, s_prev = tune["sma_f"].iloc[i - 1], tune["sma_s"].iloc[i - 1]
    f_now, s_now = tune["sma_f"].iloc[i], tune["sma_s"].iloc[i]

    if not in_pos and f_prev <= s_prev and f_now > s_now:
        in_pos = True
        entry_i = i + 1                       # act NEXT day...
        entry_px = tune["open"].iloc[i + 1]   # ...at the OPEN (no lookahead)
    elif in_pos and f_prev >= s_prev and f_now < s_now:
        exit_i = i + 1
        exit_px = tune["open"].iloc[i + 1]
        trades.append((entry_i, exit_i, exit_px / entry_px - 1 - COST_PCT))
        in_pos = False

if in_pos:   # still holding at the boundary -> close at final practice bar
    exit_i = len(tune) - 1
    exit_px = tune["close"].iloc[-1]
    trades.append((entry_i, exit_i, exit_px / entry_px - 1 - COST_PCT))

print("=== TRADES (practice field only) ===")
for e, x, r in trades:
    print(f"  {tune['date'].iloc[e].date()} -> {tune['date'].iloc[x].date()}"
          f"  ({x - e:3d} days held)  {r * 100:+6.2f}%")

n = len(trades)
rets = np.array([t[2] for t in trades])
wins = rets[rets > 0]
losses = rets[rets <= 0]
total = float(np.prod(1 + rets) - 1) if n else 0.0

win_rate = len(wins) / n if n else 0.0
avg_win = float(wins.mean()) if len(wins) else 0.0
avg_loss = float(losses.mean()) if len(losses) else 0.0
expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

bh_entry = tune["open"].iloc[SLOW + 1]
bh = float(tune["close"].iloc[-1] / bh_entry - 1 - COST_PCT)

print(f"\nTrades: {n} | Win rate: {win_rate * 100:.0f}%")
print(f"Avg win: {avg_win * 100:+.2f}% | Avg loss: {avg_loss * 100:+.2f}%")
print(f"Expectancy per trade: {expectancy * 100:+.2f}%   (rule 6 number)")
print(f"Strategy total return: {total * 100:+.2f}%")
print(f"Buy-and-hold, same window: {bh * 100:+.2f}%")

if n:
    opens = tune["open"].to_numpy()
    hold_lens = [x - e for e, x, _ in trades]
    sims = np.empty(N_RANDOM)
    for k in range(N_RANDOM):
        tot = 1.0
        for h in hold_lens:
            s = int(rng.integers(SLOW + 1, len(tune) - h))
            r = opens[s + h] / opens[s] - 1 - COST_PCT
            tot *= 1 + r
        sims[k] = tot - 1

    beat = float((sims < total).mean() * 100)
    print(f"\n=== RANDOM-MONKEY TEST ({N_RANDOM} simulated traders) ===")
    print("Each monkey: same number of trades, same holding lengths,")
    print("random entry dates, same costs. Pure luck, no signal.")
    print(f"Median monkey return: {np.median(sims) * 100:+.2f}%")
    print(f"Best monkey: {sims.max() * 100:+.2f}% | "
          f"Worst: {sims.min() * 100:+.2f}%")
    print(f"Our SMA crossover beats {beat:.0f}% of the monkeys.")
    print("Read: ~50% = coin flip. Below ~80% = likely just luck.")
else:
    print("\nNo trades generated -- nothing to test.")
