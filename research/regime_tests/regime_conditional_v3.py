"""
regime_conditional_v3.py -- Test 17c

FIXES A BIAS IN MY OWN CONTROL that favored the strategy.

v2's monkey test matched HOLD LENGTHS but used random entries. That is
not a fair control, because the strategy's hold length is ENDOGENOUS:
it exits when RSI > 55, so a short hold happens precisely BECAUSE price
rose quickly. Short hold = winner by construction. Handing monkeys the
same hold distribution with blind entries gives the strategy an
outcome-selected exit and the monkey a scheduled one.

v3 CONTROL: monkeys enter on a RANDOM eligible day in the same regime,
then follow the IDENTICAL exit rule (RSI > 55 or 20 days) on that same
stock's real RSI series. Hold length becomes endogenous for both. The
ONLY difference is the entry signal -- which is exactly what we want to
isolate.

ALSO: v2's survivorship diagnostic was confounded -- the oldest bucket
(2006-2009) is also the financial crisis, and dip-buying fails in a
crash for reasons unrelated to survivorship. v3 reports the diagnostic
BOTH with and without 2008-2009 to decouple the two.

PRE-REGISTERED (unchanged):
  1. choppy expectancy > unconditional
  2. choppy t > 2.0
  3. beats > 70% of regime-matched, SAME-EXIT-RULE monkeys
  4. exceeds 0.05% cost
"""

import numpy as np
import pandas as pd

TRAIL, RSI_PERIOD = 60, 14
RSI_ENTRY, RSI_EXIT, MAX_HOLD = 30, 55, 20
COST = 0.0005
PRICE_MIN, PRICE_MAX, MIN_VOL = 5.0, 100.0, 250_000
N_MONKEY = 400
PRACTICE_END = "2018-12-31"
rng = np.random.default_rng(42)

print("=" * 68)
print("TEST 17c -- FAIR MONKEY CONTROL (same exit rule both sides)")
print("=" * 68, flush=True)

raw = pd.read_parquet("data/yf_universe.parquet",
                      columns=["ticker", "date", "open", "close", "volume"])
raw["date"] = pd.to_datetime(raw["date"])
raw = raw[raw["date"] <= PRACTICE_END]
raw = raw[(raw["close"] > 0) & (raw["open"] > 0)]
raw = raw.sort_values(["ticker", "date"]).reset_index(drop=True)
raw["adv20"] = raw.groupby("ticker")["volume"].transform(
    lambda s: s.rolling(20, min_periods=20).mean())
raw["eligible"] = ((raw["close"] >= PRICE_MIN) &
                   (raw["close"] <= PRICE_MAX) &
                   (raw["adv20"] >= MIN_VOL))


def wilder_rsi(close, period=RSI_PERIOD):
    n = len(close); out = np.full(n, np.nan)
    if n < period + 1:
        return out
    d = np.diff(close)
    g = np.where(d > 0, d, 0.0); l = np.where(d < 0, -d, 0.0)
    ag, al = g[:period].mean(), l[:period].mean()
    for i in range(period, n - 1):
        ag = (ag * (period - 1) + g[i]) / period
        al = (al * (period - 1) + l[i]) / period
        rs = ag / al if al > 0 else np.inf
        out[i + 1] = 100.0 - 100.0 / (1.0 + rs)
    return out


print("Computing RSI...", flush=True)
parts = []
for i, (tk, g) in enumerate(raw.groupby("ticker", sort=False)):
    if i % 400 == 0:
        print(f"  {i}...", flush=True)
    parts.append(pd.Series(wilder_rsi(g["close"].values), index=g.index))
raw["rsi"] = pd.concat(parts).sort_index()

labels = pd.read_parquet("data/regime_labels_practice.parquet")
df = raw.merge(labels, on=["ticker", "date"], how="inner")
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
print(f"Merged: {len(df):,} rows", flush=True)


def run_exit(rsi, opens, entry_i, n):
    """Shared exit logic: RSI > 55, else MAX_HOLD. Returns (ret, hold)."""
    if entry_i >= n - 1:
        return None, None
    epx = opens[entry_i]
    xi = None
    for j in range(entry_i, min(entry_i + MAX_HOLD, n - 1)):
        if np.isfinite(rsi[j]) and rsi[j] > RSI_EXIT:
            xi = j + 1
            break
    if xi is None:
        xi = min(entry_i + MAX_HOLD, n - 1)
    return opens[xi] / epx - 1.0 - COST, xi - entry_i


# ── real trades + per-ticker arrays retained for the fair monkey test ─────────
print("\nGenerating real trades...", flush=True)
trades = []
stock_arrays = {}      # ticker -> (rsi, opens, regs, elig, n)

for tk, g in df.groupby("ticker", sort=False):
    rsi = g["rsi"].values; opens = g["open"].values
    regs = g["regime"].values; elig = g["eligible"].values
    dates = g["date"].values; n = len(g)
    stock_arrays[tk] = (rsi, opens, regs, elig, n)
    i = RSI_PERIOD + 1
    while i < n - 1:
        if (np.isfinite(rsi[i-1]) and np.isfinite(rsi[i])
                and rsi[i-1] < RSI_ENTRY <= rsi[i] and elig[i]):
            ret, hold = run_exit(rsi, opens, i + 1, n)
            if ret is not None:
                trades.append((regs[i], hold, ret, dates[i]))
                i = i + 1 + hold + 1
                continue
        i += 1

tr = pd.DataFrame(trades, columns=["regime", "hold", "ret", "date"])
print(f"  {len(tr):,} real trades", flush=True)

# ── FAIR monkey test: random entry + SAME exit rule ──────────────────────────
print("\nBuilding fair monkey pool (random entry, identical exit rule)...",
      flush=True)
entry_pool = {}
for tk, (rsi, opens, regs, elig, n) in stock_arrays.items():
    for reg in ("calm", "trending", "choppy"):
        idx = np.where((regs == reg) & elig)[0]
        idx = idx[(idx > RSI_PERIOD + 1) & (idx < n - MAX_HOLD - 2)]
        if len(idx):
            entry_pool.setdefault(reg, []).append((tk, idx))


def fair_monkeys(regime, n_trades, n_sims=N_MONKEY):
    """Random entries in same regime, THEN the identical exit rule."""
    if regime not in entry_pool:
        return None
    cands = entry_pool[regime]
    sims = np.empty(n_sims)
    for s in range(n_sims):
        tot, cnt = 0.0, 0
        for _ in range(n_trades):
            tk, idx = cands[rng.integers(len(cands))]
            rsi, opens, regs, elig, n = stock_arrays[tk]
            st = int(idx[rng.integers(len(idx))])
            ret, _ = run_exit(rsi, opens, st + 1, n)
            if ret is not None:
                tot += ret; cnt += 1
        sims[s] = tot / cnt if cnt else 0.0
    return sims


print("\n" + "=" * 68)
print("RESULTS -- fair control")
print("=" * 68)

um = tr["ret"].mean()
ut = um / (tr["ret"].std(ddof=1) / np.sqrt(len(tr)))
print(f"\nUNCONDITIONAL: n={len(tr):,}  mean={um*100:+.3f}%  t={ut:.2f}")

res = {}
for reg in ("choppy", "calm", "trending"):
    sub = tr[tr["regime"] == reg]
    if len(sub) < 30:
        continue
    r = sub["ret"].values
    m = r.mean(); t = m / (r.std(ddof=1) / np.sqrt(len(r)))
    # sample size for monkeys capped for runtime
    n_mk = min(len(sub), 300)
    sims = fair_monkeys(reg, n_mk)
    beat = float((sims < m).mean() * 100) if sims is not None else np.nan
    mk_m = sims.mean() if sims is not None else np.nan
    print(f"\n{reg.upper()}: n={len(r):,}")
    print(f"  strategy mean = {m*100:+.3f}%   t = {t:.2f}")
    print(f"  FAIR monkey mean = {mk_m*100:+.3f}%  "
          f"(random entry, same exit rule)")
    print(f"  edge over fair monkey = {(m - mk_m)*100:+.3f}pp")
    print(f"  beats {beat:.0f}% of fair monkeys")
    res[reg] = {"m": m, "t": t, "beat": beat, "mk": mk_m}

# unconditional fair monkey
all_sims = []
for reg in ("choppy", "calm", "trending"):
    s = fair_monkeys(reg, 200)
    if s is not None:
        all_sims.append(s)
if all_sims:
    pooled_mk = np.mean([s.mean() for s in all_sims])
    print(f"\nUNCONDITIONAL fair monkey mean ~ {pooled_mk*100:+.3f}%")
    print(f"  strategy edge over fair monkey = {(um - pooled_mk)*100:+.3f}pp")

# ── survivorship diagnostic, crisis-decoupled ────────────────────────────────
print("\n" + "=" * 68)
print("SURVIVORSHIP DIAGNOSTIC (with and without the 2008-09 crisis)")
print("=" * 68)
tr["yr"] = pd.to_datetime(tr["date"]).dt.year
for lab, y0, y1 in [("2006-2009", 2006, 2009), ("2010-2013", 2010, 2013),
                    ("2014-2018", 2014, 2018)]:
    s = tr[(tr["yr"] >= y0) & (tr["yr"] <= y1)]
    if len(s) >= 30:
        print(f"  {lab}: n={len(s):,}  mean={s['ret'].mean()*100:+.3f}%")
ex = tr[~tr["yr"].isin([2008, 2009])]
print(f"\n  Excluding 2008-2009 entirely: n={len(ex):,}  "
      f"mean={ex['ret'].mean()*100:+.3f}%")
for lab, y0, y1 in [("2006-2007", 2006, 2007), ("2010-2013", 2010, 2013),
                    ("2014-2018", 2014, 2018)]:
    s = ex[(ex["yr"] >= y0) & (ex["yr"] <= y1)]
    if len(s) >= 30:
        print(f"  {lab}: n={len(s):,}  mean={s['ret'].mean()*100:+.3f}%")
print("  (monotonic rise toward the past = survivorship fingerprint)")

print("\n" + "=" * 68)
print("VERDICT")
print("=" * 68)
c = res.get("choppy")
if c:
    k = [c["m"] > um, c["t"] > 2.0, c["beat"] > 70, c["m"] > COST]
    for ok, nm in zip(k, ["expectancy > unconditional", "t > 2.0",
                          "beats >70% FAIR monkeys", "exceeds cost"]):
        print(f"  {'PASS' if ok else 'FAIL'}  {nm}")
    print(f"\n  OVERALL: {'PASS' if all(k) else 'FAIL'}")
print("=" * 68, flush=True)
