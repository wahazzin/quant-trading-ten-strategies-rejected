"""
regime_conditional_v2.py -- Test 17b (corrected)

Test 17 FAILED its criteria (choppy t=1.80 vs required 2.0), but diagnosis
of the t-statistics revealed the result was contaminated regardless:

  implied std dev by bucket:  calm 7.4%  trending 17.7%  CHOPPY 130%

A 130% std dev on <=20-day holds is not a market phenomenon. Two causes:

  (1) DESIGN ERROR: no price/liquidity filter was applied. Every other test
      in this project filters $5-$100 with a volume floor. Sub-$1 penny
      stocks produce +400% moves; unadjusted reverse splits produce more.
      Because "choppy" is DEFINED as high volatility, junk sorts itself
      into that bucket by construction.

  (2) SURVIVORSHIP BIAS: RSI mean-reversion is buy-the-dip. This universe
      is stocks still listed in 2026, so every dip was bought by a company
      that survived. Dips that went to zero are absent. This is fatal for
      dip-buying specifically -- the same reason long-term reversal was
      never tested on this data.

THIS VERSION:
  - applies the standard $5-$100 price + 250k volume filter at ENTRY
  - reports MEDIAN alongside mean (robust to outliers)
  - winsorizes at 1st/99th pct as a robustness check
  - adds a SURVIVORSHIP DIAGNOSTIC: splits the practice window into three
    sub-periods. Survivorship bias grows with time distance (a 2006 stock
    has had 20 years to die; a 2018 stock only 8). If dip-buying looks
    progressively BETTER further back in time, that is the fingerprint.

PRE-REGISTERED (unchanged from Test 17, plus the diagnostic):
  1. choppy expectancy > unconditional
  2. choppy t > 2.0
  3. beats >70% regime-matched monkeys
  4. exceeds 0.05% cost
  NEW 5. survivorship diagnostic must NOT show monotonic decay toward
        the present -- if it does, the effect is an artifact.
"""

import numpy as np
import pandas as pd
from pathlib import Path

TRAIL, RSI_PERIOD = 60, 14
RSI_ENTRY, RSI_EXIT, MAX_HOLD = 30, 55, 20
COST = 0.0005
PRICE_MIN, PRICE_MAX, MIN_VOL = 5.0, 100.0, 250_000
N_MONKEY = 1000
PRACTICE_END = "2018-12-31"
rng = np.random.default_rng(42)

print("=" * 68)
print("TEST 17b -- REGIME-CONDITIONAL RSI, FILTERED + SURVIVORSHIP DIAGNOSTIC")
print("=" * 68, flush=True)

print("\nLoading...", flush=True)
raw = pd.read_parquet("data/yf_universe.parquet",
                      columns=["ticker", "date", "open", "close", "volume"])
raw["date"] = pd.to_datetime(raw["date"])
raw = raw[raw["date"] <= PRACTICE_END]
raw = raw[(raw["close"] > 0) & (raw["open"] > 0)]
raw = raw.sort_values(["ticker", "date"]).reset_index(drop=True)

# 20-day average volume for the liquidity filter
raw["adv20"] = raw.groupby("ticker")["volume"].transform(
    lambda s: s.rolling(20, min_periods=20).mean())

# eligibility flag evaluated AT ENTRY (trailing data only)
raw["eligible"] = ((raw["close"] >= PRICE_MIN) &
                   (raw["close"] <= PRICE_MAX) &
                   (raw["adv20"] >= MIN_VOL))
print(f"  {len(raw):,} rows | eligible stock-days: "
      f"{raw['eligible'].sum():,} ({raw['eligible'].mean()*100:.1f}%)",
      flush=True)


def wilder_rsi(close, period=RSI_PERIOD):
    n = len(close)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    d = np.diff(close)
    gain = np.where(d > 0, d, 0.0)
    loss = np.where(d < 0, -d, 0.0)
    ag, al = gain[:period].mean(), loss[:period].mean()
    for i in range(period, n - 1):
        ag = (ag * (period - 1) + gain[i]) / period
        al = (al * (period - 1) + loss[i]) / period
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

print("Loading cached regime labels...", flush=True)
labels = pd.read_parquet("data/regime_labels_practice.parquet")
df = raw.merge(labels, on=["ticker", "date"], how="inner")
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
print(f"  merged: {len(df):,} rows", flush=True)


# ── trade generation (entry must pass eligibility filter) ─────────────────────
print("\nGenerating trades (filtered)...", flush=True)
trades = []
for tk, g in df.groupby("ticker", sort=False):
    rsi   = g["rsi"].values
    opens = g["open"].values
    regs  = g["regime"].values
    elig  = g["eligible"].values
    dates = g["date"].values
    n = len(g)
    i = RSI_PERIOD + 1
    while i < n - 1:
        if (np.isfinite(rsi[i-1]) and np.isfinite(rsi[i])
                and rsi[i-1] < RSI_ENTRY <= rsi[i] and elig[i]):
            ei = i + 1
            epx = opens[ei]
            reg = regs[i]
            xi = None
            for j in range(ei, min(ei + MAX_HOLD, n - 1)):
                if np.isfinite(rsi[j]) and rsi[j] > RSI_EXIT:
                    xi = j + 1
                    break
            if xi is None:
                xi = min(ei + MAX_HOLD, n - 1)
            trades.append((reg, xi - ei, opens[xi]/epx - 1.0 - COST,
                           dates[i]))
            i = xi + 1
        else:
            i += 1

tr = pd.DataFrame(trades, columns=["regime", "hold", "ret", "date"])
print(f"  {len(tr):,} trades (was 22,840 unfiltered)", flush=True)

# ── regime-matched monkey pool (eligible days only) ──────────────────────────
print("Building monkey pool...", flush=True)
pool = {}
for tk, g in df.groupby("ticker", sort=False):
    opens = g["open"].values
    regs  = g["regime"].values
    elig  = g["eligible"].values
    for reg in ("calm", "trending", "choppy"):
        idx = np.where((regs == reg) & elig)[0]
        idx = idx[(idx > RSI_PERIOD) & (idx < len(opens) - MAX_HOLD - 2)]
        if len(idx):
            pool.setdefault(reg, []).append((opens, idx))


def monkeys(regime, holds, n_sims=N_MONKEY):
    if regime not in pool or not len(holds):
        return None
    ents = pool[regime]
    sims = np.empty(n_sims)
    for s in range(n_sims):
        tot = 0.0
        for h in holds:
            opens, idx = ents[rng.integers(len(ents))]
            st = int(idx[rng.integers(len(idx))])
            en = min(st + h, len(opens) - 1)
            tot += opens[en]/opens[st] - 1.0 - COST
        sims[s] = tot / len(holds)
    return sims


def winsorize(x, lo=0.01, hi=0.99):
    a, b = np.quantile(x, [lo, hi])
    return np.clip(x, a, b)


print("\n" + "=" * 68)
print("RESULTS (price $5-$100, ADV >= 250k, applied at entry)")
print("=" * 68)

um = tr["ret"].mean()
ut = um / (tr["ret"].std(ddof=1)/np.sqrt(len(tr)))
print(f"\nUNCONDITIONAL: n={len(tr):,}  mean={um*100:+.3f}%  "
      f"median={tr['ret'].median()*100:+.3f}%  t={ut:.2f}")

res = {}
for reg in ("choppy", "calm", "trending"):
    sub = tr[tr["regime"] == reg]
    if len(sub) < 30:
        print(f"\n{reg.upper()}: n={len(sub)} -- underpowered")
        continue
    r = sub["ret"].values
    m, md = r.mean(), np.median(r)
    sd = r.std(ddof=1)
    t = m / (sd/np.sqrt(len(r)))
    w = winsorize(r)
    wt = w.mean() / (w.std(ddof=1)/np.sqrt(len(w)))
    sims = monkeys(reg, sub["hold"].values)
    beat = float((sims < m).mean()*100) if sims is not None else np.nan
    print(f"\n{reg.upper()}: n={len(r):,}")
    print(f"  mean={m*100:+.3f}%   median={md*100:+.3f}%   sd={sd*100:.1f}%")
    print(f"  t={t:.2f}   winsorized mean={w.mean()*100:+.3f}% (t={wt:.2f})")
    print(f"  beats {beat:.0f}% of regime-matched monkeys "
          f"(monkey mean {sims.mean()*100:+.3f}%)")
    res[reg] = {"m": m, "t": t, "beat": beat, "wt": wt}

# ── SURVIVORSHIP DIAGNOSTIC ──────────────────────────────────────────────────
print("\n" + "=" * 68)
print("SURVIVORSHIP DIAGNOSTIC")
print("Bias grows with time distance: a 2006 stock had 20 years to die,")
print("a 2018 stock only 8. Monotonic decay toward the present = artifact.")
print("=" * 68)
tr["yr"] = pd.to_datetime(tr["date"]).dt.year
buckets = [("2006-2009", 2006, 2009), ("2010-2013", 2010, 2013),
           ("2014-2018", 2014, 2018)]
for lab, y0, y1 in buckets:
    s = tr[(tr["yr"] >= y0) & (tr["yr"] <= y1)]
    if len(s) < 30:
        print(f"  {lab}: n={len(s)} underpowered"); continue
    c = s[s["regime"] == "choppy"]
    print(f"  {lab}: all n={len(s):,} mean={s['ret'].mean()*100:+.3f}%  |  "
          f"choppy n={len(c):,} mean="
          f"{c['ret'].mean()*100:+.3f}%" if len(c) else "")

print("\n" + "=" * 68)
print("VERDICT")
print("=" * 68)
c = res.get("choppy")
if c:
    k = [c["m"] > um, c["t"] > 2.0, c["beat"] > 70, c["m"] > COST]
    names = ["expectancy > unconditional", "t > 2.0",
             "beats >70% monkeys", "exceeds cost"]
    for ok, nm in zip(k, names):
        print(f"  {'PASS' if ok else 'FAIL'}  {nm}")
    print(f"\n  OVERALL: {'PASS' if all(k) else 'FAIL'}")
    print("  (criterion 5 -- survivorship decay -- read the diagnostic above)")
print("=" * 68, flush=True)
