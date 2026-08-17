"""
Test 19 -- portfolio simulation + alpha regression.

Test 18 PASSED its criteria (monthly t=3.50, beat SPY +0.705pp) but three
problems remained:
  1. t on the DIFFERENCE vs SPY was only 1.81 -- my criterion 4 was too
     weak (compared point estimates instead of testing the difference)
  2. per-trade mean != portfolio return. 329 trades fired in one month;
     you cannot hold 329 equal-weighted positions on a real account
  3. 2009 (+8.03%) dominates and is the most survivorship-contaminated
     year in the sample (2008 dips that went bankrupt are absent)

This builds an ACTUALLY TRADEABLE portfolio: max N concurrent positions,
equal weight, cash when signals are scarce. Then regresses monthly
portfolio returns on SPY to separate alpha from beta -- because "buy
oversold stocks in a rising market" is very likely just beta, which is
what Tests 5, 7, 8 and 11 all turned out to be.

PRE-REGISTERED PASS CRITERIA (fixed before running):
  1. annualized alpha t-statistic > 2.0
  2. alpha positive after 0.05% round-trip costs
  3. result survives excluding 2009
  4. Sharpe > SPY's Sharpe over the same window
"""
import numpy as np
import pandas as pd

RSI_PERIOD, RSI_ENTRY, RSI_EXIT, MAX_HOLD = 14, 30, 55, 20
COST = 0.0005
PRICE_MIN, PRICE_MAX, MIN_VOL = 5.0, 100.0, 250_000
MAX_POSITIONS = 20          # realistic concurrent position limit
PRACTICE_END = "2018-12-31"

print("=" * 68)
print("TEST 19 -- PORTFOLIO SIMULATION + ALPHA REGRESSION")
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

print("Collecting signals...", flush=True)
sigs = []
for tk, g in raw.groupby("ticker", sort=False):
    rsi = g["rsi"].values; opens = g["open"].values
    elig = g["eligible"].values; dates = g["date"].values
    n = len(g)
    for i in range(RSI_PERIOD + 1, n - 1):
        if (np.isfinite(rsi[i-1]) and np.isfinite(rsi[i])
                and rsi[i-1] < RSI_ENTRY <= rsi[i] and elig[i]):
            ei = i + 1
            if ei >= n - 1:
                continue
            xi = None
            for j in range(ei, min(ei + MAX_HOLD, n - 1)):
                if np.isfinite(rsi[j]) and rsi[j] > RSI_EXIT:
                    xi = j + 1; break
            if xi is None:
                xi = min(ei + MAX_HOLD, n - 1)
            sigs.append({"ticker": tk, "entry": dates[ei], "exit": dates[xi],
                         "ret": opens[xi] / opens[ei] - 1.0 - COST})

sig = pd.DataFrame(sigs).sort_values("entry").reset_index(drop=True)
print(f"  {len(sig):,} raw signals", flush=True)

# ---- portfolio simulation: capacity-constrained, equal weight ----
print(f"Simulating portfolio (max {MAX_POSITIONS} concurrent)...", flush=True)
all_days = pd.DatetimeIndex(sorted(raw["date"].unique()))
sig_by_day = {d: grp for d, grp in sig.groupby("entry")}

open_pos, taken = [], []
for d in all_days:
    open_pos = [p for p in open_pos if p["exit"] > d]
    room = MAX_POSITIONS - len(open_pos)
    if room > 0 and d in sig_by_day:
        todays = sig_by_day[d]
        for _, r in todays.head(room).iterrows():
            open_pos.append({"exit": r["exit"], "ret": r["ret"]})
            taken.append({"entry": d, "exit": r["exit"], "ret": r["ret"]})

pf = pd.DataFrame(taken)
print(f"  {len(pf):,} trades taken ({len(pf)/len(sig)*100:.0f}% of signals "
      f"-- rest rejected for lack of capacity)", flush=True)

# monthly portfolio return: each trade uses 1/MAX_POSITIONS of capital
pf["month"] = pd.to_datetime(pf["exit"]).dt.to_period("M")
monthly_pf = pf.groupby("month")["ret"].sum() / MAX_POSITIONS

spy = pd.read_parquet("data/spy_yf.parquet")
spy["date"] = pd.to_datetime(spy["date"])
spy = spy[spy["date"] <= PRACTICE_END].sort_values("date")
spy["month"] = spy["date"].dt.to_period("M")
spy_m = spy.groupby("month")["close"].last().pct_change().dropna()

common = monthly_pf.index.intersection(spy_m.index)
r = monthly_pf.loc[common]; s = spy_m.loc[common]


def report(rr, ss, label):
    n = len(rr)
    beta = np.cov(rr, ss)[0, 1] / np.var(ss, ddof=1)
    alpha_m = rr.mean() - beta * ss.mean()
    resid = rr - (beta * ss + alpha_m)
    se = resid.std(ddof=2) / np.sqrt(n)
    t = alpha_m / se if se > 0 else np.nan
    sr = rr.mean() / rr.std(ddof=1) * np.sqrt(12)
    ss_ = ss.mean() / ss.std(ddof=1) * np.sqrt(12)
    print(f"\n{label}  (n={n} months)")
    print(f"  strategy mean monthly : {rr.mean()*100:+.3f}%")
    print(f"  SPY mean monthly      : {ss.mean()*100:+.3f}%")
    print(f"  beta                  : {beta:.3f}")
    print(f"  annualized alpha      : {alpha_m*12*100:+.2f}%")
    print(f"  alpha t-statistic     : {t:.2f}")
    print(f"  Sharpe strategy / SPY : {sr:.2f} / {ss_:.2f}")
    return {"t": t, "alpha": alpha_m, "sr": sr, "ss": ss_}


print("\n" + "=" * 68)
print("RESULTS")
print("=" * 68)
full   = report(r, s, "FULL PERIOD")
m1     = ~r.index.year.isin([2009])
ex09   = report(r[m1], s[m1], "EXCLUDING 2009")
m2     = ~r.index.year.isin([2008, 2009])
ex0809 = report(r[m2], s[m2], "EXCLUDING 2008-2009")

print("\n" + "=" * 68)
print("VERDICT vs PRE-REGISTERED CRITERIA")
print("=" * 68)
k1 = full["t"] > 2.0
k2 = full["alpha"] > 0
k3 = ex09["t"] > 2.0
k4 = full["sr"] > full["ss"]
print(f"  {'PASS' if k1 else 'FAIL'}  alpha t > 2.0           ({full['t']:.2f})")
print(f"  {'PASS' if k2 else 'FAIL'}  alpha positive net cost")
print(f"  {'PASS' if k3 else 'FAIL'}  survives excl. 2009     ({ex09['t']:.2f})")
print(f"  {'PASS' if k4 else 'FAIL'}  Sharpe > SPY            "
      f"({full['sr']:.2f} vs {full['ss']:.2f})")
print(f"\n  OVERALL: {'PASS' if all([k1, k2, k3, k4]) else 'FAIL'}")
print("=" * 68, flush=True)
