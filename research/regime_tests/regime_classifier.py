"""
regime_classifier.py v2 -- vectorized for speed.
Same logic, same pre-registered thresholds, fully pandas/numpy.
Should complete in 3-5 minutes instead of timing out.
"""

import numpy as np
import pandas as pd
from pathlib import Path

TRAIL        = 60
FWD          = 20
PCTL_LO      = 33
PCTL_HI      = 67
PRACTICE_END = "2018-12-31"
DATA_PATH    = Path("data/yf_universe.parquet")

print("=" * 65)
print("REGIME CLASSIFIER v2 (vectorized)")
print("Practice window: start -> 2018-12-31  |  post-2019 sealed")
print("=" * 65, flush=True)

print("\nLoading...", flush=True)
raw = pd.read_parquet(DATA_PATH, columns=["ticker", "date", "close"])
raw["date"] = pd.to_datetime(raw["date"])
raw = raw[raw["date"] <= PRACTICE_END].copy()
raw = raw.sort_values(["ticker", "date"]).reset_index(drop=True)
print(f"  {len(raw):,} rows | {raw['ticker'].nunique()} tickers", flush=True)

# log returns, grouped
print("\nComputing rolling features...", flush=True)
raw["log_ret"] = raw.groupby("ticker")["close"].transform(
    lambda x: np.log(x).diff()
)

# trailing vol: std of log_ret over TRAIL window, annualised
raw["trail_vol"] = (
    raw.groupby("ticker")["log_ret"]
    .transform(lambda x: x.rolling(TRAIL, min_periods=TRAIL).std())
    * np.sqrt(252)
)

# trend_r2: rolling R2 of log_price on linear index
# computed per-ticker via apply (fast enough at this level)
def rolling_r2(prices, window=TRAIL):
    """prices: 1-D numpy array of close prices."""
    x = np.arange(window, dtype=float)
    x -= x.mean()
    ss_xx = (x ** 2).sum()
    result = np.full(len(prices), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.log(prices)
    for i in range(window, len(log_p)):
        y = log_p[i - window: i]
        if not np.isfinite(y).all():
            continue
        y = y - y.mean()
        b = (x * y).sum() / ss_xx
        resid = y - b * x
        ss_tot = (y ** 2).sum()
        if ss_tot > 0:
            result[i] = 1.0 - (resid ** 2).sum() / ss_tot
        else:
            result[i] = 0.0
    return result

print("  Rolling R2 (this takes 2-3 min)...", flush=True)
r2_parts = []
tickers = raw["ticker"].unique()
for i, tk in enumerate(tickers):
    if i % 300 == 0:
        print(f"    {i}/{len(tickers)}", flush=True)
    sub = raw[raw["ticker"] == tk]["close"].values
    r2 = rolling_r2(sub)
    r2_parts.append(pd.Series(r2, index=raw[raw["ticker"] == tk].index))

raw["trend_r2"] = pd.concat(r2_parts).sort_index()
feat = raw.dropna(subset=["trail_vol", "trend_r2"]).copy()
print(f"  Valid stock-days: {len(feat):,}", flush=True)

# ── assign trailing regimes (cross-sectional percentiles per date) ─────────────
print("\nAssigning trailing regimes...", flush=True)

def label_regime(vol, r2, vlo, vhi, tlo, thi):
    if r2 >= thi:
        return "trending"
    elif vol <= vlo:
        return "calm"
    else:
        return "choppy"

regime_list = []
for d, grp in feat.groupby("date"):
    if len(grp) < 20:
        continue
    vlo = grp["trail_vol"].quantile(PCTL_LO / 100)
    vhi = grp["trail_vol"].quantile(PCTL_HI / 100)
    tlo = grp["trend_r2"].quantile(PCTL_LO / 100)
    thi = grp["trend_r2"].quantile(PCTL_HI / 100)
    labels = [label_regime(r["trail_vol"], r["trend_r2"], vlo, vhi, tlo, thi)
              for _, r in grp.iterrows()]
    tmp = grp[["ticker", "date"]].copy()
    tmp["regime_trailing"] = labels
    regime_list.append(tmp)

feat_labeled = pd.concat(regime_list, ignore_index=True)
print(feat_labeled["regime_trailing"].value_counts(normalize=True).round(3),
      flush=True)

# ── forward regime ─────────────────────────────────────────────────────────────
print("\nComputing forward regimes...", flush=True)

# join each stock-day with its forward window
raw_indexed = raw.set_index(["ticker", "date"])[["close"]]
fwd_rows = []

for i, (tk, grp) in enumerate(feat.groupby("ticker")):
    if i % 300 == 0:
        print(f"  {i}/{len(tickers)}", flush=True)
    prices = np.log(grp["close"].values)
    dates  = grp["date"].values
    n = len(prices)
    for j in range(n):
        if j + FWD >= n:
            break
        fwd_p = prices[j + 1: j + FWD + 1]
        rets  = np.diff(fwd_p)
        if len(rets) < 5:
            continue
        fwd_vol = rets.std() * np.sqrt(252)
        x  = np.arange(len(fwd_p), dtype=float); x -= x.mean()
        y  = fwd_p - fwd_p.mean()
        ss_xx = (x ** 2).sum()
        if ss_xx == 0:
            continue
        b = (x * y).sum() / ss_xx
        resid = y - b * x
        ss_tot = (y ** 2).sum()
        fwd_r2 = 1.0 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else 0.0
        fwd_rows.append({"ticker": tk, "date": dates[j],
                         "fwd_vol": fwd_vol, "fwd_r2": fwd_r2})

fwd = pd.DataFrame(fwd_rows)
print(f"  Forward pairs: {len(fwd):,}", flush=True)

fwd_labeled_rows = []
for d, grp in fwd.groupby("date"):
    if len(grp) < 20:
        continue
    vlo = grp["fwd_vol"].quantile(PCTL_LO / 100)
    vhi = grp["fwd_vol"].quantile(PCTL_HI / 100)
    tlo = grp["fwd_r2"].quantile(PCTL_LO / 100)
    thi = grp["fwd_r2"].quantile(PCTL_HI / 100)
    labels = [label_regime(r["fwd_vol"], r["fwd_r2"], vlo, vhi, tlo, thi)
              for _, r in grp.iterrows()]
    tmp = grp[["ticker", "date"]].copy()
    tmp["regime_forward"] = labels
    fwd_labeled_rows.append(tmp)

fwd_labeled = pd.DataFrame(pd.concat(fwd_labeled_rows, ignore_index=True))

# ── evaluation ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("RESULTS")
print("=" * 65, flush=True)

merged = feat_labeled.merge(fwd_labeled, on=["ticker", "date"])
print(f"Matched pairs: {len(merged):,}\n")

correct          = (merged["regime_trailing"] == merged["regime_forward"]).mean()
most_common_frac = merged["regime_forward"].value_counts(normalize=True).iloc[0]

print(f"Overall predictive accuracy:   {correct:.3f}  ({correct*100:.1f}%)")
print(f"Chance baseline (most common): {most_common_frac:.3f}  "
      f"({most_common_frac*100:.1f}%)")
print(f"Lift over chance:              {correct - most_common_frac:+.3f} "
      f"({(correct - most_common_frac)*100:+.1f}pp)\n")

labels = ["calm", "trending", "choppy"]
print("Confusion matrix (rows=trailing, cols=actual forward):")
cm = pd.crosstab(merged["regime_trailing"], merged["regime_forward"],
                 normalize="index").reindex(index=labels, columns=labels,
                                            fill_value=0.0)
print(cm.round(3).to_string())

print("\nPer-regime precision / recall:")
for lab in labels:
    tp = ((merged["regime_trailing"] == lab) &
          (merged["regime_forward"]  == lab)).sum()
    fp = ((merged["regime_trailing"] == lab) &
          (merged["regime_forward"]  != lab)).sum()
    fn = ((merged["regime_trailing"] != lab) &
          (merged["regime_forward"]  == lab)).sum()
    prec   = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    print(f"  {lab:10s}  prec={prec:.3f}  recall={recall:.3f}  n={tp+fp:,}")

print("\nPersistence (prob same regime after N days):")
feat_s = feat_labeled.sort_values(["ticker", "date"])
for hor in [5, 10, 20]:
    same = total = 0
    for tk, grp in feat_s.groupby("ticker"):
        reg = grp["regime_trailing"].values
        for i in range(len(reg) - hor):
            same  += reg[i] == reg[i + hor]
            total += 1
    print(f"  {hor:2d}-day: {same/total:.3f}  (random baseline ~0.333)")

print("\n" + "=" * 65)
print("PASS criteria: accuracy > chance by +5pp AND 20-day persistence > 0.45")
print("=" * 65, flush=True)
