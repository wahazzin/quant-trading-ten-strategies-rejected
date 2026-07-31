"""
event_study_v2.py -- fixes a confound in event_study.py (v1): subtracting
SPY's return measured EVENT impact plus our universe's own baseline
drift relative to SPY. Our universe underperformed SPY by ~5%/year over
2019-2026, which alone predicts roughly -0.02%/-0.10%/-0.20%/-0.40% at
1/5/10/20 trading days -- almost exactly what v1 measured as the
"event effect." The sanity check below confirms this explicitly before
any event analysis is trusted.

New benchmark: a matched non-event control, per stock. For each stock,
every trading day NOT within 20 trading days after any of that stock's
own 8-K filings is a "clean" day. That stock's mean forward return
starting from its clean days, at the same horizon, is its own baseline:

    adjusted_AR = (post-event forward return) - (that stock's mean
                   clean-window forward return, same horizon)

This differences out each stock's own idiosyncratic drift without
needing a market benchmark at all -- SPY is used ONLY in the sanity
check section below, never in the adjusted_AR event results.

Data loading, the evidence boundary (2019-01-01 onward; everything
before stays sealed, same as v1), and the event-matching logic (a
filing maps to the first trading day on/after it, entered at the next
day's open, forward returns open-to-open) are reused verbatim from
event_study.py. Only the benchmark changes.

No strategy is built, nothing is tuned, and no item code is selected
as "best" -- this reports what the pooled data shows, including
explicitly flagging when a statistically significant effect is too
small to survive a 0.2% round-trip cost.
"""
import os
import numpy as np
import pandas as pd

EDGAR_PATH = os.path.join("data", "edgar_8k.parquet")
PRICE_PATH = os.path.join("data", "yf_universe.parquet")
SPY_PATH = os.path.join("data", "spy_yf.parquet")
EVIDENCE_START = pd.Timestamp("2019-01-01")

HORIZONS = [1, 5, 10, 20]
TARGET_ITEMS = ["1.03", "2.02", "4.02", "5.02", "8.01"]
SIZE_LOOKBACK_DAYS = 60
MIN_SIZE_LOOKBACK = 20
CLEAN_WINDOW_DAYS = 20     # trading days after a filing considered "event-affected", not clean
MIN_CLEAN_OBS = 20         # minimum clean-day observations per stock/horizon to trust its baseline
ROUNDTRIP_COST = 0.002
V1_IMPLIED_DRIFT = {1: -0.0002, 5: -0.0010, 10: -0.0020, 20: -0.0040}  # v1's approximate confound


# ============================================================
# LOAD + EVIDENCE BOUNDARY (verbatim from event_study.py)
# ============================================================
raw_price = pd.read_parquet(PRICE_PATH)
raw_price["date"] = pd.to_datetime(raw_price["date"])
raw_spy = pd.read_parquet(SPY_PATH)
raw_spy["date"] = pd.to_datetime(raw_spy["date"])
raw_edgar = pd.read_parquet(EDGAR_PATH)
raw_edgar["filing_date"] = pd.to_datetime(raw_edgar["filing_date"])

print("=" * 96)
print("EVIDENCE BOUNDARY CONFIRMATION")
print("=" * 96)
print(f"Raw price panel on disk: {raw_price['date'].min().date()} to {raw_price['date'].max().date()} "
      f"({len(raw_price)} rows)")
print(f"Raw SPY panel on disk: {raw_spy['date'].min().date()} to {raw_spy['date'].max().date()} "
      f"({len(raw_spy)} rows)")
print(f"Raw 8-K filings on disk: {raw_edgar['filing_date'].min().date()} to {raw_edgar['filing_date'].max().date()} "
      f"({len(raw_edgar)} filings)")

price = raw_price[raw_price["date"] >= EVIDENCE_START].sort_values(["ticker", "date"]).reset_index(drop=True)
spy = raw_spy[raw_spy["date"] >= EVIDENCE_START].sort_values("date").reset_index(drop=True)
edgar = raw_edgar[raw_edgar["filing_date"] >= EVIDENCE_START].reset_index(drop=True)
del raw_price, raw_spy, raw_edgar

print(f"\nPrice rows kept (date >= {EVIDENCE_START.date()}): {len(price)}")
print(f"SPY rows kept: {len(spy)}")
print(f"8-K filings kept: {len(edgar)}")
print("CONFIRMED: no row dated before 2019-01-01 is used anywhere below this point.")
print("Same fresh holdout boundary as event_study.py v1 -- pre-2019 stays sealed.")


# ============================================================
# BUILD PER-TICKER PRICE LOOKUPS + SPY OPEN LOOKUP (verbatim from v1)
# ============================================================
spy_open_lookup = pd.Series(spy["open"].to_numpy(dtype=float), index=spy["date"])
ticker_groups = {t: g.reset_index(drop=True) for t, g in price.groupby("ticker")}

print(f"\nTickers with price data: {len(ticker_groups)}")
print(f"Unique tickers with 8-K filings: {edgar['ticker'].nunique()}")


# ============================================================
# PER TICKER: locate events (verbatim matching logic from v1), mark
# contaminated days, compute each stock's clean-day baseline per
# horizon, and pool clean-day SPY-benchmarked ARs for the sanity check.
# ============================================================
event_rows = []
clean_pool_frames = []
stock_clean_means = {}
skipped_no_price = set()

for ticker, g in edgar.groupby("ticker"):
    pdf = ticker_groups.get(ticker)
    if pdf is None:
        skipped_no_price.add(ticker)
        continue

    dates = pdf["date"].to_numpy()
    opens = pdf["open"].to_numpy(dtype=float)
    closes = pdf["close"].to_numpy(dtype=float)
    volumes = pdf["volume"].to_numpy(dtype=float)
    n = len(dates)

    filing_dates = g["filing_date"].to_numpy()
    day_idx = np.searchsorted(dates, filing_dates, side="left")   # same matching logic as v1

    contaminated = np.zeros(n, dtype=bool)
    for idx in day_idx:
        if idx >= n:
            continue
        lo = idx + 1
        hi = min(n, idx + 1 + CLEAN_WINDOW_DAYS)
        if lo < n:
            contaminated[lo:hi] = True
    clean_mask = ~contaminated
    clean_positions = np.where(clean_mask)[0]

    # forward return starting from EVERY position D, all horizons (vectorized)
    fwd = {}
    for h in HORIZONS:
        f = np.full(n, np.nan)
        last_D = n - 2 - h   # need D+1+h <= n-1
        if last_D >= 0:
            Ds = np.arange(0, last_D + 1)
            f[Ds] = opens[Ds + 1 + h] / opens[Ds + 1] - 1
        fwd[h] = f

    stock_clean_means[ticker] = {}
    for h in HORIZONS:
        vals = fwd[h][clean_positions]
        vals = vals[~np.isnan(vals)]
        stock_clean_means[ticker][h] = float(vals.mean()) if len(vals) >= MIN_CLEAN_OBS else np.nan

        # pool clean-day SPY-benchmarked AR for the sanity check (vectorized per ticker/horizon)
        valid_D = clean_positions[clean_positions + 1 + h < n]
        valid_D = valid_D[~np.isnan(fwd[h][valid_D])]
        if len(valid_D) == 0:
            continue
        entry_dates_h = dates[valid_D + 1]
        exit_dates_h = dates[valid_D + 1 + h]
        spy_entry_vals = spy_open_lookup.reindex(entry_dates_h).to_numpy()
        spy_exit_vals = spy_open_lookup.reindex(exit_dates_h).to_numpy()
        spy_ret_vals = spy_exit_vals / spy_entry_vals - 1
        ar_vals = fwd[h][valid_D] - spy_ret_vals
        ar_vals = ar_vals[~np.isnan(ar_vals)]
        if len(ar_vals):
            clean_pool_frames.append(pd.DataFrame({"horizon": h, "ar_vs_spy": ar_vals}))

    for idx, items, accession in zip(day_idx, g["items"].to_numpy(), g["accession"].to_numpy()):
        entry_idx = idx + 1
        if idx >= n or entry_idx >= n:
            continue

        lo = max(0, idx - SIZE_LOOKBACK_DAYS)
        window_days = idx - lo
        size_proxy = (float(np.median(closes[lo:idx] * volumes[lo:idx]))
                      if window_days >= MIN_SIZE_LOOKBACK else np.nan)

        row = {"ticker": ticker, "items": items if isinstance(items, str) else "",
               "accession": accession, "size_proxy": size_proxy}
        for h in HORIZONS:
            event_ret = fwd[h][idx] if idx < n else np.nan
            baseline = stock_clean_means[ticker].get(h, np.nan)
            row[f"adj_ar_{h}"] = (event_ret - baseline) if (not np.isnan(event_ret) and not np.isnan(baseline)) else np.nan
        event_rows.append(row)

events_df = pd.DataFrame(event_rows)
clean_pool_df = pd.concat(clean_pool_frames, ignore_index=True) if clean_pool_frames else pd.DataFrame(columns=["horizon", "ar_vs_spy"])

print(f"\nEvents with usable forward-return windows: {len(events_df)}")
print(f"Tickers skipped (8-K filer but no price data in our universe): {len(skipped_no_price)}")
print(f"Pooled clean-day observations (all horizons combined): {len(clean_pool_df)}")


# ============================================================
# SANITY CHECK -- clean-day drift vs SPY, confirming (or not) the v1 confound
# ============================================================
print()
print("=" * 96)
print("SANITY CHECK: clean (non-event) day forward returns vs SPY -- does this alone")
print("reproduce v1's reported \"event effect\"?")
print("=" * 96)
header_sc = f"{'Horizon':<10}{'n':>10}{'MeanAR vs SPY':>15}{'t-stat':>9}{'v1 implied drift':>18}"
print(header_sc)
print("-" * len(header_sc))
for h in HORIZONS:
    vals = clean_pool_df.loc[clean_pool_df["horizon"] == h, "ar_vs_spy"].to_numpy()
    n = len(vals)
    mean = float(vals.mean()) if n else float("nan")
    std = float(vals.std(ddof=1)) if n > 1 else float("nan")
    t = mean / (std / np.sqrt(n)) if n > 1 and std > 0 else float("nan")
    print(f"{str(h)+'d':<10}{n:>10}{mean*100:>14.3f}%{t:>9.2f}{V1_IMPLIED_DRIFT[h]*100:>17.3f}%")

print()
print("If the \"MeanAR vs SPY\" column closely tracks the \"v1 implied drift\" column above,")
print("that confirms v1's SPY-relative \"event effect\" was mostly/entirely this universe's")
print("own baseline underperformance vs SPY, not an event-specific reaction.")


# ============================================================
# REPORTING (adjusted_AR only -- no SPY involved from here on)
# ============================================================
def compute_row(df, horizon):
    vals = df[f"adj_ar_{horizon}"].dropna().to_numpy()
    n = len(vals)
    if n < 2:
        return n, float("nan"), float("nan")
    mean = float(vals.mean())
    std = float(vals.std(ddof=1))
    t = mean / (std / np.sqrt(n)) if std > 0 else float("nan")
    return n, mean, t


def print_group(label, df):
    print(f"\n{label} -- {len(df)} events")
    header = f"{'Horizon':<10}{'n':>8}{'MeanAdjAR':>11}{'t-stat':>9}   {'Verdict'}"
    print(header)
    print("-" * len(header))
    for h in HORIZONS:
        n, mean, t = compute_row(df, h)
        if n < 2 or np.isnan(mean):
            print(f"{str(h)+'d':<10}{n:>8}{'n/a':>11}{'n/a':>9}   n/a")
            continue
        significant = (not np.isnan(t)) and abs(t) > 2
        if not significant:
            verdict = "not significant"
        elif abs(mean) > ROUNDTRIP_COST:
            verdict = f"significant AND exceeds {ROUNDTRIP_COST*100:.1f}% cost -- potentially tradeable"
        else:
            verdict = f"significant but BELOW {ROUNDTRIP_COST*100:.1f}% cost -- not tradeable"
        print(f"{str(h)+'d':<10}{n:>8}{mean*100:>10.3f}%{t:>9.2f}   {verdict}")


print()
print("=" * 96)
print("OVERALL: adjusted abnormal returns (vs each stock's own clean-day baseline)")
print("=" * 96)
print_group("ALL EVENTS", events_df)

print()
print("=" * 96)
print("BY ITEM CODE (an event can appear in more than one bucket if it reports multiple items)")
print("=" * 96)
for code in TARGET_ITEMS:
    mask = events_df["items"].str.contains(code, regex=False, na=False)
    print_group(f"Item {code}", events_df[mask])

other_mask = ~events_df["items"].apply(lambda s: any(c in s for c in TARGET_ITEMS))
print_group("Other (none of the target items above)", events_df[other_mask])

print()
print("=" * 96)
print("BY COMPANY SIZE PROXY (median $ volume, prior 60 trading days, median-split)")
print("=" * 96)
valid_size = events_df.dropna(subset=["size_proxy"])
n_no_size = len(events_df) - len(valid_size)
median_proxy = valid_size["size_proxy"].median()
print(f"Events with a valid size proxy: {len(valid_size)} ({n_no_size} excluded -- insufficient trailing history)")
print(f"Median proxy value (60-day median $ volume): ${median_proxy:,.0f}")

low = valid_size[valid_size["size_proxy"] < median_proxy]
high = valid_size[valid_size["size_proxy"] >= median_proxy]
print_group("Low dollar-volume half (under-covered names)", low)
print_group("High dollar-volume half (well-covered names)", high)

print()
print("=" * 96)
print("This is an event-study diagnostic only: no trading strategy was built, no parameter")
print("was tuned, and no item code was selected as \"best.\" Everything before 2019-01-01")
print("remains sealed for any strategy eventually built from this result.")
