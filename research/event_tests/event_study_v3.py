"""
event_study_v3.py -- re-examines Test 9 (SEC 8-K event study) under the
corrected cost basis discovered migrating the forward test to Alpaca
(RESEARCH_LOG.md Phase 6 / ROADMAP.md constraint C1). Test 9 was
rejected partly on economic grounds: the best surviving effect (~0.41%,
item 5.02, 10-day horizon) was smaller than the ~0.63% round-trip cost
assumed for the account at the time. Real cost on Alpaca is ~0.009% --
about 70x smaller. This retest exists ONLY because the cost input
changed, not because of any dissatisfaction with the original verdict.

Two corrections are applied that Test 14 (the sentiment shock study)
found necessary but were never applied here:

  1. DECLUSTERING. 8-Ks bunch in earnings season -- a company doesn't
     file one isolated 8-K, it often files several within days (an
     8-K for the earnings release itself, another for an accompanying
     press release or officer change, etc.), and v2's overlapping
     20-day-forward windows counted each as an independent observation.
     Test 14 found this exact pattern cut a t-stat from -4.02 to -1.60
     once corrected. Same fix here, same window (10 trading days,
     keep only the first 8-K per ticker in any such window, greedy,
     matching sentiment_shock_robustness.py's Check 1 exactly) --
     applied across ALL of a ticker's 8-Ks before any item-code filter,
     since a cluster can mix item codes for the same underlying event.

  2. LIQUIDITY SPLIT. Test 14's one sentiment-direction effect vanished
     entirely in the liquid half (t=-0.34) and lived only in illiquid
     names -- not tradeable at real size. Same median-split-by-trailing-
     60-day-dollar-volume check here, reusing v2's own size_proxy
     construction verbatim.

Everything else -- data loading, the evidence boundary (2019-01-01
onward, pre-2019 sealed), the matched non-event control (adjusted_AR =
event forward return minus that stock's own clean-day baseline, no SPY
involved), and the event-matching logic -- is reused verbatim from
event_study_v2.py. Only the two corrections above and the cost
comparison are new.

No parameter is tuned (the 10-day decluster window matches Test 14's,
not chosen here), no item code beyond 5.02 and the "all items pooled"
view is examined, and no strategy is built. This is a correctness
re-run under a corrected cost input, not a new search.
"""
import os
import numpy as np
import pandas as pd

EDGAR_PATH = os.path.join("data", "edgar_8k.parquet")
PRICE_PATH = os.path.join("data", "yf_universe.parquet")
SPY_PATH = os.path.join("data", "spy_yf.parquet")
EVIDENCE_START = pd.Timestamp("2019-01-01")

HORIZONS = [1, 5, 10, 20]
FOCUS_ITEM = "5.02"  # the only item code that cleared both bars in v2
SIZE_LOOKBACK_DAYS = 60
MIN_SIZE_LOOKBACK = 20
CLEAN_WINDOW_DAYS = 20
MIN_CLEAN_OBS = 20
DECLUSTER_WINDOW = 10          # trading days -- same as Test 14's Check 1, not tuned here
OLD_ROUNDTRIP_COST = 0.0063    # the ~0.63% assumption Test 9 was rejected against
REAL_ROUNDTRIP_COST = 0.00009  # Alpaca's real ~0.009%, confirmed in Phase 6 (RESEARCH_LOG.md)

pd.set_option("display.width", 140)


def t_stat(vals):
    n = len(vals)
    if n < 2:
        return n, float("nan"), float("nan")
    mean = float(vals.mean())
    std = float(vals.std(ddof=1))
    t = mean / (std / np.sqrt(n)) if std > 0 else float("nan")
    return n, mean, t


# ============================================================
# LOAD + EVIDENCE BOUNDARY (verbatim from event_study_v2.py)
# ============================================================
raw_price = pd.read_parquet(PRICE_PATH)
raw_price["date"] = pd.to_datetime(raw_price["date"])
raw_spy = pd.read_parquet(SPY_PATH)
raw_spy["date"] = pd.to_datetime(raw_spy["date"])
raw_edgar = pd.read_parquet(EDGAR_PATH)
raw_edgar["filing_date"] = pd.to_datetime(raw_edgar["filing_date"])

print("=" * 100)
print("EVIDENCE BOUNDARY CONFIRMATION")
print("=" * 100)
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
print("Same evidence boundary as event_study_v2.py -- pre-2019 stays sealed.")
print("\nThis re-examination was triggered SOLELY by the cost-model correction discovered")
print(f"migrating to Alpaca ({OLD_ROUNDTRIP_COST*100:.2f}% assumed -> {REAL_ROUNDTRIP_COST*100:.3f}% real),")
print("not by any dissatisfaction with v2's original verdict.")


# ============================================================
# BUILD PER-TICKER PRICE LOOKUPS (verbatim from v2)
# ============================================================
ticker_groups = {t: g.reset_index(drop=True) for t, g in price.groupby("ticker")}
print(f"\nTickers with price data: {len(ticker_groups)}")
print(f"Unique tickers with 8-K filings: {edgar['ticker'].nunique()}")


# ============================================================
# PER TICKER: locate events, mark contaminated days, compute clean-day
# baseline, forward returns, build event rows (verbatim from v2, with
# day_idx retained per event for the declustering step below).
# ============================================================
event_rows = []
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
    day_idx = np.searchsorted(dates, filing_dates, side="left")

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

    fwd = {}
    for h in HORIZONS:
        f = np.full(n, np.nan)
        last_D = n - 2 - h
        if last_D >= 0:
            Ds = np.arange(0, last_D + 1)
            f[Ds] = opens[Ds + 1 + h] / opens[Ds + 1] - 1
        fwd[h] = f

    stock_clean_means = {}
    for h in HORIZONS:
        vals = fwd[h][clean_positions]
        vals = vals[~np.isnan(vals)]
        stock_clean_means[h] = float(vals.mean()) if len(vals) >= MIN_CLEAN_OBS else np.nan

    for idx, items, accession in zip(day_idx, g["items"].to_numpy(), g["accession"].to_numpy()):
        entry_idx = idx + 1
        if idx >= n or entry_idx >= n:
            continue

        lo = max(0, idx - SIZE_LOOKBACK_DAYS)
        window_days = idx - lo
        size_proxy = (float(np.median(closes[lo:idx] * volumes[lo:idx]))
                      if window_days >= MIN_SIZE_LOOKBACK else np.nan)

        row = {"ticker": ticker, "items": items if isinstance(items, str) else "",
               "accession": accession, "size_proxy": size_proxy, "day_idx": int(idx)}
        for h in HORIZONS:
            event_ret = fwd[h][idx] if idx < n else np.nan
            baseline = stock_clean_means.get(h, np.nan)
            row[f"adj_ar_{h}"] = (event_ret - baseline) if (not np.isnan(event_ret) and not np.isnan(baseline)) else np.nan
        event_rows.append(row)

events_df = pd.DataFrame(event_rows)
print(f"\nEvents with usable forward-return windows: {len(events_df)}")
print(f"Tickers skipped (8-K filer but no price data in our universe): {len(skipped_no_price)}")


# ============================================================
# CORRECTION 1: DECLUSTERING -- across ALL of a ticker's 8-Ks, before
# any item-code filter (a cluster can mix item codes for one event).
# ============================================================
def decluster(df, window=DECLUSTER_WINDOW):
    kept_idx = []
    for ticker, g in df.sort_values("day_idx").groupby("ticker"):
        last_kept = -10 ** 9
        for i, pos in zip(g.index, g["day_idx"]):
            if pos - last_kept >= window:
                kept_idx.append(i)
                last_kept = pos
    return df.loc[kept_idx].copy()


print()
print("=" * 100)
print(f"CORRECTION 1: DECLUSTERING (keep first 8-K per ticker per {DECLUSTER_WINDOW}-trading-day window)")
print("=" * 100)
declustered_all = decluster(events_df)
print(f"All items pooled: {len(events_df)} events before -> {len(declustered_all)} after "
      f"({len(declustered_all) / len(events_df) * 100:.1f}% survive)")

item_502_before = events_df[events_df["items"].str.contains(FOCUS_ITEM, regex=False, na=False)]
item_502_after = declustered_all[declustered_all["items"].str.contains(FOCUS_ITEM, regex=False, na=False)]
print(f"Item {FOCUS_ITEM} only:  {len(item_502_before)} events before -> {len(item_502_after)} after "
      f"({len(item_502_after) / len(item_502_before) * 100:.1f}% survive)"
      if len(item_502_before) else f"Item {FOCUS_ITEM} only: 0 events before declustering")


# ============================================================
# CORRECTION 2: LIQUIDITY SPLIT -- median split by trailing 60-day $ volume,
# computed within each declustered scope (all-items, item 5.02) separately.
# ============================================================
def liquidity_halves(df):
    valid = df.dropna(subset=["size_proxy"])
    n_no_size = len(df) - len(valid)
    if len(valid) == 0:
        return valid, valid, float("nan"), n_no_size
    median_proxy = valid["size_proxy"].median()
    low = valid[valid["size_proxy"] < median_proxy]
    high = valid[valid["size_proxy"] >= median_proxy]
    return low, high, median_proxy, n_no_size


def report_scope(label, declustered_df, n_before):
    print()
    print("=" * 100)
    print(f"SCOPE: {label}")
    print("=" * 100)
    print(f"n before declustering: {n_before}   |   n after declustering: {len(declustered_df)}")

    low, high, median_proxy, n_no_size = liquidity_halves(declustered_df)
    print(f"Median 60-day $ volume (declustered events): ${median_proxy:,.0f}  "
          f"({n_no_size} events excluded -- insufficient trailing history)")

    for half_label, half_df in [("ILLIQUID half (below median $ volume)", low),
                                 ("LIQUID half (at/above median $ volume)", high)]:
        print(f"\n  {half_label} -- {len(half_df)} events")
        header = f"  {'Horizon':<10}{'n':>8}{'MeanAdjAR':>12}{'t-stat':>9}   {'vs real 0.009% cost'}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for h in HORIZONS:
            vals = half_df[f"adj_ar_{h}"].dropna().to_numpy()
            n, mean, t = t_stat(vals)
            if n < 2 or np.isnan(mean):
                print(f"  {str(h) + 'd':<10}{n:>8}{'n/a':>12}{'n/a':>9}   n/a")
                continue
            significant = (not np.isnan(t)) and abs(t) > 2
            if not significant:
                verdict = "not significant"
            elif abs(mean) > REAL_ROUNDTRIP_COST:
                verdict = f"significant AND exceeds {REAL_ROUNDTRIP_COST*100:.3f}% real cost -- economically tradeable"
            else:
                verdict = f"significant but somehow below {REAL_ROUNDTRIP_COST*100:.3f}% cost (essentially never, at this cost level)"
            old_cost_note = "" if abs(mean) <= OLD_ROUNDTRIP_COST else f"  [also clears the OLD {OLD_ROUNDTRIP_COST*100:.2f}% assumption]"
            print(f"  {str(h) + 'd':<10}{n:>8}{mean * 100:>11.3f}%{t:>9.2f}   {verdict}{old_cost_note}")


report_scope("ALL ITEMS POOLED", declustered_all, len(events_df))
report_scope(f"ITEM {FOCUS_ITEM} ONLY (executive/director departures)", item_502_after, len(item_502_before))

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)
print("This is a correctness re-run of Test 9 under a corrected cost input (real Alpaca cost")
print(f"~{REAL_ROUNDTRIP_COST*100:.3f}% vs the ~{OLD_ROUNDTRIP_COST*100:.2f}% originally assumed), with the two robustness")
print("corrections Test 14 found necessary (declustering, liquidity split) applied for the first")
print("time to this event study. No item code beyond 5.02 was examined, no strategy was built, and")
print(f"the {DECLUSTER_WINDOW}-day decluster window was fixed in advance (matching Test 14's Check 1), not tuned here.")
print("Everything before 2019-01-01 remains sealed.")
