"""
event_study.py -- does SEC 8-K material-event filing content carry
tradeable information? This is an EVENT STUDY, not a strategy: the
first question is whether the signal contains information at all.

Evidence boundary: this script uses ONLY data with date >= 2019-01-01,
enforced by truncating the raw price panels and the 8-K filing list
immediately after loading, before any other computation. The prior
holdout (pre-2019 is fine there; 2019+ was spent, see RESEARCH_LOG.md
Test 8) was for the price-only factor hypothesis class. This is a
genuinely different hypothesis -- filing content, not price history --
so it gets its own fresh holdout: everything before 2019-01-01 is now
sealed for any strategy eventually built from this event-study result,
and is not touched anywhere in this script.

Execution, no lookahead: a filing's calendar date maps to "day N" = the
first trading day on or after that date (filings occurring on a non-
trading day are treated as known as of the next trading day). The event
is acted on at day N+1's open. Forward returns are computed open-to-open
at 1/5/10/20 trading days. Abnormal return = stock return minus SPY's
return over the identical calendar window (same entry/exit dates), to
isolate event impact from market-wide drift.

No strategy is built, nothing is tuned, and no item code is selected as
"best performing" -- this reports exactly what the pooled data shows.
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


# ============================================================
# LOAD + EVIDENCE BOUNDARY
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
print("This is a NEW, fresh holdout boundary for the event-study hypothesis class --")
print("distinct from the price-factor holdout already spent (RESEARCH_LOG.md Test 8).")
print(f"Everything before {EVIDENCE_START.date()} is sealed for any strategy built from this.")


# ============================================================
# BUILD PER-TICKER PRICE LOOKUPS + SPY OPEN LOOKUP
# ============================================================
spy_open_lookup = pd.Series(spy["open"].to_numpy(dtype=float), index=spy["date"])

ticker_groups = {t: g.reset_index(drop=True) for t, g in price.groupby("ticker")}

print(f"\nTickers with price data: {len(ticker_groups)}")
print(f"Unique tickers with 8-K filings: {edgar['ticker'].nunique()}")


# ============================================================
# BUILD EVENT-LEVEL FORWARD/ABNORMAL RETURNS
# ============================================================
results = []
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
    day_idx = np.searchsorted(dates, filing_dates, side="left")  # first trading day >= filing date

    for idx, items, accession in zip(day_idx, g["items"].to_numpy(), g["accession"].to_numpy()):
        entry_idx = idx + 1
        if idx >= n or entry_idx >= n:
            continue   # filing at/after the last available trading day -- nothing to enter into

        entry_open = opens[entry_idx]
        entry_date = dates[entry_idx]

        lo = max(0, idx - SIZE_LOOKBACK_DAYS)
        window_days = idx - lo
        size_proxy = (float(np.median(closes[lo:idx] * volumes[lo:idx]))
                      if window_days >= MIN_SIZE_LOOKBACK else np.nan)

        row = {"ticker": ticker, "items": items if isinstance(items, str) else "",
               "accession": accession, "size_proxy": size_proxy}

        for h in HORIZONS:
            exit_idx = entry_idx + h
            if exit_idx >= n:
                row[f"ar_{h}"] = np.nan
                continue
            exit_open = opens[exit_idx]
            exit_date = dates[exit_idx]
            stock_ret = exit_open / entry_open - 1
            spy_entry = spy_open_lookup.get(entry_date, np.nan)
            spy_exit = spy_open_lookup.get(exit_date, np.nan)
            if pd.isna(spy_entry) or pd.isna(spy_exit):
                row[f"ar_{h}"] = np.nan
            else:
                row[f"ar_{h}"] = stock_ret - (spy_exit / spy_entry - 1)

        results.append(row)

events_df = pd.DataFrame(results)
print(f"\nEvents with usable forward-return windows: {len(events_df)}")
print(f"Tickers skipped (8-K filer but no price data in our universe): {len(skipped_no_price)}")


# ============================================================
# REPORTING
# ============================================================
def compute_row(df, horizon):
    vals = df[f"ar_{horizon}"].dropna().to_numpy()
    n = len(vals)
    if n < 2:
        return n, float("nan"), float("nan")
    mean = float(vals.mean())
    std = float(vals.std(ddof=1))
    t = mean / (std / np.sqrt(n)) if std > 0 else float("nan")
    return n, mean, t


def print_group(label, df):
    print(f"\n{label} -- {len(df)} events")
    header = f"{'Horizon':<10}{'n':>8}{'MeanAR':>10}{'t-stat':>9}"
    print(header)
    print("-" * len(header))
    for h in HORIZONS:
        n, mean, t = compute_row(df, h)
        mean_str = f"{mean*100:>9.3f}%" if not np.isnan(mean) else f"{'n/a':>10}"
        t_str = f"{t:>9.2f}" if not np.isnan(t) else f"{'n/a':>9}"
        print(f"{str(h)+'d':<10}{n:>8}{mean_str}{t_str}")


print()
print("=" * 96)
print("OVERALL: pooled abnormal returns across all 8-K events")
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
