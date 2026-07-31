"""
fetch_yf_universe.py -- random-sampled US common-stock daily history via
yfinance (free, no API key/account needed).

Scope history: an earlier full-universe attempt (5,494 currently-listed
tickers) was stopped as over-scoped and hit cumulative rate limiting. A
follow-up 600-ticker pilot fetched cleanly (93.7% success, no rate
limiting) but, after the price/volume/history filters, averaged only
9.3 stocks per decile in the practice window -- too few to trust. This
run draws 2,500 tickers instead (same seed, same proven pacing: batches
of 50, 2s sleep, one retry pass) specifically to fix that diversification
problem, since 2,500 is still well inside the range that fetched cleanly
before (the earlier failures only appeared deep into a 5,494-ticker
single session).

Universe: NASDAQ + NYSE/AMEX ("otherlisted") symbols from NASDAQ
Trader's free, no-auth symbol directories, filtered to ETF == 'N',
Test Issue == 'N', and ticker containing none of $ . ^. From that list,
tickers are drawn via Python's random.Random(seed=42).sample() -- a
RANDOM sample, not the first N alphabetically, and not hand-picked or
filtered on any performance measure.

This is CURRENTLY LISTED tickers only -- a deliberately accepted
survivorship bias (see chat): conservative for momentum (the worst
historical losers we'd short are absent, understating the edge) and
minimal for low volatility. This is why long-term reversal is not
tested in factor_test.py.

Saves the combined long-format table to data/yf_universe.parquet with
columns: ticker, date, open, high, low, close, volume. Prices are
split- and dividend-adjusted (auto_adjust=True).

After saving, also reports the average ELIGIBLE universe size per month
(same price/volume/history filters and holdout boundary factor_test.py
uses) -- that number, not raw ticker count, is what actually determines
decile size, and it's the thing the prior pilot got wrong.
"""
import os
import random
import time
import urllib.request
from io import StringIO

import numpy as np
import pandas as pd
import yfinance as yf

START = "2006-01-01"
END = "2026-07-01"
SAMPLE_SIZE = 2500
SEED = 42
BATCH_SIZE = 50
SLEEP_BETWEEN_BATCHES = 2
OUT_PATH = os.path.join("data", "yf_universe.parquet")

# must match factor_test.py exactly, for the eligible-universe-size preview below
HOLDOUT_START = pd.Timestamp("2019-01-01")
PRICE_MIN, PRICE_MAX = 5.0, 100.0
MIN_ADV = 250_000
MIN_HISTORY_MONTHS = 24

NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
BAD_CHARS = ("$", ".", "^")


def fetch_text(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def build_universe():
    nas = pd.read_csv(StringIO(fetch_text(NASDAQ_URL)), sep="|")
    nas = nas[nas["Symbol"].notna()]
    nas = nas[~nas["Symbol"].astype(str).str.contains("File Creation Time", na=False)]
    oth = pd.read_csv(StringIO(fetch_text(OTHER_URL)), sep="|")
    oth = oth[oth["ACT Symbol"].notna()]
    oth = oth[~oth["ACT Symbol"].astype(str).str.contains("File Creation Time", na=False)]

    nas_common = nas[(nas["Test Issue"] == "N") & (nas["ETF"] == "N")]["Symbol"].astype(str)
    oth_common = oth[(oth["Test Issue"] == "N") & (oth["ETF"] == "N")]["ACT Symbol"].astype(str)

    syms = sorted(set(nas_common) | set(oth_common))
    return [s for s in syms if not any(ch in s for ch in BAD_CHARS)]


def to_long(sub, ticker):
    sub = sub.reset_index()
    sub = sub.rename(columns={"Date": "date", "Open": "open", "High": "high",
                               "Low": "low", "Close": "close", "Volume": "volume"})
    sub["ticker"] = ticker
    return sub[["ticker", "date", "open", "high", "low", "close", "volume"]]


print("Building full ticker universe from NASDAQ Trader symbol directories...", flush=True)
full_universe = build_universe()
print(f"Full common-stock universe (pre-sample): {len(full_universe)} tickers", flush=True)

rng = random.Random(SEED)
sample = rng.sample(full_universe, SAMPLE_SIZE)
print(f"Random sample drawn (seed={SEED}): {SAMPLE_SIZE} tickers", flush=True)

results = {}
failed = []

n_batches = (len(sample) + BATCH_SIZE - 1) // BATCH_SIZE
for b in range(n_batches):
    batch = sample[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
    t0 = time.time()
    try:
        data = yf.download(batch, start=START, end=END, group_by="ticker",
                            threads=True, progress=False, auto_adjust=True)
    except Exception as e:
        print(f"[batch {b+1}/{n_batches}] failed entirely: {e}", flush=True)
        failed.extend(batch)
        time.sleep(SLEEP_BETWEEN_BATCHES)
        continue

    got_cols = set(data.columns.get_level_values(0)) if isinstance(data.columns, pd.MultiIndex) else set()
    ok = 0
    for t in batch:
        if t not in got_cols:
            failed.append(t)
            continue
        sub = data[t].dropna(how="all")
        if sub.empty:
            failed.append(t)
            continue
        results[t] = to_long(sub, t)
        ok += 1

    print(f"[batch {b+1}/{n_batches}] {time.time()-t0:.1f}s, {ok}/{len(batch)} ok", flush=True)
    time.sleep(SLEEP_BETWEEN_BATCHES)

# one light retry pass for anything that failed (covers transient rate limiting)
if failed:
    print(f"\nRetrying {len(failed)} failed tickers once, smaller batches, longer pause...", flush=True)
    retry_list = failed
    failed = []
    retry_batch_size = 20
    n_retry_batches = (len(retry_list) + retry_batch_size - 1) // retry_batch_size
    for b in range(n_retry_batches):
        batch = retry_list[b * retry_batch_size:(b + 1) * retry_batch_size]
        try:
            data = yf.download(batch, start=START, end=END, group_by="ticker",
                                threads=True, progress=False, auto_adjust=True)
        except Exception as e:
            print(f"  retry batch {b+1}/{n_retry_batches} failed entirely: {e}", flush=True)
            failed.extend(batch)
            time.sleep(8)
            continue
        got_cols = set(data.columns.get_level_values(0)) if isinstance(data.columns, pd.MultiIndex) else set()
        for t in batch:
            if t in got_cols:
                sub = data[t].dropna(how="all")
                if not sub.empty:
                    results[t] = to_long(sub, t)
                    continue
            failed.append(t)
        time.sleep(8)

print("\nCombining results...", flush=True)
if not results:
    raise SystemExit("No tickers returned usable data -- stopping rather than proceeding on nothing.")

panel = pd.concat(list(results.values()), ignore_index=True)
panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

os.makedirs("data", exist_ok=True)
panel.to_parquet(OUT_PATH, index=False)

counts = panel.groupby("ticker")["date"].agg(["min", "max", "count"])
span_years = (counts["max"] - counts["min"]).dt.days / 365.25
n_15plus = int((span_years >= 15).sum())
fail_frac = len(failed) / len(sample)

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Sampled tickers: {len(sample)}")
print(f"Tickers with usable data: {panel['ticker'].nunique()}")
print(f"Tickers that failed (after retry): {len(failed)} ({fail_frac*100:.1f}% of sample)")
print(f"Total rows: {len(panel)}")
print(f"Date range: {panel['date'].min().date()} to {panel['date'].max().date()}")
print(f"Tickers with 15+ years of history: {n_15plus}")
print(f"Saved to {OUT_PATH}")

if fail_frac > 0.15:
    print()
    print(f"WARNING: {fail_frac*100:.1f}% of the sample failed to fetch (rate limiting or no data).")
    print("This is a meaningfully large failure rate -- reporting it plainly rather than")
    print("treating the resulting panel as if it were the full clean sample.")
    print(f"Failed tickers: {failed}")

# ------------------------------------------------------------------
# Eligible-universe-size preview (same filters + holdout boundary as
# factor_test.py) -- this is the number that actually sets decile size,
# not raw ticker count.
# ------------------------------------------------------------------
print()
print("=" * 60)
print("ELIGIBLE UNIVERSE SIZE PREVIEW (practice window, same filters as factor_test.py)")
print("=" * 60)

practice = panel[panel["date"] < HOLDOUT_START].copy()
practice["month"] = practice["date"].dt.to_period("M")

close_last = practice.sort_values("date").groupby(["ticker", "month"])["close"].last()
vol_mean = practice.groupby(["ticker", "month"])["volume"].mean()
prev = pd.DataFrame({"close": close_last, "avg_volume": vol_mean}).reset_index()
prev = prev.sort_values(["ticker", "month"]).reset_index(drop=True)
prev["hist_months"] = prev.groupby("ticker").cumcount()

eligible = (
    (prev["close"] >= PRICE_MIN) & (prev["close"] <= PRICE_MAX) &
    (prev["avg_volume"] > MIN_ADV) &
    (prev["hist_months"] >= MIN_HISTORY_MONTHS)
)
per_month = prev.loc[eligible].groupby("month").size()
avg_universe = float(per_month.mean()) if len(per_month) else 0.0
avg_decile = avg_universe / 10

print(f"Months in practice window: {practice['month'].nunique()}")
print(f"Average eligible universe size per month: {avg_universe:.1f}")
print(f"Implied average decile size: {avg_decile:.1f}")
if avg_decile < 25:
    print(f"WARNING: average decile size ({avg_decile:.1f}) is still under 25 -- "
          f"the factor test result should be treated with corresponding caution.")
else:
    print(f"Average decile size ({avg_decile:.1f}) clears the 25-stock bar set for this run.")
