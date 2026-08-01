"""
fetch_alpaca_news_wide.py -- Test 14, Task 1: widen the sentiment
universe beyond the original 11 hand-picked tickers (which Test 13's
FinBERT IC scan found were dominated by GOOGL/NVDA article volume,
diluting everything else). Fetches headlines + dates ONLY -- no FinBERT
scoring here, since Test 14 only needs to score the much smaller
shock-day subset (research/signal_tests/sentiment_shock_test.py), not
this full corpus.

Universe: 100 tickers, randomly sampled with seed 42 from the tickers
already present in data/yf_universe.parquet (2,342 tickers) -- same
random.Random(42).sample() convention as fetch_yf_universe.py, just
applied to the already-fetched ticker list instead of a fresh NASDAQ
Trader pull, since price history for the sample needs to already exist
for the later forward-return calculation.

Combined multi-symbol query (one paginated stream for all 100 tickers
at once, same as sentiment_test.py) -- verified this stays well under
any URL-length concern (100 tickers ~= 450 characters in the symbols
param). Checkpointed every 100 pages so a crash doesn't lose progress.

CRITICAL boundary fix (found the hard way in check_alpaca_news_api.py):
date-only start=end values are a zero-width window to this API. All
start/end values here carry explicit T00:00:00Z components.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import random
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

UNIVERSE_PATH = os.path.join("data", "yf_universe.parquet")
OUT_PATH = os.path.join("data", "alpaca_news_wide.parquet")
SAMPLE_SIZE = 100
SEED = 42
START = "2021-01-01"

NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
CHECKPOINT_EVERY_PAGES = 100

API_KEY = os.environ.get("APCA_API_KEY_ID")
API_SECRET = os.environ.get("APCA_API_SECRET_KEY")
if not API_KEY or not API_SECRET:
    raise SystemExit("APCA_API_KEY_ID / APCA_API_SECRET_KEY not found in .env.")
HEADERS = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": API_SECRET}

END = datetime.now(timezone.utc).strftime("%Y-%m-%d")

print("=" * 100)
print("TICKER SAMPLE")
print("=" * 100)
universe_tickers = sorted(pd.read_parquet(UNIVERSE_PATH, columns=["ticker"])["ticker"].unique())
print(f"Tickers available in {UNIVERSE_PATH}: {len(universe_tickers)}")
sample = random.Random(SEED).sample(universe_tickers, SAMPLE_SIZE)
print(f"Random sample (seed={SEED}): {SAMPLE_SIZE} tickers")
print(sample)

if os.path.exists(OUT_PATH):
    existing = pd.read_parquet(OUT_PATH)
    print(f"\nFound existing {OUT_PATH} ({len(existing)} articles) -- loading instead of refetching.")
    print("Delete this file first if you want a full refetch.")
else:
    symbols_param = ",".join(sample)
    all_rows = []
    seen_ids = set()
    page_token = None
    page_count = 0

    print()
    print("=" * 100)
    print("FETCH")
    print("=" * 100)
    print(f"Fetching news for {SAMPLE_SIZE} tickers, {START} to {END} (one combined multi-symbol stream)...")
    while True:
        params = {"start": f"{START}T00:00:00Z", "end": f"{END}T23:59:59Z",
                  "symbols": symbols_param, "limit": 50, "sort": "asc"}
        if page_token:
            params["page_token"] = page_token

        resp = requests.get(NEWS_URL, headers=HEADERS, params=params, timeout=30)
        if resp.status_code != 200:
            print(f"HTTP {resp.status_code} on page {page_count}: {resp.text[:300]}")
            time.sleep(2)
            continue

        data = resp.json()
        for a in data.get("news", []):
            if a["id"] in seen_ids:
                continue
            seen_ids.add(a["id"])
            all_rows.append({
                "id": a["id"],
                "headline": a["headline"],
                "created_at": a["created_at"],
                "symbols": list(a.get("symbols", [])),
            })

        page_token = data.get("next_page_token")
        page_count += 1
        if page_count % CHECKPOINT_EVERY_PAGES == 0:
            pd.DataFrame(all_rows).to_parquet(OUT_PATH, index=False)
            print(f"  [page {page_count}] checkpoint written -- {len(all_rows)} unique articles so far "
                  f"(latest: {all_rows[-1]['created_at']})", flush=True)

        if not page_token:
            break
        time.sleep(0.05)  # ~200/min headroom, same pacing as sentiment_test.py

    existing = pd.DataFrame(all_rows)
    existing.to_parquet(OUT_PATH, index=False)
    print(f"Fetch complete: {len(existing)} unique articles across {page_count} pages, saved to {OUT_PATH}")

print()
print("=" * 100)
print("COVERAGE SUMMARY")
print("=" * 100)
exploded = existing.explode("symbols").rename(columns={"symbols": "symbol"})
exploded = exploded[exploded["symbol"].isin(sample)]
per_ticker = exploded.groupby("symbol").size().reindex(sample, fill_value=0).sort_values(ascending=False)
print(f"Total articles: {len(existing)}")
print(f"Tickers with at least one article: {(per_ticker > 0).sum()} / {SAMPLE_SIZE}")
print(f"Articles per ticker -- min {per_ticker.min()}, median {per_ticker.median():.0f}, "
      f"mean {per_ticker.mean():.1f}, max {per_ticker.max()} ({per_ticker.idxmax()})")
print("\nTop 10 tickers by article count:")
print(per_ticker.head(10).to_string())
print("\nBottom 10 tickers by article count (includes zero-coverage names):")
print(per_ticker.tail(10).to_string())
