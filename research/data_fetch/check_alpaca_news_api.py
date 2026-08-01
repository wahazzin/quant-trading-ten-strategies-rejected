"""
check_alpaca_news_api.py -- one-off diagnostic: does Alpaca's news API
(v1beta1/news) support historical date-range queries, and if so, how far
back does coverage actually go? This determines whether a news-sentiment
signal can be BACKTESTED (needs multi-year historical coverage) or only
FORWARD-TESTED (if the API only serves recent articles regardless of
what start/end dates are requested).

Reads keys from .env -- NEVER hardcode credentials here. Supports either
of Alpaca's two common env var naming conventions:
  APCA_API_KEY_ID / APCA_API_SECRET_KEY        (Alpaca's own SDK convention)
  ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY    (common alternate naming)

Checks, in order:
  1. A recent (last 7 days) query -- sanity check that the endpoint and
     credentials work at all, independent of any historical-coverage
     question.
  2. A 2020 date range (as requested) for a single liquid ticker (AAPL).
  3. A bisection-style sweep across earlier years (2018, 2016, 2015) to
     find approximately where coverage starts, IF the 2020 query
     succeeds -- otherwise this is skipped, since there'd be nothing to
     bracket.
  4. Rate-limit headers from the responses (if Alpaca sends them).
  5. Article density: count articles for one ticker over one full day,
     to give a rough per-ticker-per-day estimate.
"""
import os
import sys
import time
import json
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
TEST_TICKER = "AAPL"

API_KEY = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY_ID")
API_SECRET = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_API_SECRET_KEY")

if not API_KEY or not API_SECRET:
    print("No Alpaca API credentials found in .env.")
    print("Expected one of these env var pairs:")
    print("  APCA_API_KEY_ID / APCA_API_SECRET_KEY")
    print("  ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY")
    print("Add them to .env and re-run -- nothing else in this script will work without them.")
    sys.exit(1)

HEADERS = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": API_SECRET}


def query_news(start, end, symbols=None, limit=50, page_token=None):
    params = {"start": start, "end": end, "limit": min(limit, 50), "sort": "asc"}
    if symbols:
        params["symbols"] = symbols
    if page_token:
        params["page_token"] = page_token
    resp = requests.get(NEWS_URL, headers=HEADERS, params=params, timeout=30)
    return resp


def summarize(resp, label):
    print(f"--- {label} ---")
    print(f"HTTP status: {resp.status_code}")
    rate_headers = {k: v for k, v in resp.headers.items() if "rate" in k.lower() or "limit" in k.lower()}
    if rate_headers:
        print(f"Rate-limit headers: {rate_headers}")
    if resp.status_code != 200:
        print(f"Response body: {resp.text[:500]}")
        print()
        return None
    data = resp.json()
    articles = data.get("news", [])
    print(f"Articles returned: {len(articles)}")
    if articles:
        dates = sorted(a["created_at"] for a in articles)
        print(f"Article date range in response: {dates[0]} to {dates[-1]}")
        print(f"Sample headline: {articles[0].get('headline', '')[:100]!r}")
    print()
    return data


print("=" * 90)
print("CHECK 1: sanity check -- recent news (last 7 days), no symbol filter")
print("=" * 90)
now = datetime.now(timezone.utc)
recent_start = (now - timedelta(days=7)).strftime("%Y-%m-%d")
recent_end = now.strftime("%Y-%m-%d")
resp1 = query_news(recent_start, recent_end)
data1 = summarize(resp1, f"Recent window {recent_start} to {recent_end}")
sanity_ok = resp1.status_code == 200 and data1 is not None and len(data1.get("news", [])) > 0
print(f"Sanity check result: {'API + credentials working' if sanity_ok else 'FAILED -- stopping here'}")
if not sanity_ok:
    sys.exit(1)

time.sleep(1)
print()
print("=" * 90)
print(f"CHECK 2: 2020 historical range for {TEST_TICKER} (the requested test)")
print("=" * 90)
resp2 = query_news("2020-01-01", "2020-01-31", symbols=TEST_TICKER)
data2 = summarize(resp2, f"{TEST_TICKER}, 2020-01-01 to 2020-01-31")
year_2020_has_coverage = resp2.status_code == 200 and data2 is not None and len(data2.get("news", [])) > 0
print(f"Does the 2020 query return real 2020-dated articles: {year_2020_has_coverage}")

time.sleep(1)
print()
print("=" * 90)
print("CHECK 3: sweep earlier years to bracket how far back coverage goes")
print("=" * 90)
if not year_2020_has_coverage:
    print("Skipped -- 2020 itself returned nothing, so there's no coverage to bracket further back.")
else:
    for year in [2018, 2016, 2015]:
        resp = query_news(f"{year}-01-01", f"{year}-01-31", symbols=TEST_TICKER)
        data = summarize(resp, f"{TEST_TICKER}, {year}-01-01 to {year}-01-31")
        time.sleep(1)

print()
print("=" * 90)
print("CHECK 4: article density -- one ticker, one full day (paginated to get a real count)")
print("=" * 90)
# NOTE: date-only start=end (e.g. "2020-01-15" to "2020-01-15") is a zero-width window to
# this API and silently returns 0 articles -- not a real "no coverage" result. Explicit
# T00:00:00Z start-of-day to start-of-next-day boundaries are required for a single-day query.
density_day_start = "2020-01-15T00:00:00Z"
density_day_end = "2020-01-16T00:00:00Z"
total_n = 0
page_token = None
for _ in range(10):  # hard cap -- this is a density estimate, not an exhaustive archive pull
    resp4 = query_news(density_day_start, density_day_end, symbols=TEST_TICKER, limit=50, page_token=page_token)
    if resp4.status_code != 200:
        print(f"Page request failed: {resp4.status_code} {resp4.text[:200]}")
        break
    data4 = resp4.json()
    page_articles = data4.get("news", [])
    total_n += len(page_articles)
    page_token = data4.get("next_page_token")
    if not page_token:
        break
    time.sleep(0.3)
print(f"Articles for {TEST_TICKER} on 2020-01-15: {total_n} "
      f"(rough per-ticker-per-day estimate for a liquid large-cap name)")

print()
print("=" * 90)
print("SUMMARY")
print("=" * 90)
print(f"Endpoint reachable with these credentials: {sanity_ok}")
print(f"2020 date-range query returns real 2020 articles: {year_2020_has_coverage}")
print("Conclusion: see CHECK 2/3 output above for the coverage boundary -- if CHECK 2 returned "
      "articles dated in 2020 (not just recent articles regardless of the requested range), the "
      "API supports genuine historical backtesting for news sentiment. If CHECK 2 returned nothing "
      "or only recent-dated articles, treat this as forward-test-only until proven otherwise.")
