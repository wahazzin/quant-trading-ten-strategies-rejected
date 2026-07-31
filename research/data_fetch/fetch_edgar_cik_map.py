"""
fetch_edgar_cik_map.py -- map our yfinance universe tickers to SEC CIK
numbers, using SEC's free, no-auth ticker directory.

EDGAR requires a descriptive User-Agent with a real contact string on
every request; this script and fetch_edgar_8k.py both use the same one.

Source: https://www.sec.gov/files/company_tickers.json (current SEC-
registered tickers only -- this is why match rate isn't 100%: foreign
private issuers filing 6-K instead of 8-K, tickers with share-class
notation that differs between Yahoo and SEC (e.g. BRK.B vs BRK-B), and
any ticker SEC doesn't carry will legitimately fail to match).

Saves the matched map to data/cik_map.csv (ticker, cik, title).
"""
import os
import json
import urllib.request

import pandas as pd

USER_AGENT = "tradingbot research your.email@example.com"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
UNIVERSE_PATH = os.path.join("data", "yf_universe.parquet")
OUT_PATH = os.path.join("data", "cik_map.csv")


def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


print("Fetching SEC company_tickers.json...", flush=True)
raw = fetch_json(TICKERS_URL)
sec_map = {}
for row in raw.values():
    sec_map[str(row["ticker"]).upper()] = {"cik": int(row["cik_str"]), "title": row["title"]}
print(f"SEC directory: {len(sec_map)} tickers", flush=True)

universe = pd.read_parquet(UNIVERSE_PATH, columns=["ticker"])
our_tickers = sorted(universe["ticker"].unique())
print(f"Our universe: {len(our_tickers)} tickers", flush=True)

matched, unmatched = [], []
for t in our_tickers:
    hit = sec_map.get(t.upper())
    if hit:
        matched.append({"ticker": t, "cik": hit["cik"], "title": hit["title"]})
    else:
        unmatched.append(t)

matched_df = pd.DataFrame(matched).sort_values("ticker").reset_index(drop=True)
os.makedirs("data", exist_ok=True)
matched_df.to_csv(OUT_PATH, index=False)

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Matched:   {len(matched_df)} / {len(our_tickers)} "
      f"({len(matched_df) / len(our_tickers) * 100:.1f}%)")
print(f"Unmatched: {len(unmatched)}")
if unmatched:
    print(f"Sample unmatched tickers: {unmatched[:30]}{'...' if len(unmatched) > 30 else ''}")
print(f"Saved to {OUT_PATH}")
