"""
fetch_edgar_8k.py -- fetch 8-K filing history for every ticker matched
to a CIK in data/cik_map.csv (see fetch_edgar_cik_map.py), from SEC
EDGAR's free submissions API.

Rate limit: EDGAR allows up to 10 requests/second; this uses ~9/s
(0.11s sleep between requests) for a safety margin. Same User-Agent
contact string as fetch_edgar_cik_map.py, as EDGAR requires.

Each company's submissions JSON exposes the most recent ~1000 filings
directly under "filings.recent"; older filings are paginated under
"filings.files", each entry pre-labeled with the date range it covers
(filingFrom/filingTo) -- so which extra pages are needed can be decided
from that metadata WITHOUT fetching them first. This script only pulls
additional pages when "recent" doesn't already reach back to
2019-01-01 (COVERAGE_FLOOR), since that is as far back as
event_study.py's evidence boundary requires; older history is not
fetched to avoid wasted requests on data outside the study's window.

Saves the combined table to data/edgar_8k.parquet with columns:
ticker, cik, filing_date, accession, items.
"""
import os
import time
import json
import urllib.request
import urllib.error

import pandas as pd

USER_AGENT = "tradingbot research your.email@example.com"
CIK_MAP_PATH = os.path.join("data", "cik_map.csv")
OUT_PATH = os.path.join("data", "edgar_8k.parquet")
SLEEP_BETWEEN_REQUESTS = 0.11
COVERAGE_FLOOR = "2019-01-01"


def fetch_json(url, retries=3):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(1.0 * (attempt + 1))
        except Exception:
            time.sleep(1.0 * (attempt + 1))
    return None


def extract_8k(block):
    """block: a dict with parallel arrays (form, filingDate, accessionNumber,
    items, ...) -- the shape of both filings.recent and each paginated file."""
    rows = []
    forms = block.get("form", [])
    dates = block.get("filingDate", [])
    accessions = block.get("accessionNumber", [])
    items = block.get("items", [])
    for i, form in enumerate(forms):
        if form == "8-K":
            rows.append({
                "filing_date": dates[i],
                "accession": accessions[i],
                "items": items[i] if i < len(items) else "",
            })
    return rows


cik_map = pd.read_csv(CIK_MAP_PATH, dtype={"cik": int})
print(f"Loaded {len(cik_map)} ticker->CIK mappings", flush=True)

all_rows = []
errors = []

for i, row in enumerate(cik_map.itertuples(index=False)):
    ticker, cik = row.ticker, row.cik
    cik_str = f"{cik:010d}"
    data = fetch_json(f"https://data.sec.gov/submissions/CIK{cik_str}.json")
    time.sleep(SLEEP_BETWEEN_REQUESTS)

    if data is None:
        errors.append(ticker)
        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(cik_map)}] ... ({len(all_rows)} filings so far, "
                  f"{len(errors)} errors)", flush=True)
        continue

    recent = data.get("filings", {}).get("recent", {})
    events = extract_8k(recent)

    oldest_recent = min((e["filing_date"] for e in events), default=None)
    if oldest_recent is None or oldest_recent > COVERAGE_FLOOR:
        for f in data.get("filings", {}).get("files", []):
            if f.get("filingTo", "0000-00-00") >= COVERAGE_FLOOR:
                page = fetch_json(f"https://data.sec.gov/submissions/{f['name']}")
                time.sleep(SLEEP_BETWEEN_REQUESTS)
                if page is not None:
                    events.extend(extract_8k(page))

    for e in events:
        all_rows.append({"ticker": ticker, "cik": cik, **e})

    if (i + 1) % 100 == 0:
        print(f"[{i+1}/{len(cik_map)}] {ticker}: {len(events)} 8-Ks "
              f"({len(all_rows)} filings total so far, {len(errors)} errors)", flush=True)

print(f"\nDone. {len(all_rows)} total 8-K filings across "
      f"{cik_map['ticker'].nunique() - len(errors)} companies ({len(errors)} fetch errors)", flush=True)

df = pd.DataFrame(all_rows)
df["filing_date"] = pd.to_datetime(df["filing_date"])
df = df.sort_values(["ticker", "filing_date"]).reset_index(drop=True)

os.makedirs("data", exist_ok=True)
df.to_parquet(OUT_PATH, index=False)

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total 8-K filings: {len(df)}")
print(f"Companies with at least one 8-K: {df['ticker'].nunique()}")
print(f"Date coverage: {df['filing_date'].min().date()} to {df['filing_date'].max().date()}")
print(f"Filings from {COVERAGE_FLOOR} onward: {int((df['filing_date'] >= COVERAGE_FLOOR).sum())}")
print(f"Fetch errors: {len(errors)}")
print(f"Saved to {OUT_PATH}")
