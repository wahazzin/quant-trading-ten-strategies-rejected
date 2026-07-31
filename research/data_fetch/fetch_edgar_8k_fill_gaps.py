"""
fetch_edgar_8k_fill_gaps.py -- targeted supplemental fetch to close the
coverage gap the bulk archive left behind.

fetch_edgar_8k_bulk.py's single combined submissions.zip carries each
company's "filings.recent" (most recent slice) but NOT the separately
paginated "filings.files" JSONs the live per-company API exposes for
older history. 1,577 of 2,339 tickers (67%) had "recent" not reaching
back to 2019-01-01 -- data/edgar_8k_gap_tickers.csv lists exactly which.

This script hits the LIVE per-company API (data.sec.gov/submissions/)
ONLY for those 1,577 tickers -- not all 2,339 -- to fetch the specific
older paginated files needed to reach the 2019-01-01 coverage floor,
then merges the extra 8-Ks into data/edgar_8k.parquet (deduplicated by
accession number).

Crash-proof by design (the original full per-company attempt ran ~2
hours and lost everything to a machine restart because it only wrote
output at the very end): checkpoints to data/edgar_8k_gapfill_partial.parquet
every 100 tickers, and resumes from the checkpoint on restart instead of
starting over. Prints progress with an ETA.

Rate limit: EDGAR allows up to 10 requests/second; this uses ~9/s.
"""
import os
import time
import json
import urllib.request
import urllib.error

import pandas as pd

USER_AGENT = "tradingbot research your.email@example.com"
GAP_TICKERS_PATH = os.path.join("data", "edgar_8k_gap_tickers.csv")
CIK_MAP_PATH = os.path.join("data", "cik_map.csv")
MAIN_PARQUET_PATH = os.path.join("data", "edgar_8k.parquet")
CHECKPOINT_PATH = os.path.join("data", "edgar_8k_gapfill_partial.parquet")
PROGRESS_STATE_PATH = os.path.join("data", "edgar_8k_gapfill_progress.json")
SLEEP_BETWEEN_REQUESTS = 0.11
COVERAGE_FLOOR = "2019-01-01"
CHECKPOINT_EVERY = 100


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


gap_tickers = pd.read_csv(GAP_TICKERS_PATH)["ticker"].tolist()
cik_map = pd.read_csv(CIK_MAP_PATH, dtype={"cik": int}).set_index("ticker")["cik"].to_dict()

# resume support
done_tickers = set()
checkpoint_rows = []
if os.path.exists(CHECKPOINT_PATH):
    prior = pd.read_parquet(CHECKPOINT_PATH)
    checkpoint_rows = prior.to_dict("records")
    done_tickers = set(prior["ticker"].unique())
    print(f"Resuming: {len(done_tickers)} tickers already done in {CHECKPOINT_PATH} "
          f"({len(checkpoint_rows)} rows)", flush=True)

remaining = [t for t in gap_tickers if t not in done_tickers]
print(f"Gap tickers total: {len(gap_tickers)}, remaining to fetch: {len(remaining)}", flush=True)

all_rows = checkpoint_rows
t_start = time.time()

for i, ticker in enumerate(remaining):
    cik = cik_map.get(ticker)
    if cik is None:
        continue
    cik_str = f"{cik:010d}"
    data = fetch_json(f"https://data.sec.gov/submissions/CIK{cik_str}.json")
    time.sleep(SLEEP_BETWEEN_REQUESTS)

    events = []
    if data is not None:
        recent = data.get("filings", {}).get("recent", {})
        events.extend(extract_8k(recent))
        for f in data.get("filings", {}).get("files", []):
            if f.get("filingTo", "0000-00-00") >= COVERAGE_FLOOR:
                page = fetch_json(f"https://data.sec.gov/submissions/{f['name']}")
                time.sleep(SLEEP_BETWEEN_REQUESTS)
                if page is not None:
                    events.extend(extract_8k(page))

    for e in events:
        all_rows.append({"ticker": ticker, "cik": cik, **e})

    if (i + 1) % CHECKPOINT_EVERY == 0 or (i + 1) == len(remaining):
        pd.DataFrame(all_rows).to_parquet(CHECKPOINT_PATH, index=False)
        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta_sec = (len(remaining) - (i + 1)) / rate if rate > 0 else float("nan")
        print(f"[{i+1}/{len(remaining)}] checkpointed ({len(all_rows)} rows total) -- "
              f"{rate:.2f} tickers/s, ETA {eta_sec/60:.1f} min", flush=True)

print("\nGap-fill fetch complete. Merging with the main 8-K table...", flush=True)

gap_df = pd.DataFrame(all_rows)
gap_df["filing_date"] = pd.to_datetime(gap_df["filing_date"])

main_df = pd.read_parquet(MAIN_PARQUET_PATH)
main_df["filing_date"] = pd.to_datetime(main_df["filing_date"])

combined = pd.concat([main_df, gap_df], ignore_index=True)
before = len(combined)
combined = combined.drop_duplicates(subset=["accession"]).reset_index(drop=True)
combined = combined.sort_values(["ticker", "filing_date"]).reset_index(drop=True)

combined.to_parquet(MAIN_PARQUET_PATH, index=False)

if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Gap tickers processed: {len(gap_tickers)}")
print(f"New rows fetched: {len(gap_df)}")
print(f"Duplicate rows dropped on merge: {before - len(combined)}")
print(f"Final combined total 8-K filings: {len(combined)}")
print(f"Companies with at least one 8-K: {combined['ticker'].nunique()}")
print(f"Date coverage: {combined['filing_date'].min().date()} to {combined['filing_date'].max().date()}")
print(f"Filings from {COVERAGE_FLOOR} onward: {int((combined['filing_date'] >= COVERAGE_FLOOR).sum())}")
print(f"Saved to {MAIN_PARQUET_PATH}")
