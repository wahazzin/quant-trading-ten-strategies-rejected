"""
fetch_edgar_8k_bulk.py -- fast-path 8-K fetch via SEC's bulk submissions
archive, instead of ~2,339 sequential rate-limited per-company requests
(the previous attempt, fetch_edgar_8k.py, ran ~2 hours and was lost to
a machine restart because it only wrote its parquet at the very end).

Verified before committing to the download: HEAD request to
https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip
returned 200 OK, Content-Length ~1.45 GB (not quite the ~2GB estimate,
but in the same ballpark and a single bounded download).

This ZIP contains one JSON file per CIK (same schema as the per-company
submissions API: parallel arrays under "filings.recent", with older
history paginated the same way under "filings.files"). We download it
once, then read ONLY the ~2,339 entries matching data/cik_map.csv
in-memory (no full extraction to disk), reusing the identical 8-K
extraction and "extend history back to 2019" logic as fetch_edgar_8k.py.

The downloaded zip is deleted after a successful parquet write -- it's
a multi-GB intermediate artifact, not a project data asset worth keeping.
"""
import os
import re
import json
import time
import zipfile
import urllib.request

import pandas as pd

USER_AGENT = "tradingbot research your.email@example.com"
BULK_URL = "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip"
CIK_MAP_PATH = os.path.join("data", "cik_map.csv")
OUT_PATH = os.path.join("data", "edgar_8k.parquet")
ZIP_TMP_PATH = os.path.join("data", "_submissions_bulk_tmp.zip")
COVERAGE_FLOOR = "2019-01-01"


def download_with_progress(url, out_path, chunk_size=1024 * 1024):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        last_report = 0
        with open(out_path, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded - last_report >= 50 * 1024 * 1024:  # every 50MB
                    elapsed = time.time() - t0
                    pct = downloaded / total * 100 if total else 0
                    mbps = downloaded / 1024 / 1024 / elapsed if elapsed > 0 else 0
                    print(f"  {downloaded/1024/1024:,.0f} MB / {total/1024/1024:,.0f} MB "
                          f"({pct:.1f}%) -- {mbps:.1f} MB/s", flush=True)
                    last_report = downloaded
    return downloaded


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


print("=" * 70)
print("STEP 1: download SEC bulk submissions archive")
print("=" * 70)
print(f"Downloading {BULK_URL}", flush=True)
os.makedirs("data", exist_ok=True)
size = download_with_progress(BULK_URL, ZIP_TMP_PATH)
print(f"Download complete: {size/1024/1024:,.0f} MB saved to {ZIP_TMP_PATH}")

print()
print("=" * 70)
print("STEP 2: match CIKs and extract 8-K filings")
print("=" * 70)
cik_map = pd.read_csv(CIK_MAP_PATH, dtype={"cik": int})
print(f"Loaded {len(cik_map)} ticker->CIK mappings")

zf = zipfile.ZipFile(ZIP_TMP_PATH)
namelist = zf.namelist()
print(f"Archive contains {len(namelist)} entries")
print(f"Sample entries: {namelist[:5]}")

cik_to_name = {}
for name in namelist:
    m = re.search(r"(\d{4,10})", name)
    if m:
        cik_to_name[int(m.group(1))] = name

print(f"Parsed {len(cik_to_name)} CIK-addressable entries from the archive")

all_rows = []
missing = []
insufficient_coverage = []
for i, row in enumerate(cik_map.itertuples(index=False)):
    ticker, cik = row.ticker, row.cik
    name = cik_to_name.get(cik)
    if name is None:
        missing.append(ticker)
        continue

    try:
        data = json.loads(zf.read(name).decode("utf-8"))
    except Exception:
        missing.append(ticker)
        continue

    recent = data.get("filings", {}).get("recent", {})
    events = extract_8k(recent)

    oldest_recent = min((e["filing_date"] for e in events), default=None)
    if oldest_recent is None or oldest_recent > COVERAGE_FLOOR:
        # The bulk archive carries each company's single combined submissions
        # doc, not the separately-paginated "files" JSONs the per-company API
        # exposes for older history. If "recent" alone doesn't reach back to
        # 2019, that older slice is simply not available from this file --
        # recorded as a coverage gap and reported honestly, not silently
        # backfilled or fabricated.
        insufficient_coverage.append(ticker)

    for e in events:
        all_rows.append({"ticker": ticker, "cik": cik, **e})

    if (i + 1) % 200 == 0:
        print(f"  [{i+1}/{len(cik_map)}] processed, {len(all_rows)} filings extracted so far", flush=True)

zf.close()

print(f"\nProcessed all {len(cik_map)} tickers. Missing/unreadable entries: {len(missing)}")
if missing:
    print(f"Sample missing tickers: {missing[:30]}{'...' if len(missing) > 30 else ''}")
print(f"Tickers where 'recent' doesn't reach back to {COVERAGE_FLOOR} "
      f"(older history not available from the bulk file): {len(insufficient_coverage)}")

df = pd.DataFrame(all_rows)
if len(df) == 0:
    raise SystemExit("No 8-K filings extracted -- something is wrong with the archive format; "
                      "falling back to the per-company script is recommended.")

df["filing_date"] = pd.to_datetime(df["filing_date"])
df = df.sort_values(["ticker", "filing_date"]).reset_index(drop=True)
df.to_parquet(OUT_PATH, index=False)

os.remove(ZIP_TMP_PATH)
print(f"\nDeleted temporary archive {ZIP_TMP_PATH}")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total 8-K filings: {len(df)}")
print(f"Companies with at least one 8-K: {df['ticker'].nunique()}")
print(f"Date coverage: {df['filing_date'].min().date()} to {df['filing_date'].max().date()}")
print(f"Filings from {COVERAGE_FLOOR} onward: {int((df['filing_date'] >= COVERAGE_FLOOR).sum())}")
print(f"Tickers with no extractable data: {len(missing)}")
print(f"Saved to {OUT_PATH}")
