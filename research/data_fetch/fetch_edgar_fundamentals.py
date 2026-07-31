"""
fetch_edgar_fundamentals.py -- Test 10 fundamentals fetch (value + quality
factors). Nine price/event-based strategies have already been rejected
(RESEARCH_LOG.md); this pulls the one data source not yet tried.

Bulk archive checked first, same pattern that worked for 8-Ks
(fetch_edgar_8k_bulk.py): a plain GET to
https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip
returned 200 OK, Content-Length ~1.30 GB (HEAD requests to SEC return
403 here -- SEC's edge appears to reject HEAD specifically -- so the
existence/size check below issues a real GET and reads only the
response headers before deciding whether to stream the body). That's
the same ballpark as submissions.zip, so this uses the same one-bounded-
download approach instead of ~2,340 sequential rate-limited requests
against data.sec.gov/api/xbrl/companyfacts/CIK##########.json.

For each of the 7 target us-gaap concepts, EVERY reported instance is
kept, not just the latest -- the same fiscal period gets reported
repeatedly (original 10-K, 10-K/A amendments, comparative prior-year
columns inside later filings) with different filed dates. Task 2 (the
factor test) needs to reconstruct what was knowable as of each historical
rebalance date, which requires the full history of (end, filed, val)
triples, not a collapsed final value.

CRITICAL -- lookahead protection: 'filed' (when a fact became public) is
captured on every single row alongside 'end' (the fiscal period it
describes). A fact is not usable at any simulated point in time before
its own 'filed' date, no matter how far in the past 'end' is. Enforcing
that boundary is fundamental_test.py's job; this script's job is only to
guarantee both dates exist on every row so enforcement is possible.

Checkpointed every 100 companies processed (parquet re-written to disk
each time) -- an earlier bulk fetch was lost entirely to a crash with no
partial output on disk.
"""
import os
import re
import json
import time
import zipfile
import urllib.request
import urllib.error

import pandas as pd

USER_AGENT = "tradingbot research yassinnil81@gmail.com"
BULK_URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
CIK_MAP_PATH = os.path.join("data", "cik_map.csv")
OUT_PATH = os.path.join("data", "edgar_fundamentals.parquet")
ZIP_TMP_PATH = os.path.join("data", "_companyfacts_bulk_tmp.zip")
CHECKPOINT_EVERY = 100
COVERAGE_FLOOR = "2015-01-01"

CONCEPTS = [
    "Assets", "StockholdersEquity", "Revenues", "CostOfRevenue",
    "GrossProfit", "NetIncomeLoss", "CommonStockSharesOutstanding",
]


def head_check(url):
    """SEC returns 403 on real HEAD requests here; use a GET but stop
    right after reading headers -- close() before the body streams."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        status = resp.status
        length = int(resp.headers.get("Content-Length", 0))
    return status, length


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
                if downloaded - last_report >= 50 * 1024 * 1024:
                    elapsed = time.time() - t0
                    pct = downloaded / total * 100 if total else 0
                    mbps = downloaded / 1024 / 1024 / elapsed if elapsed > 0 else 0
                    print(f"  {downloaded/1024/1024:,.0f} MB / {total/1024/1024:,.0f} MB "
                          f"({pct:.1f}%) -- {mbps:.1f} MB/s", flush=True)
                    last_report = downloaded
    return downloaded


def extract_concept_facts(cik, ticker, company_facts):
    rows = []
    us_gaap = company_facts.get("facts", {}).get("us-gaap", {})
    for concept in CONCEPTS:
        block = us_gaap.get(concept)
        if block is None:
            continue
        for unit, entries in block.get("units", {}).items():
            for e in entries:
                rows.append({
                    "ticker": ticker,
                    "cik": cik,
                    "concept": concept,
                    "unit": unit,
                    "start": e.get("start"),
                    "end": e.get("end"),
                    "val": e.get("val"),
                    "accn": e.get("accn"),
                    "fy": e.get("fy"),
                    "fp": e.get("fp"),
                    "form": e.get("form"),
                    "filed": e.get("filed"),
                })
    return rows


def checkpoint(all_rows, out_path):
    ckpt = pd.DataFrame(all_rows)
    if len(ckpt):
        ckpt["start"] = pd.to_datetime(ckpt["start"], errors="coerce")
        ckpt["end"] = pd.to_datetime(ckpt["end"], errors="coerce")
        ckpt["filed"] = pd.to_datetime(ckpt["filed"], errors="coerce")
        ckpt.to_parquet(out_path, index=False)
    return len(ckpt)


print("=" * 70)
print("STEP 1: check bulk companyfacts archive")
print("=" * 70)
try:
    status, length = head_check(BULK_URL)
    print(f"GET (headers only) {BULK_URL} -> {status}, Content-Length {length/1024/1024:,.0f} MB")
except urllib.error.HTTPError as e:
    raise SystemExit(f"Bulk archive check failed: HTTP {e.code} -- falling back to the per-company "
                      f"API would be required, but that path is not implemented in this script.")

if length > 3 * 1024 * 1024 * 1024:
    raise SystemExit(f"Bulk archive is {length/1024/1024/1024:.1f} GB -- larger than expected, "
                      f"aborting rather than committing to an unbounded download.")

print()
print("=" * 70)
print("STEP 2: download bulk companyfacts archive")
print("=" * 70)
os.makedirs("data", exist_ok=True)
size = download_with_progress(BULK_URL, ZIP_TMP_PATH)
print(f"Download complete: {size/1024/1024:,.0f} MB saved to {ZIP_TMP_PATH}")

print()
print("=" * 70)
print("STEP 3: match CIKs and extract target concepts")
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
matched = []
missing = []
no_target_concepts = []

for i, row in enumerate(cik_map.itertuples(index=False)):
    ticker, cik = row.ticker, row.cik
    name = cik_to_name.get(cik)
    if name is None:
        missing.append(ticker)
    else:
        try:
            data = json.loads(zf.read(name).decode("utf-8"))
        except Exception:
            missing.append(ticker)
            data = None

        if data is not None:
            rows = extract_concept_facts(cik, ticker, data)
            if rows:
                all_rows.extend(rows)
                matched.append(ticker)
            else:
                no_target_concepts.append(ticker)

    if (i + 1) % CHECKPOINT_EVERY == 0:
        n_rows = checkpoint(all_rows, OUT_PATH)
        print(f"  [{i+1}/{len(cik_map)}] checkpoint written -- {n_rows} rows so far, "
              f"{len(matched)} matched, {len(missing)} missing, "
              f"{len(no_target_concepts)} matched-but-no-target-concepts", flush=True)

zf.close()

print()
print(f"Processed all {len(cik_map)} tickers.")
print(f"Matched with at least one target concept: {len(matched)}")
print(f"Missing/unreadable in archive: {len(missing)}")
print(f"Present in archive but none of the 7 target concepts found: {len(no_target_concepts)}")

df = pd.DataFrame(all_rows)
if len(df) == 0:
    raise SystemExit("No fundamental facts extracted -- something is wrong with the archive format.")

df["start"] = pd.to_datetime(df["start"], errors="coerce")
df["end"] = pd.to_datetime(df["end"], errors="coerce")
df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
df = df.dropna(subset=["end", "filed", "val"])
df = df.sort_values(["ticker", "concept", "end", "filed"]).reset_index(drop=True)
df.to_parquet(OUT_PATH, index=False)

os.remove(ZIP_TMP_PATH)
print(f"\nDeleted temporary archive {ZIP_TMP_PATH}")

print()
print("=" * 70)
print("COVERAGE REPORT")
print("=" * 70)
print(f"Total fact-rows: {len(df)}")
print(f"Companies with at least one usable fact: {df['ticker'].nunique()} / {len(cik_map)}")
print(f"Filed-date range: {df['filed'].min().date()} to {df['filed'].max().date()}")
print(f"Fiscal end-date range: {df['end'].min().date()} to {df['end'].max().date()}")
print()
print("Rows per concept:")
for concept in CONCEPTS:
    sub = df[df["concept"] == concept]
    print(f"  {concept:<32} {len(sub):>8} rows, {sub['ticker'].nunique():>5} companies")

print()
before_floor = df[df["filed"] < COVERAGE_FLOOR]
companies_before_floor = before_floor["ticker"].nunique()
print(f"Companies with at least one fact FILED before {COVERAGE_FLOOR} "
      f"(i.e. usable data pre-2015): {companies_before_floor} / {df['ticker'].nunique()}")
print("XBRL tagging was phased in gradually ~2009-2011 and even where present, small/newly "
      "public companies often lack full historical detail -- this number is expected to be "
      "meaningfully lower than total coverage, and is reported as-is rather than backfilled.")

# per-year filed-date histogram, honest accounting of when usable density begins
df["filed_year"] = df["filed"].dt.year
print()
print("Fact rows by filed year:")
for yr, cnt in df["filed_year"].value_counts().sort_index().items():
    print(f"  {yr}: {cnt}")

print(f"\nSaved to {OUT_PATH}")
