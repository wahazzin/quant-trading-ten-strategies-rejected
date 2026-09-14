"""
build_cik_map.py -- rebuilds data/cik_map.csv from SEC's own official
ticker-to-CIK mapping (company_tickers.json), covering essentially every
SEC-registered US company (~10,400 entries) rather than an unexplained
narrow 2,339-row file that was missing AAPL, MSFT, NVDA, AMZN, TSLA, and
GOOG entirely.

Found and fixed while migrating ops/event_monitor.py off Windows Task
Scheduler (2026-09): switching the universe to a live, market-wide
liquidity ranking (real mega-caps included) exposed that the existing
cik_map.csv only matched 42/200 tickers -- its own sample rows are SPAC
shell companies ("Artius II Acquisition Inc.", "Abony Acquisition Corp."),
suggesting it was built for some other, narrower purpose and never
validated against the actual "200 most liquid US stocks" use case. Same
category of stale/orphaned data artifact WINS.md already documented once
(the 46 unverified per-ticker CSVs) -- this is a data-quality fix, not a
change to any pre-registered hypothesis.

Source: https://www.sec.gov/files/company_tickers.json -- SEC's own
canonical, freely available mapping, updated by SEC directly. No API key
needed. Same User-Agent contact convention as the rest of this project's
EDGAR requests.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import urllib.request
import json
import pandas as pd

SOURCE_URL = "https://www.sec.gov/files/company_tickers.json"
OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "data", "cik_map.csv")
USER_AGENT = "tradingbot research your.email@example.com"

if __name__ == "__main__":
    print("Fetching SEC's official company_tickers.json ...")
    req = urllib.request.Request(SOURCE_URL, headers={"User-Agent": USER_AGENT})
    data = json.loads(urllib.request.urlopen(req, timeout=30).read())

    rows = [{"ticker": v["ticker"].upper(), "cik": v["cik_str"], "title": v["title"]}
            for v in data.values()]
    df = pd.DataFrame(rows).drop_duplicates(subset="ticker").sort_values("ticker")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Wrote {len(df)} ticker->CIK mappings to {OUT_PATH}")
    for check in ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "GOOG"]:
        found = check in set(df["ticker"])
        print(f"  {check}: {'OK' if found else 'STILL MISSING -- investigate'}")
