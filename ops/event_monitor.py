"""
event_monitor.py -- Phase 6 forward test #2: a pre-registered live
monitor for the 8-K liquid-name effect found in
research/event_tests/event_study_v3.py's re-examination of Test 9
(declustered, liquid half, item 5.02 and all-items pooled, cleared the
real ~0.009% Alpaca cost by 10-100x at every horizon). That result
cannot be validated on owned data -- 2019-2026 is fully spent and
sealed, and re-testing it would just be curve-fitting on the same
sample. Forward testing on data that does not exist yet is the only
path left, exactly like ops/value_rebalance.py is for Test 8/15's
practice-window survivors.

LOG ONLY. This script places no orders and holds no positions -- it
only records: which of the 200 most liquid US stocks filed an 8-K
today, what item codes it carried, its trailing liquidity rank at
filing time, and (once enough calendar time has passed) its 1/5/10/20
day forward return. See RESEARCH_LOG.md's pre-registration entry for
this test -- written BEFORE this log holds a single row -- for the
exact prediction and success criterion this accumulates evidence
against.

Universe: the 200 most liquid US stocks by trailing 60-day median
dollar volume ($ = close x volume), rebuilt once per calendar month
from data/yf_universe.parquet (a static historical snapshot -- used
for ranking only, never for forward returns) and cached to
data/event_monitor_universe.csv. The universe is NOT re-picked daily,
so a stock can't be selectively included or excluded based on how an
individual filing turns out.

Filings: fetched from SEC EDGAR's free per-company submissions API
(data.sec.gov/submissions/CIK##########.json), same endpoint and
User-Agent convention as research/data_fetch/fetch_edgar_8k.py, using
data/cik_map.csv for ticker -> CIK. Only the "recent" filings block is
fetched (no historical pagination) -- this is a live monitor, not a
backfill.

Forward returns: computed from Alpaca's daily-bars market data API
(bot/broker/alpaca_client.py's get_daily_bars), NOT
data/yf_universe.parquet, since that dataset stops in mid-2026 and is
never updated going forward. Entry = the first trading day's OPEN
after the filing date (matching event_study_v2.py / v3.py's
convention, for comparability); horizon return = open[entry+h] /
open[entry] - 1.

Pre-registration guardrail: only filings dated on/after this script's
FIRST ever run (data/event_monitor_state.json's start_date) are ever
logged -- a company's older 8-Ks sitting in the same API response are
not backfilled into what is supposed to be a forward-only record.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from bot.broker.alpaca_client import AlpacaClient

UNIVERSE_SIZE = 200
DOLLAR_VOLUME_LOOKBACK_DAYS = 60
MIN_LOOKBACK_ROWS = 40  # skip thinly-covered tickers when ranking

PRICE_PATH = os.path.join("data", "yf_universe.parquet")
CIK_MAP_PATH = os.path.join("data", "cik_map.csv")
UNIVERSE_CACHE_PATH = os.path.join("data", "event_monitor_universe.csv")
STATE_PATH = os.path.join("data", "event_monitor_state.json")
LOG_PATH = os.path.join("data", "event_forward_log.parquet")

USER_AGENT = "tradingbot research your.email@example.com"  # same contact string EDGAR requires, matching fetch_edgar_8k.py
SLEEP_BETWEEN_REQUESTS = 0.11  # ~9 req/s -- same safety margin as fetch_edgar_8k.py

HORIZONS = [1, 5, 10, 20]
FWD_RET_COLS = [f"fwd_ret_{h}d" for h in HORIZONS]
LOG_COLUMNS = ["ticker", "filing_date", "item_codes", "accession", "dollar_volume_rank",
               "discovered_date", "entry_date"] + FWD_RET_COLS

print("=" * 96)
print("EVENT MONITOR -- 8-K liquid-name forward test")
print("LOG ONLY -- this script places no orders and holds no positions")
print("=" * 96)

# ============================================================
# PRE-REGISTRATION GUARDRAIL: the start date is fixed on first ever run
# and never moves. Only filings on/after it are logged, no matter what
# a company's API response happens to include.
# ============================================================
if os.path.exists(STATE_PATH):
    with open(STATE_PATH) as f:
        monitor_state = json.load(f)
else:
    monitor_state = {"start_date": datetime.now(timezone.utc).date().isoformat()}
    with open(STATE_PATH, "w") as f:
        json.dump(monitor_state, f, indent=2)
    print(f"FIRST RUN -- forward-test start date fixed at {monitor_state['start_date']}. "
          f"This must match RESEARCH_LOG.md's pre-registration entry.")

start_date = monitor_state["start_date"]
print(f"Forward-test start date (fixed, never moves): {start_date}")


# ============================================================
# UNIVERSE: 200 most liquid names, rebuilt once per calendar month
# ============================================================
this_month = pd.Timestamp.now().strftime("%Y-%m")


def build_universe():
    price = pd.read_parquet(PRICE_PATH)
    price["date"] = pd.to_datetime(price["date"])
    as_of = price["date"].max()
    rows = []
    for ticker, g in price.groupby("ticker"):
        g = g.sort_values("date").tail(DOLLAR_VOLUME_LOOKBACK_DAYS)
        if len(g) < MIN_LOOKBACK_ROWS:
            continue
        dv = float((g["close"] * g["volume"]).median())
        rows.append({"ticker": ticker, "median_dollar_volume": dv})
    ranked = pd.DataFrame(rows).sort_values("median_dollar_volume", ascending=False).reset_index(drop=True)
    ranked = ranked.head(UNIVERSE_SIZE).copy()
    ranked["dollar_volume_rank"] = np.arange(1, len(ranked) + 1)
    ranked["as_of_date"] = as_of.date().isoformat()
    ranked["computed_month"] = this_month
    ranked.to_csv(UNIVERSE_CACHE_PATH, index=False)
    return ranked


if os.path.exists(UNIVERSE_CACHE_PATH):
    universe = pd.read_csv(UNIVERSE_CACHE_PATH)
    if str(universe["computed_month"].iloc[0]) != this_month:
        print(f"Universe cache is from {universe['computed_month'].iloc[0]}, rebuilding for {this_month}...")
        universe = build_universe()
    else:
        print(f"Using cached universe (computed {universe['computed_month'].iloc[0]}, "
              f"as of {universe['as_of_date'].iloc[0]}).")
else:
    print("No universe cache found -- building for the first time...")
    universe = build_universe()

print(f"Universe: {len(universe)} tickers (top {UNIVERSE_SIZE} by trailing "
      f"{DOLLAR_VOLUME_LOOKBACK_DAYS}-day median $ volume, as of {universe['as_of_date'].iloc[0]})")

rank_by_ticker = dict(zip(universe["ticker"], universe["dollar_volume_rank"]))


# ============================================================
# CIK MAPPING
# ============================================================
cik_map = pd.read_csv(CIK_MAP_PATH, dtype={"cik": int})
cik_map = cik_map[cik_map["ticker"].isin(rank_by_ticker)].drop_duplicates("ticker")
matched = set(cik_map["ticker"])
unmatched = sorted(set(rank_by_ticker) - matched)
print(f"CIK matched: {len(matched)} / {len(rank_by_ticker)} "
      f"({len(unmatched)} unmatched -- not SEC-registered under this ticker, or a foreign filer using 6-K)")


# ============================================================
# EXISTING LOG
# ============================================================
if os.path.exists(LOG_PATH):
    log_df = pd.read_parquet(LOG_PATH)
else:
    log_df = pd.DataFrame(columns=LOG_COLUMNS)

known_accessions = set(log_df["accession"]) if len(log_df) else set()


# ============================================================
# FETCH NEW 8-Ks (recent filings block only -- this is a live monitor)
# ============================================================
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


new_rows = []
fetch_errors = []
today_str = datetime.now(timezone.utc).date().isoformat()

print()
print("Checking universe for new 8-K filings...")
for i, row in enumerate(cik_map.itertuples(index=False)):
    ticker, cik = row.ticker, row.cik
    data = fetch_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    time.sleep(SLEEP_BETWEEN_REQUESTS)
    if data is None:
        fetch_errors.append(ticker)
        continue

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    items = recent.get("items", [])
    for j, form in enumerate(forms):
        if form != "8-K":
            continue
        filing_date = dates[j]
        accession = accessions[j]
        if filing_date < start_date or accession in known_accessions:
            continue
        new_rows.append({
            "ticker": ticker,
            "filing_date": filing_date,
            "item_codes": items[j] if j < len(items) else "",
            "accession": accession,
            "dollar_volume_rank": int(rank_by_ticker[ticker]),
            "discovered_date": today_str,
            "entry_date": None,
            "fwd_ret_1d": np.nan, "fwd_ret_5d": np.nan,
            "fwd_ret_10d": np.nan, "fwd_ret_20d": np.nan,
        })
        known_accessions.add(accession)

    if (i + 1) % 50 == 0:
        print(f"  [{i + 1}/{len(cik_map)}] checked -- {len(new_rows)} new filing(s) so far, "
              f"{len(fetch_errors)} fetch error(s)")

print(f"\nNew 8-K filings found this run: {len(new_rows)}")
if fetch_errors:
    print(f"Fetch errors (skipped, will retry next run): {len(fetch_errors)} -- "
          f"{fetch_errors[:10]}{'...' if len(fetch_errors) > 10 else ''}")

if new_rows:
    log_df = pd.concat([log_df, pd.DataFrame(new_rows)], ignore_index=True)

for col in FWD_RET_COLS:
    log_df[col] = pd.to_numeric(log_df[col], errors="coerce") if len(log_df) else pd.Series(dtype="float64")


# ============================================================
# BACKFILL FORWARD RETURNS as those dates arrive
# ============================================================
client = AlpacaClient(paper=True)
client.connect()

incomplete_mask = log_df[FWD_RET_COLS].isna().any(axis=1) if len(log_df) else pd.Series(dtype=bool)
incomplete = log_df[incomplete_mask]
print(f"\nRows with at least one pending forward-return horizon: {len(incomplete)}")

updated = 0
bar_errors = []
for idx in incomplete.index:
    ticker = log_df.at[idx, "ticker"]
    filing_date = pd.Timestamp(log_df.at[idx, "filing_date"]).normalize()
    end = min(pd.Timestamp.now().normalize(), filing_date + pd.Timedelta(days=45))
    try:
        bars = client.get_daily_bars(ticker, filing_date.date().isoformat(),
                                      (end + pd.Timedelta(days=1)).date().isoformat())
    except Exception:
        bar_errors.append(ticker)
        continue
    if not bars:
        continue

    bar_dates = [pd.Timestamp(b["t"]).tz_localize(None).normalize() for b in bars]
    opens = [b["o"] for b in bars]

    entry_idx = next((k for k, d in enumerate(bar_dates) if d > filing_date), None)
    if entry_idx is None:
        continue
    if pd.isna(log_df.at[idx, "entry_date"]):
        log_df.at[idx, "entry_date"] = bar_dates[entry_idx].date().isoformat()

    for h in HORIZONS:
        col = f"fwd_ret_{h}d"
        if pd.notna(log_df.at[idx, col]):
            continue
        target_idx = entry_idx + h
        if target_idx < len(bar_dates):
            log_df.at[idx, col] = opens[target_idx] / opens[entry_idx] - 1
            updated += 1

print(f"Forward-return values newly filled in this run: {updated}")
if bar_errors:
    print(f"Bar-fetch errors (skipped, will retry next run): {len(bar_errors)} -- {bar_errors[:10]}")

os.makedirs("data", exist_ok=True)
log_df.to_parquet(LOG_PATH, index=False)


# ============================================================
# SUMMARY
# ============================================================
print()
print("=" * 96)
print("SUMMARY")
print("=" * 96)
print("LOG ONLY -- no orders were placed, no positions are held by this script.")
print(f"Total filings tracked since {start_date}: {len(log_df)}")
if len(log_df):
    complete = log_df[FWD_RET_COLS].notna().all(axis=1).sum()
    print(f"Filings with all 4 forward-return horizons filled in: {complete}")
    print(f"Filings still awaiting one or more horizons: {len(log_df) - complete}")
if new_rows:
    print("\nNew filings discovered today:")
    for r in new_rows:
        print(f"  {r['filing_date']}  {r['ticker']:<6}  items={r['item_codes']:<20}  "
              f"liquidity rank {r['dollar_volume_rank']}")
print(f"\nLog saved to {LOG_PATH}")
