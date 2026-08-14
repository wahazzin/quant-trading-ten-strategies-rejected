"""
forward_status.py -- single-command status across all three live forward
tests running in this project (RESEARCH_LOG.md, Phase 6/6b/6c):

  1. Value portfolio (20-stock Book-to-Market, Alpaca) -- runs ops/value_report.py
  2. 8-K liquid-name event monitor -- summarizes data/event_forward_log.parquet
     directly (does NOT re-run ops/event_monitor.py: that script's SEC EDGAR
     fetch takes 1-3 minutes and runs on its own daily cadence, separate
     from a quick status check)
  3. CAPM accounts (US + Swedish, no rebalancing) -- runs ops/capm_monitor.py

value_report.py and capm_monitor.py are run as subprocesses rather than
imported -- both are flat, side-effecting scripts (they write their own
state/log files as a normal part of running), so running them exactly as
they'd run standalone is more honest than re-implementing their logic here
and risking it drifting out of sync.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
import pandas as pd

EVENT_LOG_PATH = os.path.join("data", "event_forward_log.parquet")
HORIZONS = [1, 5, 10, 20]


def run_script(rel_path, title):
    print()
    print("#" * 96)
    print(f"# {title}")
    print("#" * 96)
    script_path = os.path.join(os.path.dirname(__file__), rel_path)
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"[{rel_path} exited with code {result.returncode}]")
        if result.stderr:
            print("--- stderr ---")
            print(result.stderr)


def summarize_event_log():
    print()
    print("#" * 96)
    print("# FORWARD TEST 2 -- 8-K LIQUID-NAME EVENT MONITOR (summary of data/event_forward_log.parquet)")
    print("#" * 96)
    if not os.path.exists(EVENT_LOG_PATH):
        print("No data yet -- ops/event_monitor.py has not logged any filings.")
        return

    df = pd.read_parquet(EVENT_LOG_PATH)
    if df.empty:
        print("Log exists but is empty.")
        return

    fwd_cols = [f"fwd_ret_{h}d" for h in HORIZONS]
    complete = df[fwd_cols].notna().all(axis=1).sum()
    print(f"Total filings tracked: {len(df)}")
    print(f"Date range: {df['filing_date'].min()} to {df['filing_date'].max()}")
    print(f"Filings with all 4 forward-return horizons filled in: {complete}")
    print(f"Filings still awaiting one or more horizons: {len(df) - complete}")
    print("\nNOTE: the pre-registered success criterion (RESEARCH_LOG.md, Phase 6b) requires "
          "at least 100 DECLUSTERED events (first 8-K per ticker per 10-trading-day window) -- "
          "the raw count above is not that number. Run the declustering analysis separately "
          "once enough raw filings have accumulated to make it worth checking.")

    print("\nMost recent 5 filings:")
    recent = df.sort_values("filing_date", ascending=False).head(5)
    print(recent[["filing_date", "ticker", "item_codes", "dollar_volume_rank"]].to_string(index=False))


print("=" * 96)
print("FORWARD TEST STATUS -- all three live, pre-registered forward tests")
print("=" * 96)

run_script("value_report.py", "FORWARD TEST 1 -- VALUE PORTFOLIO (Book-to-Market, 20 stocks, Alpaca)")
summarize_event_log()
run_script("capm_monitor.py", "FORWARD TEST 3 -- CAPM ACCOUNTS (US + Swedish, no rebalancing)")

print()
print("=" * 96)
print("Done. See RESEARCH_LOG.md Phase 6 / 6b / 6c for each test's pre-registration and success criterion.")
print("=" * 96)
