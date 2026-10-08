"""capm2_should_run.py -- read-only gate for the weekly rebalance (added 2026-10-08).
Runs Mondays AND Tuesdays 15:00 UTC (market open all year, both US summer and winter time). Prints RUN only if:
  - the US market is open right now, and
  - today is Monday, or today is Tuesday and Monday was NOT a trading day (holiday Monday).
Otherwise prints SKIP. Only GET requests (clock, calendar)."""
import os
import sys
from datetime import datetime, timedelta, timezone

import requests

PAPER = "https://paper-api.alpaca.markets"


def decide(now, market_open, monday_was_trading_day):
    wd = now.weekday()
    if not market_open:
        return "SKIP (market closed)"
    if wd == 0:
        return "RUN"
    if wd == 1 and not monday_was_trading_day:
        return "RUN (Monday was a market holiday)"
    return "SKIP"


if __name__ == "__main__":
    h = {"APCA-API-KEY-ID": os.environ["CAPM2_US_KEY_ID"], "APCA-API-SECRET-KEY": os.environ["CAPM2_US_SECRET_KEY"]}
    now = datetime.now(timezone.utc)
    if os.environ.get("FORCE_RUN") == "1":
        print("RUN (manual)"); sys.exit(0)
    clock = requests.get(f"{PAPER}/v2/clock", headers=h, timeout=20).json()
    monday = (now - timedelta(days=now.weekday())).date().isoformat()
    cal = requests.get(f"{PAPER}/v2/calendar", headers=h, params={"start": monday, "end": monday}, timeout=20).json()
    print(decide(now, clock.get("is_open", False), any(d.get("date") == monday for d in cal)))
