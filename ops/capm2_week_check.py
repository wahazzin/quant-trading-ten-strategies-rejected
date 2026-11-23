"""capm2_week_check.py -- read-only: alert on Discord once if an ISO week passes Tuesday's close with NO CAPM2
rebalance (added 2026-11-23 after week 46 was skipped silently: GitHub started every Mon/Tue slot after the close).

Runs inside capm2_status.yml (after every US close). From Wednesday on, if Monday or Tuesday was a trading day and
no capm2_weekly run this week reached the rebalance step, posts one alert per week (remembered in
status/week_check.json on the capm-status branch). Only GET requests to Alpaca/GitHub; never places orders."""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capm2_should_run import PAPER, done_this_week  # noqa: E402

STATE = os.path.join("status", "week_check.json")


def should_alert(now, mon_or_tue_trading, done, already_alerted_week):
    """done: True/False/None (None = history unreadable -> no alert, avoid false alarms)."""
    week = "%d-W%02d" % now.isocalendar()[:2]
    if now.weekday() < 2 or not mon_or_tue_trading or done is not False or already_alerted_week == week:
        return None
    return week


def main():
    now = datetime.now(timezone.utc)
    h = {"APCA-API-KEY-ID": os.environ["CAPM2_US_KEY_ID"], "APCA-API-SECRET-KEY": os.environ["CAPM2_US_SECRET_KEY"]}
    monday = (now - timedelta(days=now.weekday())).date()
    cal = requests.get(f"{PAPER}/v2/calendar", headers=h, timeout=20,
                       params={"start": monday.isoformat(), "end": (monday + timedelta(days=1)).isoformat()}).json()
    trading = any(d.get("date") in (monday.isoformat(), (monday + timedelta(days=1)).isoformat()) for d in cal)
    try:
        st = json.load(open(STATE))
    except Exception:  # noqa: BLE001
        st = {}
    done = done_this_week(now) if now.weekday() >= 2 and trading else None
    week = should_alert(now, trading, done, st.get("alerted_week"))
    print(f"week check: trading Mon/Tue={trading} rebalanced={done} -> {'ALERT ' + week if week else 'ok'}")
    if not week:
        return
    from urllib.request import Request, urlopen
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    msg = (f"⚠️ CAPM2: no weekly rebalance happened in {week} (every Mon/Tue run started after the close "
           f"or failed). The accounts just keep last week's holdings. Nothing to do unless it repeats; the weekly "
           f"review will look into it.")
    if url:
        try:
            urlopen(Request(url, data=json.dumps({"content": msg}).encode(),
                            headers={"Content-Type": "application/json", "User-Agent": "capm2-week-check"}), timeout=20)
        except Exception as e:  # noqa: BLE001
            print(f"Discord alert failed ({type(e).__name__})"); return
    else:
        print("DISCORD_WEBHOOK_URL not set -- would have sent:", msg)
    os.makedirs("status", exist_ok=True)
    json.dump({"alerted_week": week, "at": now.isoformat()}, open(STATE, "w"))


if __name__ == "__main__":
    main()
