"""
ff_calendar_monitor.py -- Phase 6e: log-only ForexFactory economic
calendar scrape. See RESEARCH_LOG.md's Phase 6e section for the full
pre-registration -- written BEFORE this script existed, not re-derived
here.

LOG ONLY. This script places no orders, touches no broker account, and
feeds no live trading logic anywhere in this project. It exists purely
to accumulate clean data before any hypothesis about it is proposed.

Data source: forexfactory.com/calendar -- public, server-rendered HTML,
no login, no official API. robots.txt sets no Disallow rules for any
path. Kept deliberately low-frequency (meant to run once per day, not
on every developer test run) and identifies itself honestly via
User-Agent rather than spoofing a browser.

Each event on the page carries its own data-event-id (stable across
days) and data-day-dateline (a Unix timestamp for that event's date).
Re-running this against the same event_id UPDATES the stored row rather
than duplicating it -- actual/forecast values fill in over time as a
release approaches and then happens, so the log always reflects the
latest known state for that event, not a stale first-look snapshot.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import pandas as pd
from datetime import datetime, timezone
from bs4 import BeautifulSoup

CALENDAR_URL = "https://www.forexfactory.com/calendar"
LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "ff_calendar_log.csv")
USER_AGENT = "quant-trading-research-bot/1.0 (personal research project, log-only, no trading; contact via github.com/wahazzin)"

IMPACT_MAP = {
    "red": "high",
    "ora": "medium",
    "yel": "low",
    "gra": "non-economic",
}


def fetch_calendar_html():
    resp = requests.get(CALENDAR_URL, headers={"User-Agent": USER_AGENT}, timeout=20)
    resp.raise_for_status()
    return resp.text


def parse_events(html):
    """IMPORTANT: only the FIRST event row of each calendar day carries
    data-day-dateline -- every subsequent same-day row omits it and
    must inherit the date from the most recent row that had one. This
    was found by inspecting the real page structure directly (not
    assumed) after an early version of this parser silently dropped
    75 of 81 real events by requiring every row to carry its own
    dateline."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="calendar__table")
    all_rows = table.find_all("tr") if table else soup.find_all("tr")

    events = []
    current_date = None
    last_time = ""  # FF leaves the time cell blank on consecutive same-time
                     # events within a day -- carry the last seen value
                     # forward rather than recording a false blank

    for row in all_rows:
        dateline = row.get("data-day-dateline")
        if dateline:
            current_date = datetime.fromtimestamp(int(dateline), tz=timezone.utc).date().isoformat()

        event_id = row.get("data-event-id")
        if not event_id or not current_date:
            continue

        time_cell = row.find("td", class_="calendar__time")
        time_text = time_cell.get_text(strip=True) if time_cell else ""
        if time_text:
            last_time = time_text

        currency_cell = row.find("td", class_="calendar__currency")
        currency = currency_cell.get_text(strip=True) if currency_cell else ""

        impact_cell = row.find("td", class_="calendar__impact")
        impact = "unknown"
        if impact_cell:
            icon = impact_cell.find("span", class_=lambda c: c and c.startswith("icon--ff-impact-"))
            if icon:
                for cls in icon.get("class", []):
                    if cls.startswith("icon--ff-impact-"):
                        suffix = cls.replace("icon--ff-impact-", "")
                        impact = IMPACT_MAP.get(suffix, "unknown")

        title_cell = row.find("td", class_="calendar__event")
        title = title_cell.get_text(strip=True) if title_cell else ""

        actual_cell = row.find("td", class_="calendar__actual")
        actual = actual_cell.get_text(strip=True) if actual_cell else ""

        forecast_cell = row.find("td", class_="calendar__forecast")
        forecast = forecast_cell.get_text(strip=True) if forecast_cell else ""

        previous_cell = row.find("td", class_="calendar__previous")
        previous = previous_cell.get_text(strip=True) if previous_cell else ""

        if not title:
            continue  # skip malformed/empty rows rather than log garbage

        events.append({
            "event_id": event_id,
            "date": current_date,
            "time": last_time,
            "currency": currency,
            "impact": impact,
            "title": title,
            "actual": actual,
            "forecast": forecast,
            "previous": previous,
            "scraped_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })

    return events


def upsert_log(new_events):
    """Update-or-insert by event_id, never duplicate. New scrape data
    always overwrites the stored row for that event_id -- this is
    intentional (actual/forecast fill in as a release approaches and
    happens), not a bug: the log always holds the latest known state,
    not a frozen first-look snapshot."""
    new_df = pd.DataFrame(new_events)
    if new_df.empty:
        return 0, 0

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    if os.path.exists(LOG_PATH):
        existing_df = pd.read_csv(LOG_PATH, dtype={"event_id": str})
        new_df["event_id"] = new_df["event_id"].astype(str)
        existing_ids = set(existing_df["event_id"])
        new_ids = set(new_df["event_id"])
        updated_count = len(existing_ids & new_ids)
        inserted_count = len(new_ids - existing_ids)

        combined = pd.concat([existing_df, new_df])
        combined = combined.drop_duplicates(subset="event_id", keep="last")
    else:
        combined = new_df
        updated_count = 0
        inserted_count = len(new_df)

    combined = combined.sort_values(["date", "time"], na_position="last")
    combined.to_csv(LOG_PATH, index=False)
    return inserted_count, updated_count


if __name__ == "__main__":
    print("=" * 96)
    print("FOREXFACTORY CALENDAR MONITOR (Phase 6e -- LOG ONLY, no trading)")
    print("=" * 96)

    html = fetch_calendar_html()
    events = parse_events(html)
    print(f"Parsed {len(events)} events from this scrape.")

    if events:
        by_impact = pd.Series([e["impact"] for e in events]).value_counts()
        for level, count in by_impact.items():
            print(f"  {level}: {count}")

    inserted, updated = upsert_log(events)
    print(f"\nLog: {LOG_PATH}")
    print(f"  New events added: {inserted}")
    print(f"  Existing events updated (actual/forecast filled in etc.): {updated}")
    print("=" * 96)
    print("DONE -- log-only, no orders placed, no live logic touched")
    print("=" * 96)
