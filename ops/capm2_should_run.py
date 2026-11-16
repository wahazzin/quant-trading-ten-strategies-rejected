"""capm2_should_run.py -- read-only gate for the weekly rebalance (added 2026-10-08, hardened 2026-10-12).
Scheduled Mondays AND Tuesdays at several slots 14:47-18:53 UTC (see capm2_weekly.yml; market open all year).
Prints RUN only if:
  - the US market is open right now, and
  - CAPM2 has not already rebalanced this ISO week (checked in this workflow's own run history), and
  - today is Monday, or today is Tuesday (Monday was a market holiday, OR Monday's run never got to rebalance,
    e.g. because GitHub started the scheduled job hours late, after the close -- seen on 2026-10-05: 7h20m late).
Otherwise prints SKIP. Only GET requests (Alpaca clock/calendar, GitHub Actions run history). Never orders.
If the run history cannot be read, Tuesday falls back to the original rule (only after a holiday Monday), so a
failed lookup can never cause a second rebalance in the same week.

Early slots (added 2026-11-16): if the job starts BEFORE today's open (Mon/Tue only) and the open is at most
MAX_WAIT_S away, the gate sleeps until a few minutes after the open and then decides as above. Reason: in W46
GitHub started all ten Mon/Tue slots (14:47-18:53 UTC) 5-9 h late, i.e. after the close, and the week was skipped.
A slot scheduled in the morning that starts on time simply waits; one that starts hours late lands inside market
hours. Same day, same rules -- only the time of day the job starts changes."""
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import requests

PAPER = "https://paper-api.alpaca.markets"
STEP = "Run weekly reallocation"


def decide(now, market_open, monday_was_trading_day, done_this_week=False):
    """done_this_week: True = already rebalanced, False = not yet, None = unknown (lookup failed)."""
    wd = now.weekday()
    if not market_open:
        return "SKIP (market closed)"
    if done_this_week:
        return "SKIP (already rebalanced this week)"
    if wd == 0 and done_this_week is None and now.hour >= 17:
        return "SKIP (run history unreadable; only the first Monday slot may run blind)"
    if wd == 0:
        return "RUN"
    if wd == 1 and not monday_was_trading_day:
        return "RUN (Monday was a market holiday)"
    if wd == 1 and done_this_week is False:
        return "RUN (Monday's run never rebalanced -- late/missed schedule)"
    return "SKIP"


MAX_WAIT_S = 5 * 3600 + 15 * 60   # job timeout is 350 min; keep room for install + rebalance
OPEN_BUFFER_S = 10 * 60           # start 10 min after the open, not in the opening auction


def wait_seconds(now, is_open, next_open):
    """Seconds to sleep before deciding, or 0. Only on Mon/Tue, only when the market is closed now and opens later
    the SAME UTC day within MAX_WAIT_S. now/next_open: aware datetimes."""
    if is_open or next_open is None or now.weekday() not in (0, 1):
        return 0
    if next_open.astimezone(timezone.utc).date() != now.astimezone(timezone.utc).date():
        return 0
    s = (next_open - now).total_seconds() + OPEN_BUFFER_S
    return int(s) if 0 < s <= MAX_WAIT_S else 0


def rebalance_attempted(runs_jobs, monday_start, current_run_id):
    """True if any OTHER run of this workflow since Monday 00:00 UTC got to the rebalance step
    (success OR failure both count as 'attempted' -- a failed run is never retried automatically,
    to avoid doubling orders). runs_jobs: list of (run_id, created_at_iso, [steps dicts])."""
    for run_id, created, steps in runs_jobs:
        if run_id == current_run_id:
            continue
        if datetime.fromisoformat(created.replace("Z", "+00:00")) < monday_start:
            continue
        for s in steps:
            if s.get("name") == STEP and s.get("conclusion") in ("success", "failure", "cancelled"):
                return True
    return False


def done_this_week(now):
    repo, token, run_id = os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_RUN_ID")
    if not (repo and token):
        return None
    try:
        gh = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
        monday_start = datetime.combine((now - timedelta(days=now.weekday())).date(), datetime.min.time(), timezone.utc)
        runs = requests.get(f"https://api.github.com/repos/{repo}/actions/workflows/capm2_weekly.yml/runs", headers=gh,
                            params={"created": f">={monday_start.date().isoformat()}", "per_page": 50}, timeout=20)
        runs.raise_for_status()
        rj = []
        for r in runs.json().get("workflow_runs", []):
            jobs = requests.get(r["jobs_url"], headers=gh, timeout=20)
            jobs.raise_for_status()
            steps = [s for j in jobs.json().get("jobs", []) for s in j.get("steps", [])]
            rj.append((str(r["id"]), r["created_at"], steps))
        return rebalance_attempted(rj, monday_start, str(run_id))
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: could not read run history ({type(e).__name__}); using the holiday-only Tuesday rule")
        return None


if __name__ == "__main__":
    h = {"APCA-API-KEY-ID": os.environ["CAPM2_US_KEY_ID"], "APCA-API-SECRET-KEY": os.environ["CAPM2_US_SECRET_KEY"]}
    now = datetime.now(timezone.utc)
    if os.environ.get("FORCE_RUN") == "1":
        print("RUN (manual)"); sys.exit(0)
    clock = requests.get(f"{PAPER}/v2/clock", headers=h, timeout=20).json()
    nxt = clock.get("next_open")
    w = wait_seconds(now, clock.get("is_open", False), datetime.fromisoformat(nxt.replace("Z", "+00:00")) if nxt else None)
    if w:
        print(f"Market opens later today; waiting {w // 60} min (started early, see docstring)", flush=True)
        time.sleep(w)
        now = datetime.now(timezone.utc)
        clock = requests.get(f"{PAPER}/v2/clock", headers=h, timeout=20).json()
    monday = (now - timedelta(days=now.weekday())).date().isoformat()
    cal = requests.get(f"{PAPER}/v2/calendar", headers=h, params={"start": monday, "end": monday}, timeout=20).json()
    is_open = clock.get("is_open", False)
    print(decide(now, is_open, any(d.get("date") == monday for d in cal), done_this_week(now) if is_open else None))
