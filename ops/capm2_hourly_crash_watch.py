"""
capm2_hourly_crash_watch.py -- Test 20 / Phase 6d's fast tier: checks
every CAPM2 name for an unusually large single-day drop AND a negative
shadow-sentiment reading, together. If both fire, that name is cut to
zero IMMEDIATELY -- not held until the next weekly reallocation.

Meant to be scheduled externally (Windows Task Scheduler / cron) once
per hour during market hours -- same "run once, schedule outside the
script" convention as ops/event_monitor.py and ops/capm_monitor.py, not
an internal loop. Market-hours-only is enforced INSIDE this script via
Alpaca's own clock (client.is_market_open()), so a naive hourly
scheduler that also fires outside market hours still no-ops safely.

THRESHOLDS ARE PLACEHOLDERS, not this script's invention -- they are the
exact same numbers already pre-registered as defaults inside
bot/strategy/capm2_allocation.py's check_crash() (-8% drop, -0.3
sentiment), per RESEARCH_LOG.md's Test 20 spec: "exact drop/sentiment
thresholds are a TODO before implementation, to be set once and not
tuned after seeing results." This script does not introduce new numbers
-- it reuses the ones already committed to code, and does not change
them based on anything observed here.

Same no-journal design as capm2_weekly_rebalance.py, same reason (see
that file's docstring): broker-only reads, no bot/journal/db.py writes,
Discord notification fires independently of any DB.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
from datetime import datetime, timedelta, timezone

from bot.broker.alpaca_client import AlpacaClient
from bot.broker.reconcile import reconcile, execute_plan
from bot.broker.guard import require_broker
from bot.strategy.capm2_allocation import check_crash
from bot.strategy.capm2_universe import GROUPS
from bot.monitor.sentiment_veto import SentimentVeto
from bot.monitor.discord_notify import notify_crash_cut

require_broker("alpaca")

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true",
                     help="Print what WOULD be cut without submitting sell orders.")
args = parser.parse_args()

print("=" * 96)
print("CAPM2 HOURLY CRASH WATCH (Test 20 / Phase 6d)")
print("=" * 96)
if args.dry_run:
    print("*** DRY RUN -- no orders will be submitted ***")


def todays_return(client, ticker):
    """(current price / most recent prior close) - 1 -- an intraday read
    of today's move so far, checked hourly, not a close-to-close return.
    Returns None if no prior close is available (new listing, data gap)."""
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=5)  # small buffer for weekends/holidays
    bars = client.get_daily_bars(ticker, start, end)
    if not bars:
        return None
    prev_close = bars[-1]["c"]
    try:
        current = client.get_last_price(ticker)
    except Exception as e:
        print(f"    {ticker}: could not fetch live price ({e})")
        return None
    return (current / prev_close) - 1


sentiment_veto = SentimentVeto(lookback_days=1)
empty_price_df = pd.DataFrame(columns=["date"])


def place_crash_sell(client, symbol, qty):
    """Immediate-priority sell (tif='day', not 'gtc') -- a crash cut is
    urgent by definition; it should not still be resting as a GTC order
    days later the way the weekly rebalance's orders are allowed to."""
    resp = client.place_market_order(symbol, "sell", qty, tif="day")
    if resp.get("error"):
        print(f"    REJECTED sell {qty} {symbol}: {resp['message']}")
        return {"status": "rejected", "error": resp["message"]}
    print(f"    Submitted SELL {qty} {symbol} (order {resp['id']})")
    return {"status": "submitted", "order_id": resp["id"]}


any_cuts = False
for key, spec in GROUPS.items():
    print()
    print(f"-- {spec['label']} --")

    key_id = os.environ.get(spec["key_env"])
    secret = os.environ.get(spec["secret_env"])
    if not key_id or not secret:
        print(f"  Missing {spec['key_env']}/{spec['secret_env']} in .env -- skipping this group.")
        continue

    client = AlpacaClient(paper=True, api_key=key_id, api_secret=secret)
    if not client.connect():
        print("  Could not reach this account -- skipping this run.")
        continue

    if not client.is_market_open().get("is_open"):
        print("  Market closed -- nothing to check this cycle.")
        client.disconnect()
        continue

    current_positions = client.get_positions()
    for t in spec["tickers"]:
        held = current_positions.get(t, 0)
        if held == 0:
            continue  # nothing to cut, per-name check skipped once flat

        daily_return = todays_return(client, t)
        if daily_return is None:
            print(f"  {t}: no price data this cycle -- skipping")
            continue
        sentiment_score = sentiment_veto.score(t, empty_price_df)
        triggered = check_crash(daily_return, sentiment_score)
        print(f"  {t}: today {daily_return:+.2%}, sentiment "
              f"{'n/a' if sentiment_score is None else f'{sentiment_score:+.2f}'} "
              f"-> {'CUT TO ZERO' if triggered else 'no trigger'}")

        if not triggered:
            continue

        open_orders_raw = client.get_open_orders_raw(symbol=t)
        diffs, warnings = reconcile({t}, {t: 0}, current_positions, open_orders_raw)
        for w in warnings:
            print(f"    WARNING: {w}")
        if t not in diffs:
            continue

        results = execute_plan(diffs, lambda sym, action, qty: place_crash_sell(client, sym, qty),
                                dry_run=args.dry_run)
        if not args.dry_run and results:
            _, _, qty, res = results[0]
            if res and res.get("status") == "submitted":
                notify_crash_cut(t, qty, daily_return * 100, sentiment_score)
                any_cuts = True

    client.disconnect()

print()
print("=" * 96)
print("DONE" + ("  (dry run)" if args.dry_run else "") +
      ("  -- no crash triggers this cycle" if not any_cuts and not args.dry_run else ""))
print("=" * 96)
