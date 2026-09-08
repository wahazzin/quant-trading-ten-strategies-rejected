"""
capm2_weekly_rebalance.py -- Test 20 / Phase 6d: performance-weighted
reallocation across the same 11-name CAPM universe as the static Phase 6c
accounts (RESEARCH_LOG.md has the full pre-registration -- read it there,
not re-derived here).

DESIGN DECISION, made before writing this: this script does NOT use
bot/journal/db.py's TradeJournal. OpenPosition.symbol is globally UNIQUE
across the entire trading.db, with no per-account/per-strategy scoping --
if any CAPM2 ticker ever collides with a ticker the value portfolio (a
DIFFERENT Alpaca account) also holds, journaling both through the same
table would silently blend two unrelated accounts' positions into one
row. That is the exact shape of incidents #3/#4 in RESEARCH_LOG.md's
Phase 6 section, one level up. ops/capm_monitor.py already sidesteps this
by reading P&L straight from Alpaca's own position objects instead of
the journal -- this script does the same. Reconciliation and sizing use
ONLY live broker reads (get_positions, get_open_orders_raw); Discord
fill notifications fire independently and need no DB.

Weight -> target shares: weight * (this account's own net liquidation),
per group. Each CAPM2 account is fully dedicated to its group (US or SE)
-- there is no separate "how much of the account to deploy" question
the way value_rebalance.py had (that account holds several unrelated
strategies; this one holds exactly one).

Trailing-return signal: 3 calendar months of daily bars, last close vs
first close in the window. Missing/insufficient data for ANY ticker in a
group aborts THAT GROUP's cycle entirely (prints an error, trades
nothing) rather than silently excluding the ticker -- a partial group
would corrupt the z-score for every other name in it too (the mean/std
compute_weights() would use are already wrong the moment one input is
missing), and CAPM2's own accounts are new enough not to have a stale-
data workaround built yet.

--dry-run prints the reconciled plan for both groups without submitting
anything.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import argparse
import pandas as pd
from datetime import datetime, timezone

from bot.broker.alpaca_client import AlpacaClient
from bot.broker.reconcile import reconcile, execute_plan
from bot.broker.guard import require_broker
from bot.strategy.capm2_allocation import compute_weights
from bot.strategy.capm2_universe import GROUPS
from bot.monitor.discord_notify import notify_fill

require_broker("alpaca")

ORDER_TIF = "gtc"
FILL_WAIT_SECONDS = 8
FILL_POLL_INTERVAL = 2
TRAILING_MONTHS = 3

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true",
                     help="Print the reconciled plan for both groups without submitting orders.")
args = parser.parse_args()

print("=" * 96)
print("CAPM2 WEEKLY REALLOCATION (Test 20 / Phase 6d)")
print("=" * 96)
if args.dry_run:
    print("*** DRY RUN -- no orders will be submitted ***")


def trailing_3mo_return(client, ticker):
    """Last close vs first close over TRAILING_MONTHS calendar months.
    Returns None if fewer than 2 bars are available (holiday-only window,
    a bad/delisted symbol, etc.) -- the caller decides what "None for
    this ticker" means for the whole group, this function never guesses."""
    end = datetime.now(timezone.utc).date()
    start = (pd.Timestamp(end) - pd.DateOffset(months=TRAILING_MONTHS)).date()
    bars = client.get_daily_bars(ticker, start, end)
    if len(bars) < 2:
        return None
    first_close = bars[0]["c"]
    last_close = bars[-1]["c"]
    return (last_close / first_close) - 1


def place_order(client, symbol, action, qty):
    resp = client.place_market_order(symbol, action, qty, tif=ORDER_TIF)
    if resp.get("error"):
        print(f"  REJECTED {action} {qty} {symbol}: {resp['message']}")
        return {"status": "rejected", "error": resp["message"]}

    order_id = resp["id"]
    print(f"  Submitted {action} {qty} {symbol} (order {order_id})")

    waited = 0
    status_resp = resp
    while waited < FILL_WAIT_SECONDS:
        time.sleep(FILL_POLL_INTERVAL)
        waited += FILL_POLL_INTERVAL
        status_resp = client.get_order(order_id)
        if status_resp["status"] == "filled":
            shares = float(status_resp["filled_qty"])
            price = float(status_resp["filled_avg_price"])
            notify_fill(symbol, action.lower(), shares, price)
            print(f"  FILLED {action} {shares:g} {symbol} @ {price:.4f}")
            return {"status": "filled", "shares": shares, "price": price}
        if status_resp["status"] in ("canceled", "expired", "rejected"):
            print(f"  {status_resp['status'].upper()} {action} {symbol}")
            return {"status": status_resp["status"]}

    print(f"  PENDING {action} {qty} {symbol}: status={status_resp['status']}, no fill within wait window")
    return {"status": "pending"}


for key, spec in GROUPS.items():
    print()
    print("=" * 96)
    print(f"{spec['label']}  ({', '.join(spec['tickers'])})")
    print("=" * 96)

    key_id = os.environ.get(spec["key_env"])
    secret = os.environ.get(spec["secret_env"])
    if not key_id or not secret:
        print(f"  Missing {spec['key_env']}/{spec['secret_env']} in .env -- skipping this group.")
        continue

    client = AlpacaClient(paper=True, api_key=key_id, api_secret=secret)
    if not client.connect():
        print(f"  Could not reach this account -- skipping {key} this run.")
        continue

    # ------------------------------------------------------------
    # SIGNAL: trailing 3-month return per ticker, this group only
    # ------------------------------------------------------------
    trailing_returns = {}
    aborted = False
    for t in spec["tickers"]:
        r = trailing_3mo_return(client, t)
        if r is None:
            print(f"  ABORTING this group: insufficient bar data for {t} -- "
                  f"a partial group would corrupt the z-score for every other name in it.")
            aborted = True
            break
        trailing_returns[t] = r
        print(f"  {t}: trailing {TRAILING_MONTHS}mo return = {r:+.2%}")
    if aborted:
        client.disconnect()
        continue

    weights = compute_weights(trailing_returns)
    print(f"\n  Weights: " + ", ".join(f"{t}={w:.1%}" for t, w in weights.items()))
    cash_weight = 1.0 - sum(weights.values())
    print(f"  Implied cash weight: {cash_weight:.1%}")


    # ------------------------------------------------------------
    # WEIGHTS -> TARGET SHARES, sized off THIS account's own equity
    # (each CAPM2 account is fully dedicated to one group -- no
    # separate "how much of the account to deploy" question here)
    # ------------------------------------------------------------
    equity = client.get_net_liquidation()
    target = {}
    for t in spec["tickers"]:
        price = client.get_last_price(t)
        target[t] = int((weights[t] * equity) // price)
    print(f"\n  Account equity: {equity:,.2f}")
    print("  Target shares: " + ", ".join(f"{t}={target[t]}" for t in spec["tickers"]))

    # ------------------------------------------------------------
    # RECONCILE + EXECUTE (broker-only reads, no journal -- see module
    # docstring for why)
    # ------------------------------------------------------------
    symbols = set(spec["tickers"])
    current_positions = client.get_positions()
    open_orders_raw = client.get_open_orders_raw()
    print(f"\n  Current positions: {current_positions if current_positions else '(flat)'}")

    diffs, warnings = reconcile(symbols, target, current_positions, open_orders_raw)
    if warnings:
        print(f"  *** {len(warnings)} trade(s) REFUSED by the safety check -- manual review needed ***")
        for w in warnings:
            print(f"    WARNING: {w}")

    print()
    results = execute_plan(diffs, lambda sym, action, qty: place_order(client, sym, action, qty),
                            dry_run=args.dry_run)

    filled = [r for r in results if r[3] and r[3].get("status") == "filled"]
    print(f"\n  Orders in plan: {len(results)}  |  Filled: {len(filled)}  |  "
          f"Pending/rejected: {len(results) - len(filled)}")

    client.disconnect()

print()
print("=" * 96)
print("DONE" + ("  (dry run -- nothing was submitted)" if args.dry_run else ""))
print("=" * 96)
