"""
value_rebalance.py -- Phase 6 forward test: execute the target value
portfolio (ops/value_portfolio.py's output) against the real Alpaca
paper account.

Migrated from IBKR to Alpaca (see ROADMAP.md) -- no gateway process to
keep running for a months-long forward test, REST-only, and no
per-order commission (see ROADMAP.md constraint C1 for the real cost
basis this replaces).

Every order is sized through bot/broker/reconcile.py's single
reconciliation function -- current position + pending orders + target
-> net diff. That function exists because three separate incidents of
stale/duplicate orders happened in this project before it did (see
reconcile.py's docstring for the full list) -- nothing in this script
computes a buy/sell diff on its own anymore.

Fills are journaled two ways:
  1. Immediately, if an order fills within the short wait window after
     submission.
  2. On catch-up, at the start of every run: any symbol where the
     broker shows a position but the local journal has no matching
     OpenPosition record gets its most recent filled order looked up
     and journaled then. A GTC order placed while markets are closed
     can fill days after submission, potentially across several script
     runs -- this makes fill-journaling correct regardless of when
     that happens.

Drops (previously-tracked names no longer in today's target) are
folded into the SAME reconciliation call as everything else --
`symbols` is target names UNION previously-tracked names, and
`reconcile()` treats a symbol absent from today's target as target=0.
This account can hold positions this strategy never touched (old test
trades, manual holdings); nothing outside `symbols` is ever considered.

--dry-run prints the reconciled plan without submitting anything.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import argparse
import pandas as pd
from datetime import datetime, timezone

from bot.broker.alpaca_client import AlpacaClient
from bot.broker.reconcile import reconcile, execute_plan
from bot.journal.db import TradeJournal

TARGET_PATH = os.path.join("data", "value_portfolio_current.csv")
PREVIOUS_PATH = os.path.join("data", "value_portfolio_previous.csv")
STATE_PATH = os.path.join("data", "value_portfolio_state.json")
BENCHMARK_SYMBOL = "SPY"
ORDER_TIF = "gtc"
FILL_WAIT_SECONDS = 8
FILL_POLL_INTERVAL = 2

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true", help="Print the reconciled plan without submitting orders.")
args = parser.parse_args()

if not os.path.exists(TARGET_PATH):
    raise SystemExit(f"{TARGET_PATH} not found -- run ops/value_portfolio.py first.")

target_df = pd.read_csv(TARGET_PATH)
target = {row.ticker: int(row.target_shares) for row in target_df.itertuples(index=False)}
print("=" * 96)
print("VALUE PORTFOLIO REBALANCE (Alpaca)")
print("=" * 96)
print(f"Target portfolio: {len(target)} names from {TARGET_PATH}")
if args.dry_run:
    print("*** DRY RUN -- no orders will be submitted ***")

previous_tickers = set()
if os.path.exists(PREVIOUS_PATH):
    prev_df = pd.read_csv(PREVIOUS_PATH)
    previous_tickers = set(prev_df["ticker"])
    print(f"Previously tracked portfolio: {len(previous_tickers)} names from {PREVIOUS_PATH}")
else:
    print("No previously tracked portfolio -- this is inception. Nothing to drop, only to buy.")

is_inception = not os.path.exists(STATE_PATH)
print(f"Inception run: {is_inception}")

client = AlpacaClient(paper=True)
connected = client.connect()
if not connected:
    raise SystemExit("Could not reach Alpaca.")

journal = TradeJournal()
symbols = set(target.keys()) | previous_tickers

current_positions = client.get_positions()
open_orders_raw = client.get_open_orders_raw()
pending_orders = client.get_open_orders()  # display only -- see reconcile.pending_quantity() for the real math
print(f"\nCurrent Alpaca positions (all symbols): {current_positions if current_positions else '(flat)'}")
print(f"Pending (open) order quantity by symbol: {pending_orders if pending_orders else '(none)'}")

# ============================================================
# CATCH-UP: journal any fills from prior runs that were never recorded
# ============================================================
print()
print("=" * 96)
print("CATCH-UP: checking for unjournaled fills")
print("=" * 96)
caught_up_any = False
for sym in sorted(symbols):
    held = current_positions.get(sym, 0)
    if held == 0:
        continue
    existing = journal.get_open_position(sym)
    if existing is not None and abs(existing["quantity"] - held) < 0.5:
        continue  # already correctly journaled

    # Check BOTH closed (fully filled) orders AND still-open orders that are
    # PARTIALLY filled -- an order stays "open" in Alpaca's API while partially
    # filled, so a closed-orders-only search misses exactly this case (it did,
    # the first time this ran: 5 partial fills went unjournaled because none of
    # their orders had closed yet).
    candidates = (client.get_closed_orders(sym) + client.get_open_orders_raw(sym))
    filled = [o for o in candidates if o.get("filled_qty") and float(o["filled_qty"]) > 0]
    if not filled:
        print(f"  {sym}: broker shows {held} shares but no filled order found -- skipping (manual check needed)")
        continue
    latest = max(filled, key=lambda o: o["filled_at"])
    fill_time = datetime.fromisoformat(latest["filled_at"].replace("Z", "+00:00"))
    already_journaled = existing["quantity"] if existing is not None else 0.0
    new_qty = float(latest["filled_qty"]) - already_journaled
    if new_qty <= 0:
        continue
    journal.record_entry_fill(symbol=sym, shares=new_qty,
                               price=float(latest["filled_avg_price"]), fill_time=fill_time)
    print(f"  {sym}: journaled catch-up fill -- {new_qty:g} shares @ {latest['filled_avg_price']} "
          f"(order {latest['id']}, status={latest.get('status')}, filled_qty={latest['filled_qty']})")
    caught_up_any = True
if not caught_up_any:
    print("  (nothing to catch up -- all broker positions already journaled)")


# ============================================================
# RECONCILE + EXECUTE
# ============================================================
def place_order(symbol, action, qty):
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
            fill_time = datetime.fromisoformat(status_resp["filled_at"].replace("Z", "+00:00"))
            if action == "BUY":
                journal.record_entry_fill(symbol=symbol, shares=shares, price=price, fill_time=fill_time)
            else:
                journal.record_exit_fill(symbol=symbol, shares=shares, price=price, fill_time=fill_time)
            print(f"  FILLED {action} {shares:g} {symbol} @ {price:.4f}")
            return {"status": "filled", "shares": shares, "price": price}
        if status_resp["status"] in ("canceled", "expired", "rejected"):
            print(f"  {status_resp['status'].upper()} {action} {symbol}")
            return {"status": status_resp["status"]}

    print(f"  PENDING {action} {qty} {symbol}: status={status_resp['status']}, no fill within wait window")
    return {"status": "pending"}


print()
print("=" * 96)
print("RECONCILED PLAN")
print("=" * 96)
diffs, warnings = reconcile(symbols, target, current_positions, open_orders_raw)
if warnings:
    print(f"*** {len(warnings)} trade(s) REFUSED by the hard safety check -- nothing submitted for these, manual review needed ***")
    for w in warnings:
        print(f"  WARNING: {w}")
results = execute_plan(diffs, place_order, dry_run=args.dry_run)

# ============================================================
# INCEPTION SNAPSHOT -- fires once real capital is confirmed deployed
# ============================================================
already_holding_target_name = any(current_positions.get(t, 0) != 0 for t in target)
any_filled_this_run = any(r[3] and r[3].get("status") == "filled" for r in results)

if is_inception and not args.dry_run and (any_filled_this_run or already_holding_target_name or caught_up_any):
    print()
    print("=" * 96)
    print("INCEPTION SNAPSHOT")
    print("=" * 96)
    equity_usd = client.get_net_liquidation()

    # The Alpaca account's total equity is not the right inception baseline: this
    # strategy deploys ~$3,500 of it (the same 20-position sizing already computed
    # for the account this portfolio was originally sized for), leaving the rest as
    # uninvested cash. Measuring "portfolio return" against WHOLE-ACCOUNT equity
    # would dilute the value strategy's actual performance by that uninvested cash
    # -- on a $98k account with $3.5k deployed, a real move in the 20 positions
    # would barely register. inception_portfolio_value is the capital actually
    # deployed (real fill price x quantity, from the journal), and is what
    # value_report.py measures returns against. Whole-account equity is kept too,
    # for context/audit, but is not the return-measurement baseline.
    deployed_value = 0.0
    for t in target:
        pos = journal.get_open_position(t)
        if pos:
            deployed_value += pos["quantity"] * pos["avg_entry_price"]

    spy_price = client.get_last_price(BENCHMARK_SYMBOL)

    inception_date = datetime.now(timezone.utc).date().isoformat()
    state = {
        "inception_date": inception_date,
        "inception_equity_usd": equity_usd,
        "inception_portfolio_value_usd": deployed_value,
        "inception_spy_price": spy_price,
        "broker": "alpaca",
    }
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)
    print(f"Inception date: {inception_date}")
    print(f"Whole-account equity (context only, NOT the return baseline): {equity_usd:.2f} USD")
    print(f"Capital actually deployed to this strategy (the real return baseline): {deployed_value:.2f} USD")
    print(f"Inception SPY reference price: {spy_price}")
    print(f"Saved to {STATE_PATH}")
elif is_inception:
    print()
    print("Inception deferred: no fills yet and no pre-existing target-name positions. "
          "Re-run this script later to check status and capture inception once real capital is deployed.")

client.disconnect()

# ============================================================
# SUMMARY
# ============================================================
print()
print("=" * 96)
print("SUMMARY")
print("=" * 96)
filled = [r for r in results if r[3] and r[3].get("status") == "filled"]
pending = [r for r in results if r[3] and r[3].get("status") == "pending"]
rejected = [r for r in results if r[3] and r[3].get("status") in ("rejected", "canceled", "expired")]
print(f"Orders in plan:  {len(results)}")
print(f"Orders filled:   {len(filled)}")
for sym, action, qty, res in filled:
    print(f"  FILLED {action} {res['shares']:g} {sym} @ {res['price']:.4f}")
print(f"Orders pending:  {len(pending)}")
print(f"Orders rejected: {len(rejected)}")
for sym, action, qty, res in rejected:
    print(f"  {res['status'].upper()} {action} {qty} {sym}")

if not args.dry_run:
    target_df.to_csv(PREVIOUS_PATH, index=False)
    print(f"\nSaved this rebalance's target list to {PREVIOUS_PATH} for next time's drop detection.")
else:
    print(f"\nDry run -- {PREVIOUS_PATH} NOT updated.")
