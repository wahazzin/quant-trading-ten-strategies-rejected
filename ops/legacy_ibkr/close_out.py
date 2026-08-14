"""
close_out.py -- final IBKR operation for this project. Ever.

The 20 GTC value-portfolio orders placed on IBKR before the Alpaca
migration (RESEARCH_LOG.md Phase 6) filled on 2026-08-03 -- AFTER the
same 20-stock portfolio had already been placed and filled on Alpaca.
That means both brokers now hold the same portfolio simultaneously:
duplicate real exposure, at two brokers, to a strategy this project
tracks as a single number. This script removes the IBKR half of that
permanently: cancel every open order, sell every position flat, and
verify both are zero. After this runs clean, Alpaca is the ONLY broker
this project's forward test touches, and IBKR is fully decommissioned
-- see bot/broker/guard.py for the mechanism that stops this from
happening again (a script written for a broker that doesn't match
ACTIVE_BROKER in .env now refuses to run).

This is NOT a real trade and is deliberately NOT journaled -- exactly
like ops/legacy_ibkr/flatten.py's precedent for the old ghost -25 F
short: closing out a decommissioned broker's leftover exposure is a
repair/decommission operation, not a strategy entry or exit, and
polluting the trade journal with it would misrepresent the forward
test's actual trade history (which lives entirely on Alpaca as of this
migration).

Deliberately run with ACTIVE_BROKER=ibkr passed on the command line for
this one invocation only (not written into .env) -- .env's real,
persistent ACTIVE_BROKER stays "alpaca" throughout and after this runs,
proving the guard mechanism itself works: this script refuses to run
under the default .env value, and only proceeds because this one
special, human-authorized invocation explicitly overrides it.

Run with:  ACTIVE_BROKER=ibkr python ops/legacy_ibkr/close_out.py
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from bot.broker.ibkr_client import IBKRClient
from bot.broker.guard import require_broker

require_broker("ibkr")

client = IBKRClient()
connected = client.connect()
print("Connected:", connected)
if not connected:
    raise SystemExit("Could not connect to IB Gateway -- is it running? Start it and re-run.")

ib = client.ib
ib.sleep(2)

print("=" * 96)
print("IBKR FINAL CLOSE-OUT")
print("=" * 96)

# ============================================================
# STEP 1 -- cancel every open order
# ============================================================
print("\n--- STEP 1: cancel all open orders ---")
open_trades = ib.openTrades()
print(f"Open orders found: {len(open_trades)}")
for t in open_trades:
    print(f"  {t.order.action} {t.order.totalQuantity} {t.contract.symbol} "
          f"({t.order.orderType}, tif={t.order.tif}, status={t.orderStatus.status})")

if open_trades:
    ib.reqGlobalCancel()
    ib.sleep(3)
    remaining_orders = ib.openTrades()
    print(f"Open orders after cancel: {len(remaining_orders)}")
else:
    remaining_orders = []
    print("No open orders to cancel.")

# ============================================================
# STEP 2 -- sell every position flat
# ============================================================
print("\n--- STEP 2: flatten every position ---")
positions = ib.positions()
print(f"Positions found: {len(positions)}")
for pos in positions:
    print(f"  {pos.contract.symbol}: {pos.position} shares @ avg cost {pos.avgCost:.4f}")

if not positions:
    print("Already flat -- nothing to sell.")
else:
    from ib_async import Stock, MarketOrder

    ib.reqMarketDataType(3)  # delayed data -- without this, a market order can sit
    # "Submitted" indefinitely on the paper matching engine for a symbol nothing has
    # requested a quote for yet. Found live during this exact close-out: 12 of 13
    # positions flattened within seconds, but TEN sat unfilled for 30+ seconds until
    # a manual reqMktData() call for it went out -- it filled within 5 seconds of that.
    # Requesting (and holding) a market-data subscription for every symbol before
    # placing its order avoids depending on some other part of the process having
    # already warmed one up by coincidence.
    tickers = {}
    for pos in positions:
        contract = Stock(pos.contract.symbol, "SMART", "USD")
        ib.qualifyContracts(contract)
        tickers[pos.contract.symbol] = ib.reqMktData(contract, "", False, False)
    ib.sleep(3)

    for pos in positions:
        symbol = pos.contract.symbol
        qty = pos.position
        if qty == 0:
            continue
        action = "SELL" if qty > 0 else "BUY"
        contract = Stock(symbol, "SMART", "USD")
        order = MarketOrder(action, int(abs(qty)))
        order.tif = "DAY"
        print(f"  Submitting {action} {int(abs(qty))} {symbol} to flatten...")
        ib.placeOrder(contract, order)

    print("Waiting for fills...")
    for _ in range(6):  # up to 30s, polling every 5s instead of one fixed sleep
        ib.sleep(5)
        if not ib.openTrades():
            break

    for symbol, t in tickers.items():
        ib.cancelMktData(t.contract)

# ============================================================
# STEP 3 -- verify zero positions and zero open orders
# ============================================================
print("\n--- STEP 3: verify flat ---")
final_positions = ib.positions()
final_orders = ib.openTrades()

print(f"Final positions: {len(final_positions)}")
for pos in final_positions:
    print(f"  STILL OPEN: {pos.contract.symbol}: {pos.position} shares")

print(f"Final open orders: {len(final_orders)}")
for t in final_orders:
    print(f"  STILL OPEN: {t.order.action} {t.order.totalQuantity} {t.contract.symbol} "
          f"(status={t.orderStatus.status})")

client.disconnect()

print()
print("=" * 96)
print("SUMMARY")
print("=" * 96)
positions_clean = len(final_positions) == 0
orders_clean = len(final_orders) == 0
print(f"Zero positions: {positions_clean}")
print(f"Zero open orders: {orders_clean}")
if positions_clean and orders_clean:
    print("IBKR IS FULLY FLAT. This account is decommissioned for this project's forward test.")
else:
    print("NOT CLEAN -- re-run this script (some fills may need more time), or check manually "
          "in TWS/IB Gateway before assuming this is done.")
