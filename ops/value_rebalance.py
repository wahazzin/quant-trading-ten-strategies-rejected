"""
value_rebalance.py -- Phase 6 forward test, Task 2: execute the target
value portfolio (ops/value_portfolio.py's output) against the real
paper account.

Every BUY/SELL is sized against the REAL current IBKR position for that
symbol (never what this script "thinks" it holds) -- correct even if a
prior rebalance only partially filled. Drops (previously-tracked names
no longer in today's target) are determined ONLY from the last executed
target list, data/value_portfolio_previous.csv -- never from "IBKR shows
a position here and it's not in today's target." That distinction
matters: this account can hold positions this strategy never touched
(e.g. the stale F bracket-order legs ops/verify_system.py flagged), and
nothing gets sold on the strength of an unrecognized symbol alone.

At inception (no data/value_portfolio_previous.csv yet), there is
nothing to drop -- every target name is simply bought from whatever its
current IBKR position is (0, for a fresh account).

Rejections are caught and reported per order; one rejected order does
not stop the rest of the rebalance. TradeMonitor is attached before any
order is placed so every fill is journaled with a real entry price.

On the run that completes without data/value_portfolio_state.json yet
existing, this IS inception: net liquidation (converted to USD) and
SPY's current price are captured and persisted there for
ops/value_report.py to measure "since inception" performance against.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
from datetime import datetime, timezone
from ib_async import Stock, MarketOrder

from bot.broker.ibkr_client import IBKRClient
from bot.broker.trade_monitor import TradeMonitor
from bot.broker.fx import FXConverter
from bot.journal.db import TradeJournal

TARGET_PATH = os.path.join("data", "value_portfolio_current.csv")
PREVIOUS_PATH = os.path.join("data", "value_portfolio_previous.csv")
STATE_PATH = os.path.join("data", "value_portfolio_state.json")
INSTRUMENT_CURRENCY = "USD"
BENCHMARK_SYMBOL = "SPY"
ORDER_WAIT_SECONDS = 4
ORDER_TIF = "GTC"  # DAY orders submitted while markets are closed are rejected outright
                    # by this gateway (error 399) rather than queued -- GTC rests correctly
                    # until the next session instead of dying on submission.

if not os.path.exists(TARGET_PATH):
    raise SystemExit(f"{TARGET_PATH} not found -- run ops/value_portfolio.py first.")

target_df = pd.read_csv(TARGET_PATH)
target = {row.ticker: int(row.target_shares) for row in target_df.itertuples(index=False)}
print("=" * 96)
print("VALUE PORTFOLIO REBALANCE")
print("=" * 96)
print(f"Target portfolio: {len(target)} names from {TARGET_PATH}")

previous_tickers = set()
if os.path.exists(PREVIOUS_PATH):
    prev_df = pd.read_csv(PREVIOUS_PATH)
    previous_tickers = set(prev_df["ticker"])
    print(f"Previously tracked portfolio: {len(previous_tickers)} names from {PREVIOUS_PATH}")
else:
    print("No previously tracked portfolio -- this is inception. Nothing to drop, only to buy.")

is_inception = not os.path.exists(STATE_PATH)
print(f"Inception run: {is_inception}")

errors_by_orderid = {}


def on_error(reqId, errorCode, errorString, contract):
    errors_by_orderid.setdefault(reqId, []).append((errorCode, errorString))


client = IBKRClient()
connected = client.connect()
print("Connected:", connected)
if not connected:
    raise SystemExit("Could not connect to IB Gateway.")

ib = client.ib
ib.errorEvent += on_error
ib.sleep(2)

journal = TradeJournal()
monitor = TradeMonitor(ib, journal)  # attached BEFORE any order so fills are journaled

current_positions = {p.contract.symbol: p.position for p in ib.positions()}
print(f"Current IBKR positions (all symbols): {current_positions if current_positions else '(flat)'}")

# Inception is "real capital deployed", not "orders submitted" -- if every
# order rejects (e.g. markets closed) or every GTC order is still resting
# unfilled, there is nothing to measure yet. This also catches a resting
# order from a PRIOR run that has since filled between runs.
already_holding_target_name = any(current_positions.get(t, 0) != 0 for t in target)

drop_tickers = previous_tickers - set(target.keys())

orders_placed = []
orders_filled = []
orders_rejected = []


def place_and_wait(ticker, action, qty, label):
    contract = Stock(ticker, "SMART", INSTRUMENT_CURRENCY)
    try:
        ib.qualifyContracts(contract)
    except Exception as e:
        print(f"  REJECTED {label} {ticker}: contract qualification failed ({e})")
        orders_rejected.append((ticker, label, str(e)))
        return

    order = MarketOrder(action, qty)
    order.tif = ORDER_TIF
    trade = ib.placeOrder(contract, order)
    orders_placed.append((ticker, action, qty))
    print(f"  Submitted {action} {qty} {ticker} ({label})")

    ib.sleep(ORDER_WAIT_SECONDS)
    status = trade.orderStatus.status
    my_errors = errors_by_orderid.get(trade.order.orderId, [])

    if status in ("Cancelled", "Inactive") or my_errors:
        err_str = "; ".join(f"{c}: {m}" for c, m in my_errors) if my_errors else f"status={status}"
        print(f"  REJECTED {action} {ticker}: {err_str}")
        orders_rejected.append((ticker, label, err_str))
        return

    fills = trade.fills
    if not fills:
        print(f"  PENDING {action} {ticker}: status={status}, no fill observed within the wait window")
        return

    total_shares = sum(f.execution.shares for f in fills)
    avg_price = sum(f.execution.shares * f.execution.price for f in fills) / total_shares
    print(f"  FILLED {action} {int(total_shares)} {ticker} @ avg {avg_price:.4f}")
    orders_filled.append((ticker, action, int(total_shares), avg_price))


print()
print("=" * 96)
print("SELLS -- names dropped from the target since the last tracked rebalance")
print("=" * 96)
if not drop_tickers:
    print("(none)")
for ticker in sorted(drop_tickers):
    current_qty = current_positions.get(ticker, 0)
    if current_qty == 0:
        print(f"  {ticker}: already flat, nothing to sell")
        continue
    place_and_wait(ticker, "SELL", int(abs(current_qty)), "drop")

print()
print("=" * 96)
print("BUYS / ADJUSTS -- target names")
print("=" * 96)
for ticker, target_shares in sorted(target.items()):
    current_qty = int(current_positions.get(ticker, 0))
    diff = target_shares - current_qty
    if diff == 0:
        print(f"  {ticker}: already at target ({target_shares} shares)")
        continue
    action = "BUY" if diff > 0 else "SELL"
    place_and_wait(ticker, action, abs(diff), "rebalance")

# Inception snapshot -- captured while still connected, before disconnect.
# Only fires once real capital has actually been deployed (a fill this run,
# or a resting order from a prior run that has since filled) -- not merely
# because orders were submitted.
if is_inception and (orders_filled or already_holding_target_name):
    print()
    print("=" * 96)
    print("INCEPTION SNAPSHOT")
    print("=" * 96)
    ib.sleep(2)
    equity = client.get_net_liquidation()
    account_currency = client.get_account_currency()
    fx = FXConverter(ib)
    equity_usd = fx.convert(equity, account_currency, INSTRUMENT_CURRENCY)

    spy_contract = Stock(BENCHMARK_SYMBOL, "SMART", "USD")
    ib.qualifyContracts(spy_contract)
    spy_bars = ib.reqHistoricalData(
        spy_contract, endDateTime="", durationStr="2 D", barSizeSetting="1 day",
        whatToShow="TRADES", useRTH=True, formatDate=1,
    )
    spy_price = spy_bars[-1].close if spy_bars else None

    inception_date = datetime.now(timezone.utc).date().isoformat()
    state = {
        "inception_date": inception_date,
        "inception_equity_usd": equity_usd,
        "inception_spy_price": spy_price,
    }
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)
    print(f"Inception date: {inception_date}")
    print(f"Inception equity: {equity_usd:.2f} USD")
    print(f"Inception SPY reference price: {spy_price}")
    print(f"Saved to {STATE_PATH}")
elif is_inception:
    print()
    print("Inception deferred: no fills yet and no pre-existing target-name positions -- "
          "orders are likely still resting (GTC) until markets reopen. Re-run this script "
          "after the next session to capture inception once real capital is actually deployed.")

client.disconnect()

print()
print("=" * 96)
print("SUMMARY")
print("=" * 96)
print(f"Orders placed:   {len(orders_placed)}")
print(f"Orders filled:   {len(orders_filled)}")
for t, a, q, p in orders_filled:
    print(f"  FILLED {a} {q} {t} @ {p:.4f}")
print(f"Orders rejected: {len(orders_rejected)}")
for t, label, err in orders_rejected:
    print(f"  REJECTED ({label}) {t}: {err}")
pending = len(orders_placed) - len(orders_filled) - len(orders_rejected)
if pending > 0:
    print(f"Orders still pending (no fill/rejection observed within the wait window): {pending}")

target_df.to_csv(PREVIOUS_PATH, index=False)
print(f"\nSaved this rebalance's target list to {PREVIOUS_PATH} for next time's drop detection.")
