"""
benchmark_buy.py -- establish the buy-and-hold benchmark position.

Every strategy this project tested was rejected (see RESEARCH_LOG.md).
The declared benchmark is buy-and-hold, which beat every one of them.
This script puts that benchmark ON the account so it can be tracked
the same way any strategy would be: real fills, real journal entries,
real equity history.

Sizing: ~80% of account equity, modeled on position_sizing.py's
affordability formula (equity / entry_price) rather than calling
calculate_position_size() directly -- that function requires a
stop_price to do its risk-based sizing, and this is a no-stop,
buy-and-hold position by design, so there's no stop distance to size
against. The 80% cap mirrors that function's max_position_pct role,
just scaled up from its 20% default since this IS the whole strategy,
not one bracketed swing trade.

If the order is rejected (e.g. PRIIPs/KID documentation restrictions on
EU accounts buying US-domiciled ETFs), this catches it, prints the
exact IBKR error, and does NOT retry or fall back to another symbol --
that decision is left to a human.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ib_async import Stock, MarketOrder

from bot.broker.ibkr_client import IBKRClient
from bot.broker.guard import require_broker

require_broker("ibkr")
from bot.broker.trade_monitor import TradeMonitor
from bot.broker.fx import FXConverter
from bot.journal.db import TradeJournal

SYMBOL = "SPY"
TARGET_EQUITY_PCT = 0.80
INSTRUMENT_CURRENCY = "USD"

errors_seen = []


def on_error(reqId, errorCode, errorString, contract):
    errors_seen.append((errorCode, errorString))
    print(f"IBKR error {errorCode}: {errorString}")


client = IBKRClient()
connected = client.connect()
print("Connected:", connected)

if not connected:
    raise SystemExit("Could not connect to IB Gateway.")

ib = client.ib
ib.errorEvent += on_error
ib.sleep(2)

existing_position = client.get_position(SYMBOL)
print(f"Current {SYMBOL} position: {existing_position}")

if existing_position != 0:
    print(f"Already holding {existing_position} shares of {SYMBOL} -- not flat, nothing to do.")
    client.disconnect()
    raise SystemExit(0)

journal = TradeJournal()
monitor = TradeMonitor(ib, journal)   # attached BEFORE the order so the fill is caught

equity = client.get_net_liquidation()
account_currency = client.get_account_currency()
print(f"Account equity: {equity:.2f} {account_currency}")

fx = FXConverter(ib)
equity_in_instrument_ccy = fx.convert(equity, account_currency, INSTRUMENT_CURRENCY)
print(f"Equity in {INSTRUMENT_CURRENCY}: {equity_in_instrument_ccy:.2f}")

contract = Stock(SYMBOL, "SMART", INSTRUMENT_CURRENCY)
ib.qualifyContracts(contract)

bars = ib.reqHistoricalData(
    contract, endDateTime="", durationStr="2 D", barSizeSetting="1 day",
    whatToShow="TRADES", useRTH=True, formatDate=1,
)
if not bars:
    print("Could not fetch a reference price for sizing -- aborting.")
    client.disconnect()
    raise SystemExit(1)
reference_price = bars[-1].close
print(f"Reference price ({bars[-1].date}): {reference_price:.2f}")

# affordability sizing, modeled on position_sizing.py's affordability_size
# (equity / entry_price), scaled to the 80% target instead of full equity
target_value = equity_in_instrument_ccy * TARGET_EQUITY_PCT
shares = int(target_value / reference_price)
print(f"Target: {TARGET_EQUITY_PCT*100:.0f}% of equity = {target_value:.2f} {INSTRUMENT_CURRENCY} "
      f"-> {shares} shares at ~{reference_price:.2f}")

if shares <= 0:
    print("Calculated share count is 0 -- aborting, nothing to buy.")
    client.disconnect()
    raise SystemExit(1)

order = MarketOrder("BUY", shares)
order.tif = "DAY"
trade = ib.placeOrder(contract, order)

print("Order submitted, waiting for fill or rejection...")
ib.sleep(8)

status = trade.orderStatus.status
print(f"Order status: {status}")

if status in ("Cancelled", "Inactive") or errors_seen:
    print("\nORDER WAS NOT FILLED.")
    print("Trade log:")
    for entry in trade.log:
        print(f"  {entry.time} [{entry.status}] {entry.message}")
    if errors_seen:
        print("IBKR error events received:")
        for code, msg in errors_seen:
            print(f"  {code}: {msg}")
    print("\nNot retrying and not falling back to another symbol -- "
          "pick a different instrument if this is a PRIIPs/KID restriction.")
    client.disconnect()
    raise SystemExit(1)

fills = trade.fills
if not fills:
    print(f"Order status is '{status}' but no fills recorded yet -- "
          f"check IBKR TWS/Gateway manually before assuming success.")
    client.disconnect()
    raise SystemExit(1)

total_shares = sum(f.execution.shares for f in fills)
avg_fill_price = sum(f.execution.shares * f.execution.price for f in fills) / total_shares

print(f"\nFILLED: {int(total_shares)} shares of {SYMBOL} @ avg {avg_fill_price:.4f}")

ib.sleep(1)  # let TradeMonitor's execDetailsEvent handler run
recorded = journal.get_open_position(SYMBOL)
if recorded is None:
    print("WARNING: fill happened but the journal has no open position recorded for "
          f"{SYMBOL} -- TradeMonitor may not have caught the execution event.")
else:
    print(f"Journal confirms open position: {recorded['quantity']} shares @ "
          f"entry {recorded['avg_entry_price']:.4f} (entered {recorded['entry_time']})")

client.disconnect()
