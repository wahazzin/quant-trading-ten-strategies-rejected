"""
flatten.py -- close ANY leftover position in F so we start from zero.

Run this MONDAY after 15:30 (Swedish time), when the US market is open.
It will not work on the weekend because the market is closed.

Why this script exists: cleanup.py found we are SHORT 25 shares of F
(an old stop-loss ticket sold shares we didn't own). To fix a short,
you BUY the shares back. This script figures out the position on its
own and places the correct order either way.

NOTE: This deliberately does NOT attach the TradeMonitor. Buying back
a short is a repair job, not a real trade -- we don't want it polluting
the trade journal as a fake "entry".
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.broker.ibkr_client import IBKRClient
from ib_async import Stock, MarketOrder

client = IBKRClient()
connected = client.connect()
print("Connected:", connected)

if connected:
    ib = client.ib
    ib.reqMarketDataType(3)

    position = client.get_position("F")
    print(f"Current F position: {position}")

    if position == 0:
        print("Already flat. Nothing to do.")
    else:
        contract = Stock("F", "SMART", "USD")
        ib.qualifyContracts(contract)

        if position < 0:
            action, qty = "BUY", int(abs(position))
            print(f"Short {qty} shares -> BUYING {qty} to repay the broker.")
        else:
            action, qty = "SELL", int(position)
            print(f"Long {qty} shares -> SELLING {qty} to go flat.")

        order = MarketOrder(action, qty)   # DAY order on purpose:
        ib.placeOrder(contract, order)     # if it doesn't fill today,
        print("Waiting for fill...")       # it dies today. No new ghosts.
        ib.sleep(10)

        final = client.get_position("F")
        print(f"Final F position: {final}")
        if final == 0:
            print("FLAT. Clean slate achieved.")
        else:
            print("Still not flat -- is the market open? Tell Claude.")

client.disconnect()
