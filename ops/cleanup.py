"""
One-time cleanup: cancel ALL stale open orders on the paper account.

Why this exists: every bracket test left GTC stop/target legs resting
at IBKR. Those are why you kept seeing "repriced so as not to cross a
related resting order" warnings -- and almost certainly why 69 mystery
shares of F appeared out of nowhere between runs. A stale GTC order
fired while nothing was running.

Run this ONCE, then never trade with leftover orders floating around.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.broker.ibkr_client import IBKRClient

client = IBKRClient()
connected = client.connect()
print("Connected:", connected)

if connected:
    ib = client.ib

    open_trades = ib.openTrades()
    print(f"Open orders found: {len(open_trades)}")
    for t in open_trades:
        print(f"  - {t.order.action} {t.order.totalQuantity} "
              f"{t.contract.symbol} ({t.order.orderType}, "
              f"tif={t.order.tif}, status={t.orderStatus.status})")

    if open_trades:
        print("Cancelling ALL open orders (global cancel)...")
        ib.reqGlobalCancel()
        ib.sleep(3)

        remaining = ib.openTrades()
        print(f"Open orders after cancel: {len(remaining)}")
    else:
        print("No stale orders. Clean.")

    print("\nCurrent positions:")
    positions = ib.positions()
    if not positions:
        print("  FLAT - no positions. Perfect starting state.")
    for pos in positions:
        print(f"  - {pos.contract.symbol}: {pos.position} shares")

client.disconnect()
print("Cleanup done.")
