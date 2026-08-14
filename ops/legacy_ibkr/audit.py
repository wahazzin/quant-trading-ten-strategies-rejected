"""
audit.py -- account state audit.

Why this exists: equity dropped from a peak of 47042.20 SEK (Saturday)
to 37977.21 SEK (today) -- a ~19% drop -- over a WEEKEND, when the US
market was closed and nothing should have moved. The known -25 F short
position is nowhere near big enough to explain that gap on its own.
This pulls every position and the real account numbers so we know
exactly where the money went, instead of guessing.
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

if connected:
    ib = client.ib
    ib.sleep(2)  # let account data stream in fully

    print("\n=== ALL POSITIONS (every symbol) ===")
    positions = ib.positions()
    if not positions:
        print("  (none reported -- fully flat)")
    for pos in positions:
        print(f"  {pos.contract.symbol}: {pos.position} shares "
              f"@ avg cost {pos.avgCost:.4f}")

    print("\n=== ACCOUNT SUMMARY (all tags) ===")
    for item in ib.accountSummary():
        if item.tag in ("NetLiquidation", "TotalCashValue", "BuyingPower",
                        "GrossPositionValue", "RealizedPnL", "UnrealizedPnL",
                        "AvailableFunds", "ExcessLiquidity"):
            print(f"  {item.tag}: {item.value} {item.currency}")

client.disconnect()

print("\n=== EQUITY SNAPSHOT HISTORY (from trading.db) ===")
import sqlite3
conn = sqlite3.connect("trading.db")
rows = conn.execute(
    "SELECT timestamp, net_liquidation, cash FROM account_snapshots "
    "ORDER BY id"
).fetchall()
if not rows:
    print("  (empty -- no snapshot was ever logged, so we have no "
          "recorded trail of when equity actually dropped)")
for r in rows:
    print(f"  {r[0]} | NetLiq: {r[1]} | Cash: {r[2]}")

print("\n=== ALL LOGGED TRADES (full history) ===")
for r in conn.execute(
    "SELECT symbol, quantity, entry_price, exit_price, pnl, fees, "
    "entry_time FROM trades ORDER BY id"
):
    print(f"  {r}")
conn.close()
