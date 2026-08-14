"""
Verification run for the fixed journal + monitor + risk manager.

What "success" looks like this time (unlike the old fake 14.29 -> 14.29 trade):
  - "📥 Entry recorded" prints when the BUY fills
  - "✅ Trade logged" prints when the SELL fills, with entry != exit
  - The trades table shows a REAL entry price, exit price, and nonzero PnL
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from bot.broker.ibkr_client import IBKRClient
from bot.broker.guard import require_broker

require_broker("ibkr")
from bot.broker.execution import OrderExecutor
from bot.broker.trade_monitor import TradeMonitor
from bot.journal.db import TradeJournal
from bot.risk.circuit_breaker import RiskManager
from ib_async import Stock, MarketOrder

ib_client = IBKRClient()
connected = ib_client.connect()
print("Connected:", connected)

if connected:
    ib = ib_client.ib
    ib.reqMarketDataType(3)

    journal = TradeJournal()
    monitor = TradeMonitor(ib, journal)
    risk = RiskManager(journal)          # <-- now takes journal (persistence)
    executor = OrderExecutor(ib_client, risk)

    equity = ib_client.get_net_liquidation()
    print("Equity:", equity)
    print("Risk status:", risk.status(equity))

    print("🚀 Placing BUY order...")
    result = executor.place_bracket_order(
        symbol="F",
        entry_price=14.0,
        stop_price=13.0,
        take_profit_price=16.0,
        equity=equity,
    )

    print("Waiting for BUY fill...")
    ib.sleep(8)

    position = ib_client.get_position("F")
    print("Position after buy:", position)

    if position > 0:
        print("🔻 Closing position...")
        contract = Stock("F", "SMART", "USD")
        ib.qualifyContracts(contract)

        close_order = MarketOrder("SELL", int(position))
        close_order.tif = "GTC"
        ib.placeOrder(contract, close_order)

        print("Waiting for SELL fill and journal log...")
        ib.sleep(15)

        final_position = ib_client.get_position("F")
        print("Final position:", final_position)

    # Show what actually got logged
    print("\n--- LAST 5 TRADES IN JOURNAL ---")
    import sqlite3
    conn = sqlite3.connect("trading.db")
    for row in conn.execute(
        "SELECT symbol, quantity, entry_price, exit_price, pnl, fees "
        "FROM trades ORDER BY id DESC LIMIT 5"
    ):
        print(row)
    conn.close()

    print("Done.")

ib_client.disconnect()