"""
daily_snapshot.py -- fixes a known live bug: bot/journal/db.py has
defined an AccountSnapshot table and a log_account_snapshot() method
since early on, but nothing in the codebase ever called it. That is
why account equity history couldn't be reconstructed weeks ago.

Connects to IBKR, reads NetLiquidation and TotalCashValue, writes one
snapshot row, prints the current risk status, and disconnects.

Safe to run repeatedly: log_account_snapshot() only ever inserts a new
timestamped row (no uniqueness constraint), so running this twice in a
day just produces two data points, not a conflict. Safe to skip days:
account_snapshots has no expectation of daily continuity -- gaps are
just gaps, nothing depends on consecutive rows existing.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.broker.ibkr_client import IBKRClient
from bot.journal.db import TradeJournal
from bot.risk.circuit_breaker import RiskManager

client = IBKRClient()
connected = client.connect()
print("Connected:", connected)

if connected:
    ib = client.ib
    ib.sleep(2)  # let account data stream in fully

    net_liquidation = None
    total_cash = None
    for item in ib.accountSummary():
        if item.tag == "NetLiquidation":
            net_liquidation = float(item.value)
        elif item.tag == "TotalCashValue":
            total_cash = float(item.value)

    if net_liquidation is None:
        print("Could not read NetLiquidation from account summary -- not writing a snapshot.")
    else:
        journal = TradeJournal()
        journal.log_account_snapshot(net_liquidation=net_liquidation, cash=total_cash)
        print(f"Snapshot written: NetLiquidation={net_liquidation:.2f}, "
              f"TotalCashValue={total_cash if total_cash is not None else 'n/a'}")

        risk = RiskManager(journal)
        print(risk.status(net_liquidation))

    client.disconnect()
else:
    print("Could not connect to IB Gateway -- no snapshot written.")
