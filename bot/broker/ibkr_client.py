# RETIRED (2026-08-03) -- IBKR is fully decommissioned; only ops/legacy_ibkr/*.py
# imports this, each guarded by bot/broker/guard.py's require_broker("ibkr").
# See RESEARCH_LOG.md's Phase 6 section.
from ib_async import IB
from bot.config import IB_HOST, IB_PORT, IB_CLIENT_ID


class IBKRClient:
    def __init__(self):
        self.ib = IB()

    def connect(self):
        self.ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)
        return self.ib.isConnected()

    def disconnect(self):
        if self.ib.isConnected():
            self.ib.disconnect()

    def get_net_liquidation(self):
        summary = self.ib.accountSummary()
        for item in summary:
            if item.tag == "NetLiquidation":
                return float(item.value)
        return None

    def get_account_currency(self):
        summary = self.ib.accountSummary()
        for item in summary:
            if item.tag == "NetLiquidation":
                return item.currency
        return "USD"

    def get_position(self, symbol):
        for pos in self.ib.positions():
            if pos.contract.symbol == symbol:
                return pos.position
        return 0