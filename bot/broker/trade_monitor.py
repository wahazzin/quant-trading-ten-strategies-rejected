# RETIRED (2026-08-03) -- IBKR is fully decommissioned; this class is IBKR-specific
# (wires into ib_async's execDetailsEvent/commissionReportEvent) and is only used by
# ops/legacy_ibkr/main.py and ops/legacy_ibkr/benchmark_buy.py. See RESEARCH_LOG.md's
# Phase 6 section. Alpaca fills are journaled directly in ops/value_rebalance.py instead.
from datetime import datetime, timezone


class TradeMonitor:
    """
    Fill-based trade monitor, v2.

    Old version's fatal bug: it read the entry price from the CLOSING
    order's own fills (trade.fills[0]), which is why the journal showed
    entry 14.29 = exit 14.29 = PnL $0.00.

    New design:
      - Every BUY fill  -> journal.record_entry_fill()  (persisted to DB)
      - Every SELL fill -> journal.record_exit_fill()   (uses real stored entry)

    Because entries live in the database, this survives restarts:
    buy in one script run, sell in another, and it still logs correctly.
    That was one of your biggest roadblocks -- fixed at the root.
    """

    def __init__(self, ib, journal):
        self.ib = ib
        self.journal = journal
        self.ib.execDetailsEvent += self.on_execution
        self.ib.commissionReportEvent += self.on_commission

        # execId -> fee, filled in when IBKR sends commission reports
        # (they often arrive slightly after the fill itself)
        self._fees = {}

    def on_commission(self, trade, fill, report):
        try:
            if report.commission:
                self._fees[fill.execution.execId] = abs(report.commission)
        except Exception:
            pass

    def on_execution(self, trade, fill):
        try:
            symbol = trade.contract.symbol
            side = fill.execution.side  # 'BOT' (bought) or 'SLD' (sold)
            shares = fill.execution.shares
            price = fill.execution.price
            fill_time = fill.execution.time or datetime.now(timezone.utc)
            fees = self._fees.get(fill.execution.execId, 0.0)

            if side == "BOT":
                self.journal.record_entry_fill(
                    symbol=symbol,
                    shares=shares,
                    price=price,
                    fill_time=fill_time,
                )
                print(f"📥 Entry recorded: BUY {shares} {symbol} @ {price:.2f}")

            elif side == "SLD":
                self.journal.record_exit_fill(
                    symbol=symbol,
                    shares=shares,
                    price=price,
                    fill_time=fill_time,
                    fees=fees,
                )

        except Exception as e:
            # A monitor crash must never kill the trading loop.
            print(f"⚠️ TradeMonitor error: {e}")