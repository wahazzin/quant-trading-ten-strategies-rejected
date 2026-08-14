"""
weekly_report.py -- project Rule 2's weekly report generator.

Reads trades, equity history, and risk state from trading.db (no
IBKR connection needed for those -- they're already logged locally).
The one thing that needs a live connection is the SPY benchmark
return, fetched fresh from IBKR for the identical window.

Window is configurable (default 7 days): `python weekly_report.py 14`
for a 14-day window, or no argument for the default 7.

Handles the zero-trades case gracefully -- expected to be the normal
state for a buy-and-hold benchmark position that never turns over.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
from datetime import datetime, timedelta

from bot.journal.db import TradeJournal, Trade, AccountSnapshot
from bot.risk.circuit_breaker import RiskManager
from bot.broker.ibkr_client import IBKRClient
from bot.broker.guard import require_broker

require_broker("ibkr")
from ib_async import Stock

parser = argparse.ArgumentParser(description="Weekly (or N-day) performance report")
parser.add_argument("days", nargs="?", type=int, default=7, help="Report window in days (default 7)")
parser.add_argument("--symbol", default="SPY", help="Benchmark symbol (default SPY)")
args = parser.parse_args()

WINDOW_DAYS = args.days
BENCHMARK_SYMBOL = args.symbol
window_start = datetime.utcnow() - timedelta(days=WINDOW_DAYS)

journal = TradeJournal()
session = journal.Session()

print("=" * 70)
print(f"WEEKLY REPORT -- last {WINDOW_DAYS} day(s), window start {window_start.date()}")
print("=" * 70)

# ------------------------------------------------------------------
# TRADE STATS (local DB only)
# ------------------------------------------------------------------
try:
    trades = (
        session.query(Trade)
        .filter(Trade.exit_time >= window_start)
        .order_by(Trade.exit_time)
        .all()
    )

    print(f"\nTrades closed in window: {len(trades)}")
    if not trades:
        print("No trades in this period -- normal for a buy-and-hold benchmark position.")
    else:
        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        win_rate = len(wins) / len(trades)
        loss_rate = len(losses) / len(trades)
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss_mag = abs(sum(losses) / len(losses)) if losses else 0.0
        expectancy = win_rate * avg_win - loss_rate * avg_loss_mag
        total_pnl = sum(pnls)

        print(f"Win rate:              {win_rate*100:.1f}%  ({len(wins)}W / {len(losses)}L)")
        print(f"Average win:           ${avg_win:,.2f}")
        print(f"Average loss:          ${avg_loss_mag:,.2f}")
        print(f"Expectancy per trade:  ${expectancy:,.2f}")
        print(f"Total realized P&L:    ${total_pnl:,.2f}")

    # ------------------------------------------------------------------
    # EQUITY / DRAWDOWN (local DB only)
    # ------------------------------------------------------------------
    print()
    latest_snapshot = session.query(AccountSnapshot).order_by(AccountSnapshot.timestamp.desc()).first()

    if latest_snapshot is None:
        print("No account snapshots on file yet (run daily_snapshot.py) -- "
              "skipping equity change and drawdown.")
        current_equity = None
    else:
        current_equity = latest_snapshot.net_liquidation
        print(f"Latest equity snapshot: ${current_equity:,.2f} (as of {latest_snapshot.timestamp})")

        past_snapshot = (
            session.query(AccountSnapshot)
            .filter(AccountSnapshot.timestamp <= window_start)
            .order_by(AccountSnapshot.timestamp.desc())
            .first()
        )
        if past_snapshot is None:
            past_snapshot = session.query(AccountSnapshot).order_by(AccountSnapshot.timestamp).first()
            if past_snapshot is not None and past_snapshot.id != latest_snapshot.id:
                print(f"NOTE: no snapshot as old as the window start -- using earliest available "
                      f"({past_snapshot.timestamp}) as the reference point instead.")

        if past_snapshot is not None and past_snapshot.id != latest_snapshot.id:
            equity_change_pct = (current_equity / past_snapshot.net_liquidation - 1) * 100
            print(f"Equity change over window: {equity_change_pct:+.2f}% "
                  f"(${past_snapshot.net_liquidation:,.2f} -> ${current_equity:,.2f})")
        else:
            print("Only one snapshot on file -- no equity change figure yet.")

        risk = RiskManager(journal)
        print(risk.status(current_equity))
finally:
    session.close()

# ------------------------------------------------------------------
# SPY BENCHMARK RETURN (live IBKR fetch, same window)
# ------------------------------------------------------------------
print()
print(f"--- {BENCHMARK_SYMBOL} benchmark return over the identical window ---")
client = IBKRClient()
connected = client.connect()
print("Connected:", connected)

if connected:
    ib = client.ib
    contract = Stock(BENCHMARK_SYMBOL, "SMART", "USD")
    ib.qualifyContracts(contract)

    duration_str = f"{max(WINDOW_DAYS, 1)} D"   # IBKR "D" duration = calendar days
    bars = ib.reqHistoricalData(
        contract, endDateTime="", durationStr=duration_str,
        barSizeSetting="1 day", whatToShow="TRADES", useRTH=True, formatDate=1,
    )
    client.disconnect()

    if not bars or len(bars) < 2:
        print(f"Not enough {BENCHMARK_SYMBOL} bars returned to compute a window return.")
    else:
        start_bar, end_bar = bars[0], bars[-1]
        bench_return = (end_bar.close / start_bar.open - 1) * 100
        print(f"{BENCHMARK_SYMBOL}: {start_bar.date} open {start_bar.open:.2f} -> "
              f"{end_bar.date} close {end_bar.close:.2f}  ({bench_return:+.2f}%)")
else:
    print("Could not connect to IB Gateway -- skipping benchmark return.")

print()
print("=" * 70)
