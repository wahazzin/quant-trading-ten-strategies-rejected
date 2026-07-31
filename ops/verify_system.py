"""
verify_system.py -- infrastructure verification checklist.

Nine strategy tests were rejected (RESEARCH_LOG.md). This does not test
a signal -- it proves the execution/logging/risk pipeline is trustworthy
before it's ever relied on for the buy-and-hold benchmark position (or
anything else). Five checks, each PASS/FAIL, summarized at the end.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.journal.db import TradeJournal, Trade, AccountSnapshot, OpenPosition
from bot.risk.circuit_breaker import RiskManager
from bot.broker.ibkr_client import IBKRClient

results = {}

journal = TradeJournal()

# ============================================================
# CHECK 1 -- trade journal entry/exit price integrity
# ============================================================
print("=" * 70)
print("CHECK 1: Trade journal entry/exit price integrity")
print("=" * 70)
session = journal.Session()
try:
    all_trades = session.query(Trade).all()
    bad_trades = [t for t in all_trades if t.entry_price == t.exit_price]
finally:
    session.close()

print(f"Total trades on record: {len(all_trades)}")
print(f"Trades with entry_price == exit_price (the original fake-data bug): {len(bad_trades)}")
for t in bad_trades[:10]:
    print(f"  FLAGGED id={t.id} {t.symbol} entry={t.entry_price} exit={t.exit_price} pnl={t.pnl}")
results["1. Entry/exit price integrity"] = (len(bad_trades) == 0)


# ============================================================
# CHECK 2 -- account snapshot history
# ============================================================
print()
print("=" * 70)
print("CHECK 2: Account snapshot history")
print("=" * 70)
session = journal.Session()
try:
    snapshots = session.query(AccountSnapshot).order_by(AccountSnapshot.timestamp).all()
finally:
    session.close()

print(f"Snapshot rows: {len(snapshots)}")
if not snapshots:
    print("No snapshots yet -- run ops/daily_snapshot.py.")
    results["2. Account snapshots exist & advancing"] = False
else:
    timestamps = [s.timestamp for s in snapshots]
    advancing = all(timestamps[i] <= timestamps[i + 1] for i in range(len(timestamps) - 1))
    print(f"Earliest: {timestamps[0]}   Latest: {timestamps[-1]}")
    print(f"Timestamps monotonically advancing: {advancing}")
    results["2. Account snapshots exist & advancing"] = advancing


# ============================================================
# CHECK 3 -- risk state persistence
# ============================================================
print()
print("=" * 70)
print("CHECK 3: Risk state persists across restarts")
print("=" * 70)
state = journal.load_risk_state()
if state is None:
    print("No risk_state row found -- has reset_risk_baseline.py ever run?")
    results["3. Risk state persistence"] = False
else:
    print(f"day = {state['day']}")
    print(f"start_of_day_equity = {state['start_of_day_equity']}")
    print(f"peak_equity = {state['peak_equity']}")
    print("This process just started and read these values fresh from trading.db --")
    print("there is no in-memory cache to fall back on, so real numbers here IS the proof")
    print("this state survives a process restart.")
    results["3. Risk state persistence"] = state["peak_equity"] is not None


# ============================================================
# CHECK 4 -- circuit breaker drawdown logic (in-memory only)
# ============================================================
print()
print("=" * 70)
print("CHECK 4: Circuit breaker drawdown logic (in-memory test, DB untouched)")
print("=" * 70)

test_journal = TradeJournal()
risk = RiskManager(test_journal)
risk._persist = lambda: None   # hard guarantee: nothing below can write to the DB

real_peak = risk.peak_equity
real_sod = risk.start_of_day_equity
print(f"Real stored peak_equity: {real_peak}")
print(f"Real stored start_of_day_equity: {real_sod}")

if real_peak is None:
    print("No peak_equity on record -- cannot exercise the drawdown branch.")
    results["4. Circuit breaker drawdown logic"] = False
else:
    fake_equity = real_peak * 0.75   # exactly 25% below the real, stored peak
    # Isolate the DRAWDOWN check specifically: a 25% drop would also trip the
    # tighter 7.5% daily-loss check first if start_of_day_equity is close to
    # peak (as it commonly is on a fresh account). Setting start_of_day_equity
    # to the fake value neutralizes that check for this one in-memory
    # instance only, so the branch under test is unambiguous. This is never
    # persisted -- _persist is a no-op above.
    risk.start_of_day_equity = fake_equity
    print(f"Fabricated current_equity = {fake_equity:.2f} (25% below real peak {real_peak:.2f})")
    print("(start_of_day_equity temporarily set equal to the fake value, in memory only, to "
          "isolate the drawdown check from the daily-loss check for this test)")

    allowed = risk.check_trading_allowed(fake_equity)
    print(f"check_trading_allowed() returned: {allowed}")

    reloaded = journal.load_risk_state()
    untouched = (reloaded is not None and reloaded["peak_equity"] == real_peak
                 and reloaded["start_of_day_equity"] == real_sod)
    print(f"Stored state re-read after the test: peak={reloaded['peak_equity']}, "
          f"start_of_day={reloaded['start_of_day_equity']}")
    print(f"Stored state left completely untouched: {untouched}")

    results["4. Circuit breaker drawdown logic"] = (allowed is False) and untouched


# ============================================================
# CHECK 5 -- IBKR positions/orders vs local records
# ============================================================
print()
print("=" * 70)
print("CHECK 5: Open positions/orders match between IBKR and local records")
print("=" * 70)

client = IBKRClient()
connected = client.connect()
print("Connected:", connected)

if not connected:
    print("Could not connect to IB Gateway -- skipping Check 5.")
    results["5. IBKR positions/orders match local records"] = False
else:
    ib = client.ib
    ib.sleep(2)

    ibkr_positions = {p.contract.symbol: p.position for p in ib.positions()}
    print(f"IBKR positions: {ibkr_positions if ibkr_positions else '(none -- flat)'}")

    session = journal.Session()
    try:
        local_positions = {p.symbol: p.quantity for p in session.query(OpenPosition).all()}
    finally:
        session.close()
    print(f"Local open_positions table: {local_positions if local_positions else '(none)'}")

    mismatches = []
    for sym, qty in ibkr_positions.items():
        if sym not in local_positions:
            mismatches.append(f"IBKR shows {qty} {sym} but the journal has no record of it")
        elif abs(local_positions[sym] - qty) > 0.001:
            mismatches.append(f"{sym}: IBKR reports {qty}, journal reports {local_positions[sym]}")
    for sym, qty in local_positions.items():
        if sym not in ibkr_positions:
            mismatches.append(f"Journal shows an open {sym} position but IBKR reports none")

    open_orders = ib.openTrades()
    print(f"Open orders at IBKR: {len(open_orders)}")
    for t in open_orders:
        print(f"  {t.order.action} {t.order.totalQuantity} {t.contract.symbol} "
              f"({t.order.orderType}, status={t.orderStatus.status})")
    if open_orders:
        mismatches.append(f"{len(open_orders)} unexpected open order(s) resting at IBKR "
                           f"(this system uses no resting stops/targets for buy-and-hold)")

    if mismatches:
        print("Mismatches found:")
        for m in mismatches:
            print(f"  - {m}")
    else:
        print("No mismatches -- IBKR and local records agree.")

    results["5. IBKR positions/orders match local records"] = (len(mismatches) == 0)
    client.disconnect()


# ============================================================
# SUMMARY
# ============================================================
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
for check, passed in results.items():
    print(f"  {'PASS' if passed else 'FAIL':<6} {check}")

n_pass = sum(results.values())
print(f"\n{n_pass}/{len(results)} checks passed.")
