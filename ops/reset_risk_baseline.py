"""
reset_risk_baseline.py -- ONE-TIME correction.

The persisted peak_equity (47042.20 SEK) was accidentally captured
during account-funding/setup chaos days ago, not from real trading.
The journal itself was broken (fake $0.00 PnL trades) until yesterday,
so nothing before now counts as real trading history anyway.

We are confirmed FLAT with a real, clean equity number. This resets
the circuit breaker's peak and daily baseline to THAT number, so
drawdown tracking starts honestly from the moment real testing begins,
instead of comparing against contaminated setup-era noise.

Run ONCE. Never again.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.broker.ibkr_client import IBKRClient
from bot.journal.db import TradeJournal
from datetime import date

client = IBKRClient()
connected = client.connect()
print("Connected:", connected)

if connected:
    equity = client.get_net_liquidation()
    print(f"Current real equity: {equity:.2f}")

    journal = TradeJournal()
    old = journal.load_risk_state()
    print(f"Old baseline: {old}")

    journal.save_risk_state(
        day=date.today(),
        start_of_day_equity=equity,
        peak_equity=equity,
    )
    print(f"New baseline set: peak={equity:.2f}, start_of_day={equity:.2f}")

    client.disconnect()
