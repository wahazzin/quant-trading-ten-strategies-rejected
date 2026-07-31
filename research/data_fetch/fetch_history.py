"""
fetch_history.py -- Step 1 of the backtest engine.

Before we can test whether any signal works, we need real historical
price data. This has never been tested in this project -- every prior
script requested LIVE prices, never HISTORICAL ones. New API call,
new chance for IBKR to surprise us, so we test it alone first.

Pulls 2 years of daily bars for F and saves them to data/F_daily.csv
so we only have to fetch once, not every time we test an idea.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bot.broker.ibkr_client import IBKRClient
from ib_async import Stock
import pandas as pd

SYMBOL = "F"
DURATION = "2 Y"      # 2 years of daily bars: enough history to
                       # later split into a tune window and a
                       # holdout window we never peek at early
BAR_SIZE = "1 day"

client = IBKRClient()
connected = client.connect()
print("Connected:", connected)

if connected:
    ib = client.ib
    contract = Stock(SYMBOL, "SMART", "USD")
    ib.qualifyContracts(contract)

    print(f"Requesting {DURATION} of {BAR_SIZE} bars for {SYMBOL}...")
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=DURATION,
        barSizeSetting=BAR_SIZE,
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1,
    )

    if not bars:
        print("⚠️ No bars returned. Possible causes: no historical "
              "data permission, or a bad duration/bar size string.")
    else:
        df = pd.DataFrame([{
            "date": b.date,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
        } for b in bars])

        print(f"✅ Got {len(df)} daily bars.")
        print("First 3 rows:")
        print(df.head(3).to_string(index=False))
        print("Last 3 rows:")
        print(df.tail(3).to_string(index=False))

        os.makedirs("data", exist_ok=True)
        out_path = os.path.join("data", "F_daily.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved to {out_path}")

client.disconnect()
