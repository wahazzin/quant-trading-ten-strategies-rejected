"""
fetch_spy.py -- standalone historical fetch for the SPY benchmark,
needed for beta_test.py (project rule 5's benchmark comparison).
Modeled directly on fetch_history.py; does not modify it.

Pulls 3 years of daily bars for SPY and saves them to data/SPY_daily.csv.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bot.broker.ibkr_client import IBKRClient
from ib_async import Stock
import pandas as pd
import datetime as dt

SYMBOL = "SPY"
DURATION = "3 Y"
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
        print("No bars returned. Possible causes: no historical "
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

        # Drop a final in-progress bar for today's still-open session, if
        # present, so only fully completed daily bars are saved.
        today = dt.date.today()
        if len(df) and pd.to_datetime(df["date"].iloc[-1]).date() >= today:
            df = df.iloc[:-1].reset_index(drop=True)
            print("Dropped final partial bar for today's in-progress session.")

        print(f"Got {len(df)} daily bars.")
        print("First 3 rows:")
        print(df.head(3).to_string(index=False))
        print("Last 3 rows:")
        print(df.tail(3).to_string(index=False))

        os.makedirs("data", exist_ok=True)
        out_path = os.path.join("data", "SPY_daily.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved to {out_path}")

client.disconnect()
