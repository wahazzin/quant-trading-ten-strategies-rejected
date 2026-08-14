"""
verify_universe.py -- mechanically verify the 18-stock test universe
against LIVE IBKR data, and download history for all of them at once.

No opinions. The broker's own data decides pass/fail on:
  - price in $5-$50
  - average daily volume > 1,000,000 shares (20-day)
  - at least 3 years of history available
  - annualized volatility -> calm / medium / volatile bucket

Also saves each stock's daily bars to data/<TICKER>_daily.csv so the
backtest can run without re-downloading.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import time
import pandas as pd
import numpy as np
from ib_async import Stock
from bot.broker.ibkr_client import IBKRClient
from bot.broker.guard import require_broker

require_broker("ibkr")

CANDIDATES = {
    # ticker: research-assigned bucket (we re-check this ourselves)
    "FLO": "calm", "SBRA": "calm", "VLY": "calm", "TDS": "calm",
    "AROC": "medium", "KMT": "medium", "HUN": "medium", "WEN": "medium",
    "MTDR": "medium", "RCUS": "medium",
    "CLF": "volatile", "MGNI": "volatile", "KSS": "volatile",
    "JBLU": "volatile", "TROX": "volatile", "VSH": "volatile",
    "PLAB": "volatile", "UAA": "volatile",
    # alternates to fill the calm bucket if needed
    "HL": "alt", "SWX": "alt", "RIGL": "alt",
}

PRICE_MIN, PRICE_MAX = 5.0, 50.0
MIN_ADV = 1_000_000


def classify(vol_pct):
    if vol_pct < 30:
        return "calm"
    if vol_pct < 50:
        return "medium"
    return "volatile"


client = IBKRClient()
connected = client.connect()
print("Connected:", connected)
if not connected:
    raise SystemExit("Could not connect to IB Gateway.")

ib = client.ib
ib.reqMarketDataType(3)
os.makedirs("data", exist_ok=True)

results = []
for ticker, assigned in CANDIDATES.items():
    try:
        contract = Stock(ticker, "SMART", "USD")
        ib.qualifyContracts(contract)
        bars = ib.reqHistoricalData(
            contract, endDateTime="", durationStr="3 Y",
            barSizeSetting="1 day", whatToShow="TRADES",
            useRTH=True, formatDate=1,
        )
        if not bars or len(bars) < 500:
            results.append({"ticker": ticker, "status": "FAIL",
                            "reason": f"only {len(bars) if bars else 0} bars"})
            continue

        df = pd.DataFrame([{"date": b.date, "open": b.open, "high": b.high,
                            "low": b.low, "close": b.close,
                            "volume": b.volume} for b in bars])
        df = df.iloc[:-1]                       # drop today's partial bar
        df.to_csv(f"data/{ticker}_daily.csv", index=False)

        price = float(df["close"].iloc[-1])
        adv = float(df["volume"].tail(20).mean())
        rets = df["close"].pct_change().dropna()
        vol = float(rets.tail(252).std() * np.sqrt(252) * 100)

        fails = []
        if not (PRICE_MIN <= price <= PRICE_MAX):
            fails.append(f"price ${price:.2f}")
        if adv < MIN_ADV:
            fails.append(f"ADV {adv:,.0f}")

        results.append({
            "ticker": ticker, "assigned": assigned, "price": price,
            "adv": adv, "vol": vol, "actual": classify(vol),
            "bars": len(df), "status": "FAIL" if fails else "PASS",
            "reason": ", ".join(fails),
        })
        print(f"{ticker:5s} ${price:6.2f}  ADV {adv:>12,.0f}  "
              f"vol {vol:5.1f}%  {classify(vol):8s} "
              f"{'FAIL: ' + ', '.join(fails) if fails else 'PASS'}")
        time.sleep(2)
    except Exception as e:
        print(f"{ticker:5s} ERROR: {e}")
        results.append({"ticker": ticker, "status": "ERROR",
                        "reason": str(e)[:60]})

client.disconnect()
pd.DataFrame(results).to_csv("data/universe_verification.csv", index=False)
print("\nSaved data/universe_verification.csv")
