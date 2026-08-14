"""
expand_universe.py -- expand the test universe from 13 stocks to a much
larger candidate pool, to increase statistical power for the reversal
tests that keep coming back underpowered (5 tests rejected on 13 stocks;
diagnosis is sample size, not signal choice).

Mechanics mirror verify_universe.py exactly: no opinions, no
performance-based selection. The broker's own data decides pass/fail on:
  - price in $5-$50
  - 20-day average daily volume > 1,000,000 shares
  - at least 750 daily bars available (3 years of daily bars requested)

Only stocks that PASS ALL THREE checks get their data saved to
data/<TICKER>_daily.csv. A full per-ticker pass/fail record (mirroring
universe_verification.csv's format) is written to data/universe_expanded.csv.

CANDIDATES below is a static list of ~126 liquid US small/mid-cap
tickers, constructed from memory across a range of sectors (regional
banks, REITs, energy E&P, materials/chemicals, retail/restaurants,
industrials, tech/comms equipment, apparel, utilities, auto parts,
specialty finance, homebuilders, miners, airlines, healthcare, telecom).
It was NOT built or pruned using any past-return or performance data --
that is exactly the bias this project is trying to avoid. Some tickers
may fail to qualify (renamed, delisted, or misremembered) -- those are
just recorded as errors and skipped, same as verify_universe.py does.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import time
import pandas as pd
from ib_async import Stock
from bot.broker.ibkr_client import IBKRClient
from bot.broker.guard import require_broker

require_broker("ibkr")

CANDIDATES = [
    # regional banks
    "VLY", "WAL", "ZION", "CMA", "FHN", "SNV", "PB", "CFR", "HOMB", "UMBF",
    "FFIN", "BANF", "PNFP", "WSBC", "ONB",
    # REITs
    "SBRA", "OHI", "DOC", "NHI", "UMH", "IRT", "PK", "RHP", "XHR", "INN",
    "SVC", "APLE",
    # energy E&P / midstream
    "AROC", "MTDR", "CIVI", "MGY", "MUR", "PR", "RRC", "CTRA", "SM", "WKC",
    "CPE", "NOG",
    # materials / chemicals
    "HUN", "CLF", "TROX", "X", "ATI", "CENX", "KRO", "WLK", "OLN", "ASH",
    "NX", "CMP",
    # retail / restaurants
    "WEN", "KSS", "DENN", "JACK", "CAKE", "BJRI", "RUTH", "GPS", "ANF",
    "CHS", "BKE", "CATO",
    # industrials
    "KMT", "ATKR", "TRN", "WOR", "ASTE", "GBX", "NPK", "MLKN",
    # tech / comms equipment
    "MGNI", "PLAB", "SANM", "PLXS", "BHE", "CEVA", "DGII", "VIAV",
    # apparel
    "UAA", "COLM", "GIII", "DXLG", "CRI",
    # utilities
    "SWX", "NWN", "MGEE", "OGS", "CPK", "ALE", "AVA",
    # auto parts / industrial components
    "VSH", "MOD", "SUP", "SMP", "DORM",
    # specialty finance / insurance
    "SLM", "NAVI", "OMF", "RDN", "ESNT", "MTG", "NMIH",
    # homebuilders
    "TPH", "GRBK", "CVCO", "IBP", "CSWI",
    # miners
    "HL", "CDE", "EGO", "AG", "MUX",
    # airlines
    "JBLU", "ALK", "HA",
    # healthcare / biotech
    "RCUS", "RIGL", "MIRM", "SUPN", "PRGO", "ITCI",
    # telecom
    "TDS", "USM", "ATNI", "LUMN",
]

PRICE_MIN, PRICE_MAX = 5.0, 50.0
MIN_ADV = 1_000_000
MIN_BARS = 750
DURATION = "3 Y"

client = IBKRClient()
connected = client.connect()
print("Connected:", connected)
if not connected:
    raise SystemExit("Could not connect to IB Gateway.")

ib = client.ib
os.makedirs("data", exist_ok=True)

results = []
for i, ticker in enumerate(CANDIDATES):
    try:
        contract = Stock(ticker, "SMART", "USD")
        ib.qualifyContracts(contract)
        bars = ib.reqHistoricalData(
            contract, endDateTime="", durationStr=DURATION,
            barSizeSetting="1 day", whatToShow="TRADES",
            useRTH=True, formatDate=1,
        )
        if not bars:
            print(f"[{i+1}/{len(CANDIDATES)}] {ticker:6s} FAIL: no bars returned")
            results.append({"ticker": ticker, "status": "FAIL", "reason": "no bars returned"})
            time.sleep(2)
            continue

        df = pd.DataFrame([{"date": b.date, "open": b.open, "high": b.high,
                            "low": b.low, "close": b.close,
                            "volume": b.volume} for b in bars])
        df = df.iloc[:-1]                       # drop today's partial bar
        n_bars = len(df)

        if n_bars == 0:
            print(f"[{i+1}/{len(CANDIDATES)}] {ticker:6s} FAIL: no bars after dropping partial")
            results.append({"ticker": ticker, "status": "FAIL", "reason": "no bars after dropping partial"})
            time.sleep(2)
            continue

        price = float(df["close"].iloc[-1])
        adv = float(df["volume"].tail(20).mean())

        fails = []
        if n_bars < MIN_BARS:
            fails.append(f"bars {n_bars}<{MIN_BARS}")
        if not (PRICE_MIN <= price <= PRICE_MAX):
            fails.append(f"price ${price:.2f}")
        if adv < MIN_ADV:
            fails.append(f"ADV {adv:,.0f}")

        status = "FAIL" if fails else "PASS"
        if status == "PASS":
            df.to_csv(os.path.join("data", f"{ticker}_daily.csv"), index=False)

        results.append({
            "ticker": ticker, "price": price, "adv": adv, "bars": n_bars,
            "status": status, "reason": ", ".join(fails),
        })
        print(f"[{i+1}/{len(CANDIDATES)}] {ticker:6s} ${price:7.2f}  ADV {adv:>13,.0f}  "
              f"bars {n_bars:>4}  {'FAIL: ' + ', '.join(fails) if fails else 'PASS'}")
        time.sleep(2)
    except Exception as e:
        print(f"[{i+1}/{len(CANDIDATES)}] {ticker:6s} ERROR: {e}")
        results.append({"ticker": ticker, "status": "ERROR", "reason": str(e)[:80]})
        time.sleep(2)

client.disconnect()

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join("data", "universe_expanded.csv"), index=False)

counts = results_df["status"].value_counts()
print()
print("=" * 50)
print("Pass/fail summary")
print("=" * 50)
for status in ["PASS", "FAIL", "ERROR"]:
    print(f"{status:8s}: {int(counts.get(status, 0))}")
print(f"Total candidates: {len(CANDIDATES)}")
print("Saved data/universe_expanded.csv")
