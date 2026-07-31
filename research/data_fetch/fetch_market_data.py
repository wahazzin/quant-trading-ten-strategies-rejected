"""
fetch_market_data.py -- bulk US common-stock daily history, standing in
for Stooq's bulk zip download.

Stooq investigation (see chat): stooq.com now gates every request
(including the historically-documented static.stooq.com/db/h/d_us_txt.zip
bulk file and the per-ticker /q/d/l/ CSV endpoint) behind a client-side
proof-of-work bot challenge, and the static file server additionally
returns 401 Unauthorized requiring credentials we don't have. Solving
that challenge programmatically would mean defeating Stooq's own
anti-automation control specifically to automate against it, so it was
not attempted. yfinance needs no API key or account and tested cleanly
in small calibration batches -- but a first full run showed Yahoo's
rate limiting is cumulative over the session and gets progressively
worse (837 of 5494 tickers lost to YFRateLimitError in later chunks).
This version uses smaller chunks, longer pacing, and retry rounds with
increasing backoff to recover as much real coverage as possible instead
of silently accepting rate-limit noise as if it were a genuine data gap.

Universe: every NASDAQ + NYSE/AMEX ("otherlisted") symbol from NASDAQ
Trader's free, no-auth symbol directories, filtered to plain common
stock (excludes ETFs, test issues, warrants/rights/units/preferred/
depositary/notes/debentures). This is CURRENTLY LISTED tickers only --
a deliberately accepted survivorship bias (see chat): conservative for
momentum (the worst historical losers we'd short are absent, so this
understates any momentum edge) and minimal for low-volatility.

Prices are split- and dividend-adjusted (yfinance auto_adjust=True) --
needed for momentum/volatility factors to be meaningful over 20 years;
factor_test.py's data-quality pass still screens for residual bad rows.

Fetches ~2006-01-01 through today, saves the combined long-format table
to data/yfinance_us_daily.parquet with columns: ticker, date, open,
high, low, close, volume.
"""
import os
import time
import urllib.request
from io import StringIO

import pandas as pd
import yfinance as yf

START = "2006-01-01"
END = "2026-07-01"
OUT_PATH = os.path.join("data", "yfinance_us_daily.parquet")

NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

BAD_NAME_KEYWORDS = ["Warrant", "Right", "Unit", "Preferred", "Depositary", "Notes", "Debenture"]

INITIAL_CHUNK_SIZE = 250
INITIAL_SLEEP = 6
MAX_ROUNDS = 6


def fetch_text(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def build_universe():
    nas = pd.read_csv(StringIO(fetch_text(NASDAQ_URL)), sep="|")
    nas = nas[nas["Symbol"].notna()]
    nas = nas[~nas["Symbol"].astype(str).str.contains("File Creation Time", na=False)]
    oth = pd.read_csv(StringIO(fetch_text(OTHER_URL)), sep="|")
    oth = oth[oth["ACT Symbol"].notna()]
    oth = oth[~oth["ACT Symbol"].astype(str).str.contains("File Creation Time", na=False)]

    nas_common = nas[(nas["Test Issue"] == "N") & (nas["ETF"] == "N") & (nas["NextShares"] == "N")]
    oth_common = oth[(oth["Test Issue"] == "N") & (oth["ETF"] == "N")]

    def clean(df, name_col, sym_col):
        mask = ~df[name_col].astype(str).str.contains("|".join(BAD_NAME_KEYWORDS), case=False, na=False)
        return df[mask][sym_col].astype(str).tolist()

    syms = sorted(set(clean(nas_common, "Security Name", "Symbol")) |
                  set(clean(oth_common, "Security Name", "ACT Symbol")))
    return [s for s in syms if s.isalpha()]


def to_long(sub, ticker):
    sub = sub.reset_index()
    sub = sub.rename(columns={"Date": "date", "Open": "open", "High": "high",
                               "Low": "low", "Close": "close", "Volume": "volume"})
    sub["ticker"] = ticker
    return sub[["ticker", "date", "open", "high", "low", "close", "volume"]]


print("Building ticker universe from NASDAQ Trader symbol directories...", flush=True)
tickers = build_universe()
print(f"Universe: {len(tickers)} common-stock tickers", flush=True)

results = {}          # ticker -> long-format dataframe
never_available = []  # genuinely no data returned (not a rate-limit issue)

pending = list(tickers)
chunk_size = INITIAL_CHUNK_SIZE
sleep_time = INITIAL_SLEEP
round_num = 0

while pending and round_num < MAX_ROUNDS:
    round_num += 1
    n_chunks = (len(pending) + chunk_size - 1) // chunk_size
    print(f"\n=== Round {round_num}: {len(pending)} tickers pending, "
          f"chunk_size={chunk_size}, sleep={sleep_time}s, {n_chunks} chunks ===", flush=True)

    still_pending = []
    rate_limited_this_round = 0

    for c in range(n_chunks):
        chunk = pending[c * chunk_size:(c + 1) * chunk_size]
        t0 = time.time()
        try:
            df = yf.download(chunk, start=START, end=END, group_by="ticker",
                              threads=True, progress=False, auto_adjust=True)
        except Exception as e:
            print(f"  [{c+1}/{n_chunks}] chunk failed entirely ({e}); requeueing", flush=True)
            still_pending.extend(chunk)
            rate_limited_this_round += len(chunk)
            time.sleep(sleep_time)
            continue

        got_cols = set(df.columns.get_level_values(0)) if isinstance(df.columns, pd.MultiIndex) else set()
        chunk_ok = 0
        for t in chunk:
            if t in results:
                continue
            if t not in got_cols:
                still_pending.append(t)
                rate_limited_this_round += 1
                continue
            sub = df[t].dropna(how="all")
            if sub.empty:
                still_pending.append(t)
                rate_limited_this_round += 1
                continue
            results[t] = to_long(sub, t)
            chunk_ok += 1

        print(f"  [{c+1}/{n_chunks}] {time.time()-t0:.1f}s, {chunk_ok}/{len(chunk)} ok "
              f"(cumulative: {len(results)}/{len(tickers)})", flush=True)
        time.sleep(sleep_time)

    pending = still_pending
    print(f"Round {round_num} done: {len(results)} recovered so far, {len(pending)} still pending "
          f"({rate_limited_this_round} rate-limited/no-data this round)", flush=True)

    # back off harder each round, and fetch in smaller batches so a stall
    # costs less pending work
    sleep_time = min(sleep_time * 2, 30)
    chunk_size = max(20, chunk_size // 2)

if pending:
    print(f"\nGiving up on {len(pending)} tickers after {round_num} rounds "
          f"(persistent rate limiting or genuinely no data): {pending[:40]}"
          f"{'...' if len(pending) > 40 else ''}", flush=True)
    never_available = pending

print("\nCombining all recovered tickers...", flush=True)
panel = pd.concat(list(results.values()), ignore_index=True)
panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

os.makedirs("data", exist_ok=True)
panel.to_parquet(OUT_PATH, index=False)

counts = panel.groupby("ticker")["date"].agg(["min", "max", "count"])
span_years = (counts["max"] - counts["min"]).dt.days / 365.25
n_15plus = int((span_years >= 15).sum())

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total tickers with data: {panel['ticker'].nunique()}")
print(f"Total rows: {len(panel)}")
print(f"Earliest date: {panel['date'].min()}")
print(f"Latest date: {panel['date'].max()}")
print(f"Tickers with 15+ years of history: {n_15plus}")
print(f"Never recovered (rate-limited or no data) after {round_num} rounds: "
      f"{len(never_available)} of {len(tickers)} candidates")
print(f"Saved to {OUT_PATH}")
