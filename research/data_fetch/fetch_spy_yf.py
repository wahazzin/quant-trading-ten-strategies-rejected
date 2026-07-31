"""
fetch_spy_yf.py -- SPY benchmark history via yfinance, same source and
date range as fetch_yf_universe.py, for factor_test.py's benchmark
comparison and alpha test (project rule 5). Kept separate from the
earlier IBKR-sourced data/SPY_daily.csv (different source, different
date range) to avoid mixing the two.
"""
import os
import pandas as pd
import yfinance as yf

START = "2006-01-01"
END = "2026-07-01"
OUT_PATH = os.path.join("data", "spy_yf.parquet")

df = yf.download("SPY", start=START, end=END, auto_adjust=True, progress=False)
df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
df = df.reset_index().rename(columns={"Date": "date", "Open": "open", "High": "high",
                                        "Low": "low", "Close": "close", "Volume": "volume"})
df = df[["date", "open", "high", "low", "close", "volume"]]

os.makedirs("data", exist_ok=True)
df.to_parquet(OUT_PATH, index=False)

print(f"SPY rows: {len(df)}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Saved to {OUT_PATH}")
