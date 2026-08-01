"""
value_portfolio.py -- Phase 6 forward test, Task 1: generate today's
Book-to-Market value portfolio.

Ten backtested strategies are rejected (RESEARCH_LOG.md); Value (B/M) is
the one unresolved candidate -- alpha +5.28%/yr, t=1.46, underpowered
because XBRL data only reaches back to ~2010 (102 monthly observations
vs the ~192 needed for t>2 at this noise level). The backtest dataset is
fully spent (RESEARCH_LOG.md's Evidence-Boundary section) -- forward
testing from today onward is the only remaining source of uncontaminated
evidence.

Point-in-time fundamental logic -- latest_known_value(), the
annual-duration flow-fact filter, and the B/M definition itself -- is
copied verbatim from research/factor_tests/fundamental_test.py. Same
rules, just evaluated with cutoff=today instead of a historical
formation date, and with NO holdout truncation: there is nothing to
hide from ourselves when the "prediction" is about the future, not the
already-elapsed past.

Deliberate, pre-registered deviation from the backtest: TOP 20 by B/M,
not the full decile (which ran ~20-40 stocks depending on year). This
account is far too small to carry decile-count positions without
punishing commission drag -- RESEARCH_LOG.md's Test 8 measured 4.04%/yr
drag at 38 stocks vs 2.50%/yr at 20 on the same $4,700 reference size.
20 is fixed now, before seeing which 20 stocks get picked.

Ranking snapshot (universe eligibility + B/M signal) uses the LATEST
available month in data/yf_universe.parquet, exactly like every prior
test in this project -- ranking has always been on month-end data, never
a live intraday quote. Once the top names are selected, price is
refreshed live from IBKR for THOSE names only (not the full ~800-name
eligible universe) so target_shares/target_value reflect what the
account can actually transact at today.

PRIIPs note: this portfolio holds only individual US common stocks,
never ETFs -- SPY itself was rejected by IBKR (error 201) on this
account, which is exactly why SPY is tracked as a benchmark number
(ops/value_report.py) and never bought.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from ib_async import Stock

from bot.broker.ibkr_client import IBKRClient
from bot.broker.fx import FXConverter

DATA_PATH = os.path.join("data", "yf_universe.parquet")
FUND_PATH = os.path.join("data", "edgar_fundamentals.parquet")
OUT_PATH = os.path.join("data", "value_portfolio_current.csv")

PRICE_MIN, PRICE_MAX = 5.0, 100.0
MIN_ADV = 250_000
MIN_HISTORY_MONTHS = 24
MOVE_THRESHOLD = 0.80
ANNUAL_DURATION_MIN_DAYS = 340
ANNUAL_DURATION_MAX_DAYS = 386
MIN_UNIVERSE_FOR_SIGNAL = 20

TOP_N = 20
TARGET_EQUITY_PCT = 0.90
INSTRUMENT_CURRENCY = "USD"

TODAY = pd.Timestamp.now().normalize()

INSTANT_CONCEPTS = ["Assets", "StockholdersEquity", "CommonStockSharesOutstanding"]
FLOW_CONCEPTS = ["Revenues", "CostOfRevenue", "GrossProfit", "NetIncomeLoss"]


def latest_known_value(ticker_facts_by_concept, concept, cutoff):
    """Verbatim from fundamental_test.py: most recent fiscal-period value
    for `concept` whose filed date is <= cutoff."""
    sub = ticker_facts_by_concept.get(concept)
    if sub is None:
        return None
    known = sub[sub["filed"] <= cutoff]
    if known.empty:
        return None
    return float(known["val"].iloc[-1])


def value_signal(ticker_facts, cutoff, price, avg_volume):
    """Verbatim from fundamental_test.py (post data-integrity fix)."""
    be = latest_known_value(ticker_facts, "StockholdersEquity", cutoff)
    shares = latest_known_value(ticker_facts, "CommonStockSharesOutstanding", cutoff)
    if be is None or shares is None or shares <= 0 or price <= 0 or be <= 0:
        return None  # negative/zero book equity excluded -- standard B/M convention
    if shares < avg_volume:
        # Data-integrity floor: us-gaap:CommonStockSharesOutstanding is
        # unreliable for a meaningful slice of filers (Up-C/holdco
        # registrants, foreign private issuers on 20-F/6-K, redomiciled
        # entities) -- it sometimes tags a technical share class rather
        # than total float (e.g. SPG "8,000 shares" vs a real ~325M share
        # count). A share count smaller than one day's average trading
        # volume implies >100% daily turnover, which is not economically
        # plausible -- reject the artifact rather than rank on it.
        return None
    market_cap = price * shares
    return be / market_cap


# ============================================================
# LOAD (no holdout truncation -- live signal generation, not a backtest)
# ============================================================
raw = pd.read_parquet(DATA_PATH)
raw["date"] = pd.to_datetime(raw["date"])
fund = pd.read_parquet(FUND_PATH)

print("=" * 96)
print(f"VALUE PORTFOLIO GENERATION -- run date {TODAY.date()}")
print("=" * 96)
print(f"Price panel on disk spans: {raw['date'].min().date()} to {raw['date'].max().date()} "
      f"({raw['ticker'].nunique()} tickers) -- ranking snapshot uses the LATEST month in this file.")
print(f"Fundamentals table spans filed dates: {fund['filed'].min().date()} to {fund['filed'].max().date()}")

# ============================================================
# DATA QUALITY CHECKS -- verbatim from fundamental_test.py
# ============================================================
df = raw.sort_values(["ticker", "date"]).reset_index(drop=True)
n0 = len(df)
bad_price = (df[["open", "high", "low", "close"]] <= 0).any(axis=1) | df["volume"].lt(0)
df = df[~bad_price].copy()
df["prev_close"] = df.groupby("ticker")["close"].shift(1)
df["daily_ret"] = df["close"] / df["prev_close"] - 1
big_move = df["daily_ret"].abs() > MOVE_THRESHOLD
df = df[~big_move.fillna(False)].copy()
df = df.drop(columns=["prev_close", "daily_ret"])
print(f"Data quality: {n0} rows -> {len(df)} rows after price/move checks")

# ============================================================
# MONTHLY PANEL -- verbatim construction from fundamental_test.py
# ============================================================
df["month"] = df["date"].dt.to_period("M")
close_last = df.sort_values("date").groupby(["ticker", "month"])["close"].last()
vol_mean = df.groupby(["ticker", "month"])["volume"].mean()
monthly = pd.DataFrame({"close": close_last, "avg_volume": vol_mean}).reset_index()
monthly = monthly.sort_values(["ticker", "month"]).reset_index(drop=True)
monthly["hist_months"] = monthly.groupby("ticker").cumcount()
monthly["eligible"] = (
    (monthly["close"] >= PRICE_MIN) & (monthly["close"] <= PRICE_MAX) &
    (monthly["avg_volume"] > MIN_ADV) &
    (monthly["hist_months"] >= MIN_HISTORY_MONTHS)
)

latest_month = monthly["month"].max()
snapshot = monthly[(monthly["month"] == latest_month) & monthly["eligible"]].copy()
print(f"Ranking snapshot month: {latest_month}  |  eligible universe: {len(snapshot)} tickers")

# ============================================================
# POINT-IN-TIME FUNDAMENTAL SELECTION -- verbatim filtering from fundamental_test.py
# ============================================================
fund_instant = fund[fund["concept"].isin(INSTANT_CONCEPTS)].copy()
flow = fund[fund["concept"].isin(FLOW_CONCEPTS)].copy()
duration_days = (flow["end"] - flow["start"]).dt.days
is_annual = duration_days.between(ANNUAL_DURATION_MIN_DAYS, ANNUAL_DURATION_MAX_DAYS)
fund_flow_annual = flow[is_annual].copy()
fund_pit = pd.concat([fund_instant, fund_flow_annual], ignore_index=True)
fund_pit = fund_pit.sort_values(["ticker", "concept", "end"]).reset_index(drop=True)

facts_by_ticker = {}
for ticker, tdf in fund_pit.groupby("ticker"):
    facts_by_ticker[ticker] = {c: cdf for c, cdf in tdf.groupby("concept")}

signals = []
for row in snapshot.itertuples(index=False):
    ticker_facts = facts_by_ticker.get(row.ticker)
    if ticker_facts is None:
        continue
    sig = value_signal(ticker_facts, TODAY, row.close, row.avg_volume)
    if sig is not None and np.isfinite(sig):
        signals.append((row.ticker, sig, row.close, row.avg_volume))

print(f"Eligible tickers with a usable, positive B/M ratio as of today: {len(signals)}")
if len(signals) < MIN_UNIVERSE_FOR_SIGNAL:
    raise SystemExit(f"Only {len(signals)} usable candidates -- below the "
                      f"{MIN_UNIVERSE_FOR_SIGNAL}-name sanity floor. Aborting rather than "
                      f"picking a portfolio from too small a pool.")

sig_df = pd.DataFrame(signals, columns=["ticker", "bm_ratio", "panel_price", "avg_volume"])
sig_df = sig_df.sort_values("bm_ratio", ascending=False).reset_index(drop=True)

print()
print(f"Top {TOP_N} by Book-to-Market (highest = cheapest), ranked on {latest_month} panel prices:")
print(sig_df.head(TOP_N).to_string(index=False))

# ============================================================
# LIVE PRICE REFRESH + SIZING (IBKR)
# ============================================================
print()
print("=" * 96)
print("LIVE PRICING + SIZING (IBKR)")
print("=" * 96)
client = IBKRClient()
connected = client.connect()
print("Connected:", connected)
if not connected:
    raise SystemExit("Could not connect to IB Gateway -- cannot size positions without live prices/equity.")

ib = client.ib
ib.sleep(2)

equity = client.get_net_liquidation()
account_currency = client.get_account_currency()
print(f"Account equity: {equity:.2f} {account_currency}")
fx = FXConverter(ib)
equity_usd = fx.convert(equity, account_currency, INSTRUMENT_CURRENCY)
print(f"Equity in {INSTRUMENT_CURRENCY}: {equity_usd:.2f}")

target_total = equity_usd * TARGET_EQUITY_PCT
per_stock_target = target_total / TOP_N
print(f"Target: {TARGET_EQUITY_PCT*100:.0f}% of equity = {target_total:.2f} {INSTRUMENT_CURRENCY}, "
      f"equal-weighted across {TOP_N} names = {per_stock_target:.2f} {INSTRUMENT_CURRENCY} each")
print()

rows = []
skipped = []
for r in sig_df.itertuples(index=False):
    if len(rows) >= TOP_N:
        break
    contract = Stock(r.ticker, "SMART", INSTRUMENT_CURRENCY)
    try:
        ib.qualifyContracts(contract)
        bars = ib.reqHistoricalData(
            contract, endDateTime="", durationStr="2 D", barSizeSetting="1 day",
            whatToShow="TRADES", useRTH=True, formatDate=1,
        )
    except Exception as e:
        print(f"  SKIP {r.ticker}: could not qualify/fetch live price ({e})")
        skipped.append(r.ticker)
        continue
    if not bars:
        print(f"  SKIP {r.ticker}: no live bar returned")
        skipped.append(r.ticker)
        continue

    live_price = bars[-1].close
    # B/M re-derived at the live price so the reported ratio matches what
    # sizing actually used -- shares outstanding don't move intraday, only
    # price does, so this is a pure rescale of the panel-price ratio.
    ticker_facts = facts_by_ticker.get(r.ticker)
    live_bm = value_signal(ticker_facts, TODAY, live_price, r.avg_volume)
    if live_bm is None:
        print(f"  SKIP {r.ticker}: live B/M recompute failed")
        skipped.append(r.ticker)
        continue

    shares = int(per_stock_target / live_price)
    if shares <= 0:
        print(f"  SKIP {r.ticker}: live price {live_price:.2f} too high for the per-stock budget")
        skipped.append(r.ticker)
        continue

    rows.append({
        "ticker": r.ticker, "bm_ratio": live_bm, "price": live_price,
        "target_shares": shares, "target_value": shares * live_price,
    })
    print(f"  OK   {r.ticker}: live price {live_price:.2f}, B/M {live_bm:.3f}, "
          f"{shares} shares = {shares * live_price:.2f} {INSTRUMENT_CURRENCY}")

client.disconnect()

if skipped:
    print(f"\n{len(skipped)} candidate(s) skipped on live pricing, backfilled from the next-ranked "
          f"B/M names to still reach {TOP_N}: {skipped}")
if len(rows) < TOP_N:
    print(f"\nWARNING: only {len(rows)}/{TOP_N} target names could be priced live -- "
          f"the ranked candidate list was exhausted. Proceeding with {len(rows)} names.")

out_df = pd.DataFrame(rows).sort_values("bm_ratio", ascending=False).reset_index(drop=True)
out_df.to_csv(OUT_PATH, index=False)

print()
print("=" * 96)
print(f"FINAL TARGET PORTFOLIO ({len(out_df)} names) -- saved to {OUT_PATH}")
print("=" * 96)
print(out_df.to_string(index=False))
print(f"\nTotal target value: {out_df['target_value'].sum():.2f} {INSTRUMENT_CURRENCY} "
      f"({out_df['target_value'].sum() / equity_usd * 100:.1f}% of account equity)")
