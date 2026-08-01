"""
value_report.py -- Phase 6 forward test, Task 3: monthly tracking report
for the value portfolio, reusing weekly_report.py's structure (local-DB
trade/equity stats + a live IBKR fetch for the benchmark only).

Reports, since inception (data/value_portfolio_state.json, written by
ops/value_rebalance.py on its first run):
  - portfolio value now vs at inception, total and annualized return
  - SPY's return over the identical window, fetched live from IBKR --
    benchmark only, never traded (PRIIPs blocks buying it on this account)
  - the difference (portfolio - SPY)
  - current holdings with individual P&L (entry price from the trade
    journal's OpenPosition table, i.e. the REAL fill price TradeMonitor
    recorded, not the plan price from value_portfolio.py)
  - a running count of monthly observations, with a note on how many
    are needed for statistical confidence at this alpha/noise level
    (~192, per fundamental_test.py's power analysis on the ~5.5%
    backtest alpha) -- we start at zero and this counter is the only
    thing that makes the forward test's evidence accumulate honestly.

Each run appends at most one row to data/value_portfolio_monthly_log.csv
for the CURRENT calendar month (idempotent -- re-running mid-month
updates that month's row rather than creating a duplicate observation).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
from datetime import datetime, timezone
from ib_async import Stock

from bot.broker.ibkr_client import IBKRClient
from bot.broker.fx import FXConverter
from bot.journal.db import TradeJournal

STATE_PATH = os.path.join("data", "value_portfolio_state.json")
HOLDINGS_PATH = os.path.join("data", "value_portfolio_previous.csv")
MONTHLY_LOG_PATH = os.path.join("data", "value_portfolio_monthly_log.csv")
INSTRUMENT_CURRENCY = "USD"
BENCHMARK_SYMBOL = "SPY"
REQUIRED_OBSERVATIONS = 192  # power analysis, fundamental_test.py / RESEARCH_LOG.md Test 10

if not os.path.exists(STATE_PATH):
    raise SystemExit(f"{STATE_PATH} not found -- run ops/value_rebalance.py first (it writes this "
                      f"on its first successful run, i.e. inception).")
with open(STATE_PATH) as f:
    state = json.load(f)

inception_date = state["inception_date"]
inception_equity_usd = state["inception_equity_usd"]
inception_spy_price = state["inception_spy_price"]

if not os.path.exists(HOLDINGS_PATH):
    raise SystemExit(f"{HOLDINGS_PATH} not found -- run ops/value_rebalance.py first.")
holdings_target = pd.read_csv(HOLDINGS_PATH)

print("=" * 96)
print(f"VALUE PORTFOLIO REPORT -- since inception {inception_date}")
print("=" * 96)
print(f"Inception equity:            {inception_equity_usd:.2f} USD")
print(f"Inception SPY reference:     {inception_spy_price}")

# ------------------------------------------------------------------
# LIVE FETCH: current equity, current prices, SPY benchmark price
# ------------------------------------------------------------------
client = IBKRClient()
connected = client.connect()
print(f"Connected: {connected}")
if not connected:
    raise SystemExit("Could not connect to IB Gateway.")

ib = client.ib
ib.sleep(2)

equity = client.get_net_liquidation()
account_currency = client.get_account_currency()
fx = FXConverter(ib)
current_equity_usd = fx.convert(equity, account_currency, INSTRUMENT_CURRENCY)

spy_contract = Stock(BENCHMARK_SYMBOL, "SMART", "USD")
ib.qualifyContracts(spy_contract)
spy_bars = ib.reqHistoricalData(
    spy_contract, endDateTime="", durationStr="2 D", barSizeSetting="1 day",
    whatToShow="TRADES", useRTH=True, formatDate=1,
)
current_spy_price = spy_bars[-1].close if spy_bars else None

journal = TradeJournal()
holdings_rows = []
for row in holdings_target.itertuples(index=False):
    ticker = row.ticker
    pos = journal.get_open_position(ticker)
    contract = Stock(ticker, "SMART", INSTRUMENT_CURRENCY)
    try:
        ib.qualifyContracts(contract)
        bars = ib.reqHistoricalData(
            contract, endDateTime="", durationStr="2 D", barSizeSetting="1 day",
            whatToShow="TRADES", useRTH=True, formatDate=1,
        )
        live_price = bars[-1].close if bars else None
    except Exception:
        live_price = None

    if pos is None:
        holdings_rows.append({"ticker": ticker, "shares": None, "entry_price": None,
                               "live_price": live_price, "pnl_pct": None, "pnl_usd": None,
                               "note": "no open position on record"})
        continue

    shares = pos["quantity"]
    entry = pos["avg_entry_price"]
    if live_price is None:
        pnl_pct = None
        pnl_usd = None
    else:
        pnl_pct = (live_price / entry - 1) * 100
        pnl_usd = (live_price - entry) * shares
    holdings_rows.append({"ticker": ticker, "shares": shares, "entry_price": entry,
                           "live_price": live_price, "pnl_pct": pnl_pct, "pnl_usd": pnl_usd,
                           "note": ""})

client.disconnect()

# ------------------------------------------------------------------
# RETURNS SINCE INCEPTION
# ------------------------------------------------------------------
days_elapsed = (pd.Timestamp.now().normalize() - pd.Timestamp(inception_date)).days
total_return_pct = (current_equity_usd / inception_equity_usd - 1) * 100
if days_elapsed >= 1:
    annualized_pct = ((current_equity_usd / inception_equity_usd) ** (365.25 / days_elapsed) - 1) * 100
else:
    annualized_pct = float("nan")

spy_total_return_pct = None
if current_spy_price is not None and inception_spy_price:
    spy_total_return_pct = (current_spy_price / inception_spy_price - 1) * 100

print()
print("=" * 96)
print(f"PERFORMANCE SINCE INCEPTION ({days_elapsed} calendar days)")
print("=" * 96)
print(f"Portfolio value now:  {current_equity_usd:.2f} USD  (was {inception_equity_usd:.2f} USD at inception)")
print(f"Portfolio return:     {total_return_pct:+.2f}%  (annualized: {annualized_pct:+.2f}%)")
if spy_total_return_pct is not None:
    print(f"SPY return (benchmark only, not held): {spy_total_return_pct:+.2f}%  "
          f"({inception_spy_price:.2f} -> {current_spy_price:.2f})")
    print(f"Difference (portfolio - SPY):          {total_return_pct - spy_total_return_pct:+.2f} pp")
else:
    print("SPY benchmark price unavailable this run.")
if days_elapsed < 90:
    print(f"NOTE: only {days_elapsed} days elapsed -- annualized return is highly unstable this early "
          f"and should not be treated as a meaningful estimate yet.")

# ------------------------------------------------------------------
# CURRENT HOLDINGS
# ------------------------------------------------------------------
print()
print("=" * 96)
print("CURRENT HOLDINGS")
print("=" * 96)
hdf = pd.DataFrame(holdings_rows)
print(hdf.to_string(index=False))

# ------------------------------------------------------------------
# MONTHLY OBSERVATION LOG
# ------------------------------------------------------------------
this_month = pd.Timestamp.now().to_period("M")
if os.path.exists(MONTHLY_LOG_PATH):
    log_df = pd.read_csv(MONTHLY_LOG_PATH)
else:
    log_df = pd.DataFrame(columns=["month", "date", "equity_usd", "spy_price", "return_pct"])

log_df["month"] = log_df["month"].astype(str)
new_row = {
    "month": str(this_month),
    "date": datetime.now(timezone.utc).date().isoformat(),
    "equity_usd": current_equity_usd,
    "spy_price": current_spy_price,
    "return_pct": total_return_pct,
}
if str(this_month) in log_df["month"].values:
    log_df.loc[log_df["month"] == str(this_month), list(new_row.keys())] = list(new_row.values())
else:
    log_df = pd.concat([log_df, pd.DataFrame([new_row])], ignore_index=True)
log_df = log_df.sort_values("month").reset_index(drop=True)
log_df.to_csv(MONTHLY_LOG_PATH, index=False)

n_obs = len(log_df)
print()
print("=" * 96)
print("MONTHLY OBSERVATION COUNT")
print("=" * 96)
print(f"Distinct calendar months logged so far: {n_obs}")
print(f"Required for statistical confidence (t>2) at this alpha/noise level: ~{REQUIRED_OBSERVATIONS}")
if n_obs < REQUIRED_OBSERVATIONS:
    years_needed = (REQUIRED_OBSERVATIONS - n_obs) / 12
    print(f"Still need {REQUIRED_OBSERVATIONS - n_obs} more monthly observations "
          f"(~{years_needed:.1f} more years at one observation per month).")
    print("This is why the Phase 6 pre-registration commits to a multi-year horizon before "
          "any conclusion is drawn -- a handful of months proves nothing either way.")
else:
    print("Observation count has reached the pre-registered power threshold.")
print(f"Log saved to {MONTHLY_LOG_PATH}")
