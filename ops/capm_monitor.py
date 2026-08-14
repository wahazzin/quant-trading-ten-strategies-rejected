"""
capm_monitor.py -- Phase 6 forward test #3: the legacy CAPM/Jensen's-alpha
paper accounts (Test 11's US retest, Test 12's Swedish retest), now tracked
as a real, pre-registered forward test rather than a one-off benchmark
check. See RESEARCH_LOG.md's Phase 6 section for the pre-registration
(baseline numbers, hypothesis, success criterion) written BEFORE this
script's first log write.

LOG ONLY. This script places no orders and changes no positions -- both
portfolios are held exactly as they already are, with no rebalancing, ever.
That is deliberate: this measures whether the ALREADY-CHOSEN positions
persist or mean-revert relative to SPY, not an active strategy.

Two portfolios, two SEPARATE Alpaca paper accounts (different API keys from
the value-portfolio account, which is why this script -- unlike
value_rebalance.py/value_report.py -- explicitly passes api_key/api_secret
to AlpacaClient instead of relying on its APCA_API_KEY_ID/SECRET_KEY
default):
  US CAPM: NVDA, AVGO, LLY, WMT, XOM, GOOGL  (CAPM_US_KEY_ID/CAPM_US_SECRET_KEY)
  Swedish CAPM: SPOT, ERIC, AZN, ALV, OTLY   (CAPM_SE_KEY_ID/CAPM_SE_SECRET_KEY)

Neither account has a local trade journal entry (these were paper-traded
outside this codebase's TradeJournal) -- cost basis, market value, and
per-position P&L are read directly from Alpaca's own position objects
(AlpacaClient.get_positions_raw()) rather than re-derived.

Inception is FIXED at the pre-registered 2026-08-03 baseline (see
RESEARCH_LOG.md) -- not re-derived from live data on first run, because
that historical snapshot already happened and is what was pre-registered.
data/capm_forward_state.json is written once, on first run, with exactly
those numbers, and never changes after that.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
from datetime import datetime, timezone

from bot.broker.alpaca_client import AlpacaClient
from bot.broker.guard import require_broker

require_broker("alpaca")

STATE_PATH = os.path.join("data", "capm_forward_state.json")
LOG_PATH = os.path.join("data", "capm_forward_log.parquet")
BENCHMARK_SYMBOL = "SPY"
FORWARD_WINDOW_MONTHS = 12

PORTFOLIOS = {
    "US": {
        "label": "US CAPM (Test 11 retest)",
        "tickers": ["NVDA", "AVGO", "LLY", "WMT", "XOM", "GOOGL"],
        "key_env": "CAPM_US_KEY_ID",
        "secret_env": "CAPM_US_SECRET_KEY",
    },
    "SE": {
        "label": "Swedish CAPM (Test 12 retest)",
        "tickers": ["SPOT", "ERIC", "AZN", "ALV", "OTLY"],
        "key_env": "CAPM_SE_KEY_ID",
        "secret_env": "CAPM_SE_SECRET_KEY",
    },
}

# Pre-registered baseline (RESEARCH_LOG.md, Phase 6) -- fixed as of 2026-08-03,
# never recomputed. This is a HISTORICAL snapshot already taken, not something
# to re-derive live on first run.
INCEPTION_DATE = "2026-08-03"
INCEPTION = {
    "US": {"cost_basis": 101545.21, "market_value": 111007.50},
    "SE": {"cost_basis": 100076.79, "market_value": 106139.14},
}
INCEPTION_SPY_PRICE = 750.84

print("=" * 96)
print("CAPM ACCOUNTS -- FORWARD TEST STATUS")
print("LOG ONLY -- no orders placed, no positions changed, no rebalancing, ever")
print("=" * 96)

# ============================================================
# CREDENTIALS -- fail loudly and specifically if either pair is missing
# ============================================================
missing = []
for key, spec in PORTFOLIOS.items():
    if not os.environ.get(spec["key_env"]):
        missing.append(spec["key_env"])
    if not os.environ.get(spec["secret_env"]):
        missing.append(spec["secret_env"])
if missing:
    raise SystemExit(
        "Missing CAPM account credentials in .env: " + ", ".join(missing) + ". "
        "These are separate Alpaca paper accounts from the value-portfolio account "
        "(APCA_API_KEY_ID/APCA_API_SECRET_KEY) -- add the CAPM_US_* and CAPM_SE_* "
        "key pairs to .env before running this script."
    )

# ============================================================
# INCEPTION STATE -- fixed once, on first run, to the pre-registered numbers
# ============================================================
if os.path.exists(STATE_PATH):
    with open(STATE_PATH) as f:
        state = json.load(f)
    print(f"Inception (fixed, pre-registered): {state['inception_date']}")
else:
    state = {
        "inception_date": INCEPTION_DATE,
        "inception_spy_price": INCEPTION_SPY_PRICE,
        "portfolios": INCEPTION,
    }
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)
    print(f"FIRST RUN -- inception fixed at the pre-registered {INCEPTION_DATE} baseline. "
          f"This must match RESEARCH_LOG.md's pre-registration entry.")

inception_date = state["inception_date"]
inception_spy_price = state["inception_spy_price"]
inception_by_portfolio = state["portfolios"]

# ============================================================
# SPY BENCHMARK (via the default/value-portfolio account -- market data is
# account-agnostic, and this credential pair is always configured)
# ============================================================
spy_client = AlpacaClient(paper=True)
spy_client.connect()
current_spy_price = spy_client.get_last_price(BENCHMARK_SYMBOL)
spy_client.disconnect()
spy_forward_return_pct = (current_spy_price / inception_spy_price - 1) * 100
print(f"\nSPY: {inception_spy_price} ({inception_date}) -> {current_spy_price} (today) "
      f"= {spy_forward_return_pct:+.2f}%")

# ============================================================
# PER-PORTFOLIO REPORT
# ============================================================
today_str = datetime.now(timezone.utc).date().isoformat()
this_month = pd.Timestamp.now().to_period("M")
log_rows = []
summary = []

for key, spec in PORTFOLIOS.items():
    print()
    print("=" * 96)
    print(f"{spec['label']}  ({', '.join(spec['tickers'])})")
    print("=" * 96)

    client = AlpacaClient(paper=True, api_key=os.environ[spec["key_env"]],
                           api_secret=os.environ[spec["secret_env"]])
    connected = client.connect()
    if not connected:
        print(f"  Could not reach this account -- skipping {key} this run.")
        client.disconnect()
        continue

    positions = {p["symbol"]: p for p in client.get_positions_raw() if p["symbol"] in spec["tickers"]}
    client.disconnect()

    missing_tickers = [t for t in spec["tickers"] if t not in positions]
    if missing_tickers:
        print(f"  NOTE: no open position found for {missing_tickers} -- excluded from totals below "
              f"(may have been closed outside this script; this script never trades).")

    total_market_value = 0.0
    total_cost_basis = 0.0
    rows = []
    for t in spec["tickers"]:
        p = positions.get(t)
        if p is None:
            continue
        qty = float(p["qty"])
        avg_entry = float(p["avg_entry_price"])
        market_value = float(p["market_value"])
        cost_basis = float(p["cost_basis"])
        unrealized_pl = float(p["unrealized_pl"])
        unrealized_plpc = float(p["unrealized_plpc"]) * 100
        total_market_value += market_value
        total_cost_basis += cost_basis
        rows.append({"ticker": t, "qty": qty, "avg_entry_price": avg_entry,
                      "market_value": market_value, "cost_basis": cost_basis,
                      "unrealized_pl": unrealized_pl, "unrealized_plpc": unrealized_plpc})

    print(pd.DataFrame(rows).to_string(index=False) if rows else "  (no tracked positions found)")

    inception_market_value = inception_by_portfolio[key]["market_value"]
    inception_cost_basis = inception_by_portfolio[key]["cost_basis"]
    forward_return_pct = (total_market_value / inception_market_value - 1) * 100 if inception_market_value else float("nan")
    since_entry_return_pct = (total_market_value / total_cost_basis - 1) * 100 if total_cost_basis else float("nan")
    diff_vs_spy = forward_return_pct - spy_forward_return_pct

    print(f"\n  Market value now:      {total_market_value:,.2f}  (was {inception_market_value:,.2f} at {inception_date} inception)")
    print(f"  Cost basis now:        {total_cost_basis:,.2f}  (was {inception_cost_basis:,.2f} at {inception_date} inception)")
    print(f"  FORWARD-TEST return (since {inception_date} baseline, what the success criterion measures): {forward_return_pct:+.2f}%")
    print(f"  Since-entry return (Alpaca's own unrealized P&L, since each position's original entry): {since_entry_return_pct:+.2f}%")
    print(f"  vs SPY forward return ({spy_forward_return_pct:+.2f}%): {diff_vs_spy:+.2f} pp")

    log_rows.append({
        "month": str(this_month), "date": today_str, "portfolio": key,
        "market_value": total_market_value, "cost_basis": total_cost_basis,
        "forward_return_pct": forward_return_pct, "since_entry_return_pct": since_entry_return_pct,
        "spy_price": current_spy_price, "spy_forward_return_pct": spy_forward_return_pct,
    })
    summary.append({"portfolio": key, "label": spec["label"], "forward_return_pct": forward_return_pct,
                     "diff_vs_spy": diff_vs_spy})

# ============================================================
# MONTHLY LOG -- idempotent per (month, portfolio), same pattern as value_report.py
# ============================================================
if os.path.exists(LOG_PATH):
    log_df = pd.read_parquet(LOG_PATH)
else:
    log_df = pd.DataFrame(columns=["month", "date", "portfolio", "market_value", "cost_basis",
                                    "forward_return_pct", "since_entry_return_pct",
                                    "spy_price", "spy_forward_return_pct"])

for row in log_rows:
    mask = (log_df["month"] == row["month"]) & (log_df["portfolio"] == row["portfolio"])
    if mask.any():
        for col, val in row.items():
            log_df.loc[mask, col] = val
    else:
        log_df = pd.concat([log_df, pd.DataFrame([row])], ignore_index=True)

log_df = log_df.sort_values(["month", "portfolio"]).reset_index(drop=True)
os.makedirs("data", exist_ok=True)
log_df.to_parquet(LOG_PATH, index=False)

# ============================================================
# SUCCESS-CRITERION PROGRESS (informational only -- see pre-registration)
# ============================================================
inception_ts = pd.Timestamp(inception_date)
target_date = inception_ts + pd.DateOffset(months=FORWARD_WINDOW_MONTHS)
days_elapsed = (pd.Timestamp.now().normalize() - inception_ts).days
days_remaining = max(0, (target_date - pd.Timestamp.now().normalize()).days)

print()
print("=" * 96)
print("SUMMARY")
print("=" * 96)
print("LOG ONLY -- no orders were placed, no positions were changed.")
for s in summary:
    leading = "AHEAD of" if s["diff_vs_spy"] > 0 else "BEHIND"
    print(f"  {s['label']}: {s['forward_return_pct']:+.2f}% ({leading} SPY by {abs(s['diff_vs_spy']):.2f}pp)")
print(f"\nPre-registered 12-month forward window: {inception_date} -> {target_date.date()}")
print(f"Days elapsed: {days_elapsed}   |   Days remaining: {days_remaining}")
if days_remaining > 0:
    print("NOTE: the pre-registered success criterion (both portfolios beat SPY's total return) "
          "is evaluated at the 12-month mark, not before -- this is an interim reading only.")
else:
    print("The pre-registered 12-month window has elapsed -- evaluate against the success "
          "criterion in RESEARCH_LOG.md.")
print(f"\nLog saved to {LOG_PATH}")
