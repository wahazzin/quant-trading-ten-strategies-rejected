# Legacy IBKR scripts — retired 2026-08-03

Everything in this folder is retired. **Alpaca is the single source of truth
for this project's live forward test** (`ops/value_portfolio.py`,
`ops/value_rebalance.py`, `ops/value_report.py`, `ops/event_monitor.py`, all
at the top level of `ops/`) — nothing here should ever be run as part of
normal operation again.

## Why this folder exists

The forward test moved from IBKR to Alpaca on 2026-08-02 (see
`RESEARCH_LOG.md`'s Phase 6 section and `ROADMAP.md` constraint C1). The 20
GTC value-portfolio orders already resting at IBKR at that point were left
in place rather than cancelled — Gateway was down, and the account is a
paper account, so it seemed harmless. It wasn't: those orders filled on
2026-08-03, *after* the identical 20-stock portfolio had already been placed
and filled on Alpaca, producing duplicate real exposure to the same strategy
at two brokers simultaneously. That incident — the fifth broker-state
reconciliation problem this project has hit — is why IBKR is now closed out
completely rather than merely "not used going forward."

## What's here

Every script in this folder connects to IBKR (directly or via
`bot/broker/ibkr_client.py`, `bot/broker/execution.py`, `bot/broker/fx.py`,
or `bot/broker/trade_monitor.py`, which are themselves kept only because
these scripts still import them). Roughly two eras:

- **Pre-Phase-6 infrastructure/diagnostics** (`main.py`, `cleanup.py`,
  `flatten.py`, `daily_snapshot.py`, `weekly_report.py`,
  `reset_risk_baseline.py`, `verify_system.py`, `audit.py`,
  `benchmark_buy.py`) — from when the project traded a single F position and
  later a SPY buy-and-hold benchmark directly on IBKR, before the value
  portfolio and the Alpaca migration existed.
- **Pre-Phase-6 data fetchers** (`fetch_history.py`, `fetch_spy.py`,
  `verify_universe.py`, `expand_universe.py`) — one-off historical pulls for
  the 13/36-stock reversal-test universes (Tests 1–6), superseded by direct
  yfinance pulls for every test since.
- **`close_out.py`** — the final IBKR operation, added 2026-08-03: cancel
  every open order, flatten every position, verify zero/zero. This is the
  only script here that was ever meant to run again after the migration, and
  it has now done its one job.

## The guard

Every script in this folder calls `bot/broker/guard.py`'s
`require_broker("ibkr")` before doing anything else. `.env`'s
`ACTIVE_BROKER` is `alpaca` and stays that way — so by default, **every
script in this folder refuses to run**, printing exactly why. Confirmed
directly: running any of them under the default `.env` prints

```
REFUSING TO RUN: this script is written for broker 'ibkr', but ACTIVE_BROKER
in .env is 'alpaca'. If you really mean to run this script, set
ACTIVE_BROKER=ibkr in .env first -- otherwise it is about to talk to the
wrong broker, which is exactly the incident this check exists to prevent.
```

If one of these ever genuinely needs to run again (it shouldn't), override
`ACTIVE_BROKER` for that single invocation only — don't change `.env`:

```
ACTIVE_BROKER=ibkr python ops/legacy_ibkr/close_out.py
```

No script outside this folder imports `bot/broker/ibkr_client.py` or
`ib_async` — confirmed by grep across the repo as part of this migration.
