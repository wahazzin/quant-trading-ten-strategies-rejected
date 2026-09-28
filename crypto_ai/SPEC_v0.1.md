# crypto_ai v0.1 — Technical Specification

Companion to `PREREGISTRATION.md` (the rules of the experiment). This file is the engineering
design. Parameters in tables here mirror `experiment.json`, which is the source of truth.

## 1. Principles

1. **Paper-only by construction.** No broker/exchange-order code exists in this package. A test
   asserts nothing here imports `bot.broker`.
2. **The AI proposes, the risk engine disposes.** The LLM can never size around, skip, or
   override a rule. It sees the rules in its prompt so it can plan, but enforcement is code.
3. **Everything that decides is logged; everything logged is append-only.**
4. **Every arm sees the identical snapshot each cycle.** Fair comparison is structural.
5. **Confidence is logged, never read by any code path that trades.**

## 2. Architecture

```
 Coinbase public API ──► market_data ──► features ──┐
                                                    ▼
   state files ◄──── runner (one cycle, run every 6h) ────► journal (append-only JSONL)
        │                    │
        │    ┌───────────────┼──────────────────────────────┐
        │    ▼               ▼                              ▼
        │  baselines     AI trader ──► proposals ──►   RISK ENGINE ──► approved/clipped/rejected
        │ (deterministic) (LLM + theses)                  │
        │    │                                            ▼
        └────┴──────────────────────────────────►  PAPER EXCHANGE (fee+spread+slippage)
                                                            │
                                            analytics ◄─────┘  (metrics, IC, report)
```

## 3. Folder structure

```
crypto_ai/
  PREREGISTRATION.md  SPEC_v0.1.md  experiment.json   # the locked parameters
  prompts/trader_system_v1.md                          # hashed into the lock
  lock.py            # create + verify the spec lock
  journal.py         # append-only JSONL writer, atomic state writes
  market_data/       # coinbase.py (API), features.py (candles -> features)
  portfolio/         # state.py (accounting), baselines.py (btc/eth/ew/trend policies)
  execution/         # paper_exchange.py (fills, costs, stop replay)
  risk/              # engine.py (deterministic rules R1-R14)
  agents/            # schemas.py (validate AI output), llm.py (providers), trader.py
  analytics/         # metrics.py (Sharpe, IC, Newey-West...), report.py (CLI)
  runner.py          # orchestrates one cycle
tests/test_crypto_ai_*.py                              # unittest, matches existing tests
```
Deliberately *not* created yet (empty folders are procrastination): `backtesting/`, `models/`,
`api/`, `dashboard/`, `database/`. Naming note: code lives in `market_data/`, not `data/`,
so it can't be confused with the repo's gitignored `data/` storage folder.

## 4. Reuse vs new (from the equities engine)

| Existing | Decision | Why |
|---|---|---|
| `bot/monitor/discord_notify.py` | **Reuse** | Breaker/fault alerts. Silence-by-default: no per-fill pings |
| Pre-registration, lock, sealed-window discipline | **Reuse (process)** | The methodology is the durable asset |
| IC / alpha regression / Newey-West (Tests 3, 13, 15) | **Reuse (method), reimplement** | Research scripts execute on import; small pure functions are safer |
| `bot/risk/circuit_breaker.py` | **Rewrite** | Tied to IBKR/SEK/journal snapshots and market hours |
| `bot/journal/db.py` | **Do not reuse** | `OpenPosition.symbol` is globally unique (cross-account collision class of bug); binary SQLite can't be committed cleanly |
| `bot/broker/*` (Alpaca, reconcile, guard) | **Do not reuse** | No orders exist here. Instant simulated fills need no pending-order reconciliation |
| `combiner`, `capacity_gate`, `orchestrator`, `screener`, `signals` | **Later (v0.3+)** | Only relevant once ≥2 signals are independently proven (Phase 4 rule). Capacity lesson (Test 19) is carried by the turnover/liquidity rules |
| Market-hours assumptions | **New** | Crypto is 24/7; snapshots use UTC slots |

## 5. Data flow (one cycle)

1. Load `experiment.json`; verify spec lock (refuse if hashes differ).
2. Compute `cycle_id` = current UTC time floored to the 6h slot. Already logged → exit (idempotent).
3. Fetch tickers (bid/ask/24h volume) + closed hourly/daily candles for all 8 assets. Any
   stale/missing/inconsistent data → log `DATA_FAULT`, no trades this cycle, baselines still marked.
4. Build features → append to `snapshots.jsonl` (this is "what the system knew").
5. For every arm: replay hourly candles since the last cycle to trigger stops → update circuit
   breakers → mark to market → append to `equity.jsonl`.
6. Baselines compute target weights → (constrained arms) risk engine → paper exchange.
7. AI arm: build prompt (features + portfolio + theses + last cycle's risk feedback + rules) →
   LLM → validate → proposals.
8. Risk engine evaluates proposals: each APPROVED / CLIPPED / REJECTED with reason codes.
9. Fetch fresh quotes (post-LLM) → paper exchange fills approved orders with full cost model.
10. Update theses; append `decisions.jsonl`, `orders.jsonl`, `events.jsonl`; atomically save state.

## 6. Features given to the AI (per asset, all from closed candles + live ticker)

`price` (mid), `spread_bps`, `ret_1h`, `ret_6h`, `ret_24h`, `ret_7d`, `ret_30d`, `vol_7d_ann`,
`vol_30d_ann`, `volume_24h_usd`, `volume_ratio` (24h vs prior-7d daily average),
`dist_sma20`, `dist_sma50`, `drawdown_from_30d_high`, `rsi14_daily`, `corr_btc_30d`.
Indicators are one input among several, not the strategy.

## 7. AI trader contract

**Output (strict JSON, nothing else):**
```json
{
  "cycle_id": "2026-09-28T12:00Z",
  "outlook": {"BTC-USD": 1, "ETH-USD": 0, "SOL-USD": -1, "...all 8 assets required": 0},
  "decisions": [
    {"asset": "BTC-USD", "action": "ADD", "target_weight": 0.25, "stop_loss_pct": 12,
     "reasons": ["..."], "invalidation": ["..."], "confidence": 62}
  ],
  "thesis_updates": [
    {"asset": "BTC-USD", "status": "bullish", "summary": "...",
     "reasons": ["..."], "invalidation": ["..."]}
  ],
  "portfolio_note": "one or two sentences"
}
```
- `action` ∈ BUY, ADD, HOLD, REDUCE, SELL, EXIT, DO_NOTHING. SELL and EXIT both mean full close.
- `target_weight` = desired share of total equity after the trade (0-1). The system derives the
  order size; the AI never specifies dollars or quantities.
- BUY/ADD require non-empty `reasons` AND `invalidation` (no thesis, no trade) and an optional
  `stop_loss_pct` (default = max allowed; the AI may only tighten).
- `outlook` is required for all 8 assets every cycle, is evaluation-only, and is never read by
  the risk engine or exchange. `confidence` (0-100) is logged only.
- Malformed output: one retry with the same prompt; then the cycle is DO_NOTHING and logged
  `AI_FAULT`. Never "repaired" silently.

**Persistent thesis (`theses.json`, history in `theses_history.jsonl`):**
```json
{"BTC-USD": {"status": "bullish", "opened_cycle": "...", "updated_cycle": "...",
  "summary": "...", "reasons": ["..."], "invalidation": ["..."],
  "confidence_history": [{"cycle": "...", "confidence": 62}]}}
```
Each cycle the AI is shown its open theses and what changed since, and must reaffirm, revise
or close them.

## 8. Risk engine (deterministic; cannot be overridden by the AI)

| ID | Rule | Outcome |
|---|---|---|
| R1 | Max single-asset weight: 35% BTC/ETH, 15% alts | CLIP |
| R2 | Loss-at-stop ≤ 5% of equity per position (weight ≤ 5% ÷ stop%) | CLIP |
| R3 | Max gross exposure 80% (≥20% USD cash) | CLIP |
| R4 | Max non-BTC/ETH cluster 40% | CLIP |
| R5 | Every position carries a stop: default/max 18% below avg cost, AI may tighten to ≥5% | CLIP |
| R6 | Turnover ≤ 25% of equity per rolling 24h (stop/breaker exits exempt) | CLIP |
| R7 | Min order $50; one order per asset per cycle | REJECT |
| R8 | Order ≤ 0.5% of asset 24h USD volume; asset spread ≤ 50 bps | CLIP / REJECT |
| R9 | Daily equity drop ≥ 8% → no BUY/ADD for 24h | REJECT buys |
| R10 | Drawdown ≥ 25% from peak → liquidate everything, lockout 14 days, alert | FORCE EXIT |
| R11 | Stale (> 180s) or missing quote → asset skipped this cycle | REJECT |
| R12 | Invalid schema, missing thesis, action contradicts weight change | REJECT |
| R13 | Halted / locked out → only sells allowed | REJECT buys |
| R14 | No shorting, no leverage: sell ≤ held, buy ≤ cash (structural, in the exchange) | REJECT |

Interface: `evaluate(proposals, portfolio, snapshot, cfg, now) -> list[RiskDecision]`. Sells are
processed before buys; each rule sees the running hypothetical portfolio so approvals can't
jointly breach a cap.

## 9. Paper exchange model

- Market orders only. Buy fills at `ask × (1 + slippage)`, sell at `bid × (1 − slippage)`,
  from a quote fetched **after** the LLM returns (latency logged).
- Fee 40 bps of notional per side. Slippage = base (2 bps BTC/ETH, 6 bps alts) +
  `0.5 × (order_usd ÷ volume_24h_usd)` in fractional terms. Spread cost is the real observed spread.
- Costs are logged separately per fill (`fee_usd`, `spread_cost_usd`, `slippage_cost_usd`).
- No partial fills in v0.1: R8 caps every order at 0.5% of 24h volume, so they can't occur by
  construction. An order that would exceed available cash is scaled down, never overdrawn.
- **Stops:** each cycle replays the hourly closed candles since the previous cycle. If a candle
  opens at/below the stop → fill at that open (gap); if only the low touches it → fill at the
  stop; both minus slippage and fee. Uses only already-closed candles.
- Marking: mid price. Equity = cash + Σ quantity × mid.

## 10. Logging format (`state/`, all append-only JSONL unless noted)

| File | One row per | Key fields |
|---|---|---|
| `cycles.jsonl` | cycle | cycle_id, started_at, finished_at, status, ai_latency_s, tokens_in/out, model_id |
| `snapshots.jsonl` | cycle | cycle_id, per-asset mid/bid/ask/volume + features |
| `prompts.jsonl` | AI call | cycle_id, prompt_sha256, system_sha256, user_prompt, raw_response |
| `decisions.jsonl` | AI cycle | outlook, decisions, confidence, thesis_updates, risk_decisions (status, reason codes) |
| `orders.jsonl` | fill | arm, asset, side, qty, price, fee/spread/slippage costs, cause (ai/stop/breaker/baseline) |
| `equity.jsonl` | arm × cycle | equity, cash, exposure, drawdown, peak |
| `events.jsonl` | event | DATA_FAULT, AI_FAULT, STOP_TRIGGERED, BREAKER, LOCK_MISMATCH, DELISTED |
| `theses.json`, `arms/*.json`, `lock.json` | (overwritten atomically) | current state |

Reconstruction: `snapshots.jsonl` + `prompts.jsonl` show exactly what the AI knew and said at
any timestamp.

Future SQL index (v0.3, derived from the JSONL, never the source of truth):
```sql
CREATE TABLE cycle(cycle_id TEXT PRIMARY KEY, started_at TEXT, status TEXT, model_id TEXT);
CREATE TABLE snapshot(cycle_id TEXT, asset TEXT, mid REAL, bid REAL, ask REAL, vol24h_usd REAL,
                      features_json TEXT, PRIMARY KEY(cycle_id, asset));
CREATE TABLE decision(cycle_id TEXT, asset TEXT, action TEXT, target_weight REAL,
                      confidence REAL, outlook INTEGER, risk_status TEXT, risk_codes TEXT);
CREATE TABLE fill(fill_id INTEGER PRIMARY KEY, cycle_id TEXT, arm TEXT, asset TEXT, side TEXT,
                  qty REAL, price REAL, fee_usd REAL, spread_usd REAL, slip_usd REAL, cause TEXT);
CREATE TABLE equity(cycle_id TEXT, arm TEXT, equity REAL, cash REAL, drawdown REAL,
                    PRIMARY KEY(cycle_id, arm));
```

## 11. Biggest technical challenges

1. **Cloud cadence:** GitHub cron can delay or skip runs. Skips are logged, never backfilled; the
   1,240-completed-cycle rule handles coverage.
2. **State persistence:** runners are ephemeral. State lives on a separate `crypto-ai-data`
   branch so main's history stays clean and doesn't conflict with your own pushes.
3. **LLM reliability and cost:** malformed JSON, timeouts, drift. Bounded retries, hard
   fallback to DO_NOTHING, token usage logged so cost is measured, not guessed.
4. **Stop realism:** cycle-based decisions can't react intra-cycle; the candle replay is the fix.
5. **Sample size:** see PREREGISTRATION §6. The hardest problem is patience, not code.

## 12. Roadmap

| Version | Adds | Gate to proceed |
|---|---|---|
| **0.1** (this) | Everything above, price/volume only, forward-only | Tests pass; dry-run clean; owner confirms ⚑ items; lock |
| 0.2 | Random-policy Monte Carlo, weekly digest to Discord, cost-sensitivity reruns | First interim report needs it |
| 0.3 | SQL index + read-only dashboard | ≥3 months of data |
| 0.4 | News/sentiment modules, each pre-registered as its own IC test first | Sentiment must show IC alone (Rule 9) before the AI sees it |
| 0.5 | New info-menu arms (`ai_pvn` etc.), started via amendment | 0.4 gate |
| 1.0 | Decision point (12 or 24 months) | Verdict per PREREGISTRATION §7 |
