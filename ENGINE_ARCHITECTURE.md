# TRADING ENGINE — ARCHITECTURE

Built after 19 rejected hypothesis tests. The point of this design is that
**the machine is separate from the signal.** Nineteen signals failed; the
engine that screens, ranks, sizes, executes, and monitors is reusable
regardless of which signal eventually earns its place inside it.

---

## Data flow

```
   UNIVERSE SCREENER          (dynamic — not a fixed ticker list)
            │
            ▼
   SIGNAL LAYER               (pluggable; each signal scores every stock)
            │
            ▼
   COMBINER                   (weights signals into one score; ML slot)
            │
            ▼
   SENTIMENT VETO             (logged only, does not act — forward test)
            │
            ▼
   RISK MANAGER               (position sizing, stops, circuit breaker)
            │
            ▼
   CAPACITY GATE              (max N positions — Test 19's lesson)
            │
            ▼
   EXECUTION                  (Alpaca, via reconcile.py safety layer)
            │
            ▼
   JOURNAL + MONITOR          (every decision logged; monthly vs SPY)
```

---

## Design rules, each traceable to a finding

**1. Dynamic universe, never a fixed ticker list.**
The prior CAPM bot could only trade 6 hardcoded symbols, so it could never
be better than the initial guess. The screener rebuilds eligibility every
rebalance from structural filters (price, liquidity, history) — never from
past returns, which would bake hindsight into selection.

**2. Signals are plugins with a common interface.**
Each signal implements `score(universe_data) -> Series`. Adding or removing
a signal must not require touching the engine. This is what makes the
machine outlive any individual hypothesis.

**3. Capacity is simulated BEFORE any result is believed.**
Test 19: RSI showed monthly t=3.50 at trade level, then **negative alpha**
once limited to 20 concurrent positions — only 22% of signals were even
takeable. A per-trade average is not a portfolio return. The capacity gate
is therefore part of the engine, not an afterthought in analysis.

**4. Sentiment is logged, not acted on.**
Tests 13/14 found no tradeable signal (IC≈0; shock-day effect died to
declustering and vanished in liquid names). It runs as a recorded veto so
a forward record accumulates without contaminating live decisions.

**5. The combiner is an explicit, swappable slot.**
Starts as a transparent weighted average. Upgrades to a trained model only
when there are ≥2 signals with independently demonstrated edge. ML combines
signals; it does not manufacture them.

**6. Retraining uses champion/challenger, never blind adaptation.**
Rule 10: "A loss doesn't automatically teach the model anything." A
retrained challenger must beat the live champion on held-out data before
promotion. Every promotion decision is logged with the evidence that
justified it.

**7. Every decision is journaled — including the ones not taken.**
Rejected signals, capacity-blocked entries, and vetoed trades are all
recorded. Without that, you cannot later ask "would the veto have helped?"

**8. Benchmarked against SPY, always.**
Rule 5. Absolute profit is meaningless — the prior CAPM accounts made
+9.32% and +6.06% while SPY made +10.80%.

---

## Status

| Component | State |
|---|---|
| Universe screener | to build |
| Signal interface + registry | to build |
| Signal: value (book-to-market) | exists in research, to port |
| Combiner (transparent) | to build |
| Sentiment veto (logged) | to build |
| Risk manager | **exists** (`bot/risk/`) |
| Capacity gate | to build |
| Execution + reconciliation | **exists** (`bot/broker/`) |
| Journal | **exists** (`bot/journal/`) |
| Monitor vs SPY | partial (`ops/value_report.py`) |
| Retraining harness | to build (after ≥2 proven signals) |

**Honest status of the signal inside it: unproven.** The engine is being
built so that when a signal does clear the bar, the machine is already
running and tested. Swapping a signal in must be a config change, not a
rebuild.
