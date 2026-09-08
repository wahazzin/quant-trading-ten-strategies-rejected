# TRADING SYSTEM ROADMAP v2
*Revised July 2026, after Phase 2 completed and all price-only signals were rejected.*

This document exists so that if the project stalls, gets confusing, or is picked up
after a long break, the next person (or the next AI session) can read this one file
and know exactly where things stand and what happens next.

---

## 1. THE GOAL, STATED HONESTLY

Build a rules-based automated trading system with a **demonstrable, statistically
verified edge**, deployed only after it proves itself in paper trading.

Not: "make money fast." Not: "find the best stock."
The deliverable is **evidence**, and the system is what produces it.

**Current honest status: no validated edge has been found. Nothing is ready for
live capital.** That is a legitimate research outcome, not a failure.

---

## 2. WHERE WE ARE

| Phase | Status |
|---|---|
| 0 — Infrastructure (data, broker, storage) | ✅ Complete |
| 1 — Risk & execution core | ✅ Complete |
| 2 — Technical / price signals | ✅ Complete — **all rejected** |
| 3 — Event & sentiment layer | ✅ Complete — 8-K liquid-name effect now a live forward test (Phase 6b); news/shock sentiment rejected (Tests 13-14) |
| 3b — Regime-switching + RSI revisit | ✅ Complete — all rejected (Tests 16-19) |
| **Engine build** (dynamic universe, pluggable signals, capacity gate) | 🔄 **IN PROGRESS** — see ENGINE_ARCHITECTURE.md |
| 4 — Combination of proven signals | ⬜ Blocked — 0 validated signals; value_bm is unproven, forward-testing only |
| 5 — Monitoring & decay detection | ⬜ Not started |
| 6 — Paper trading (30–40 trading days) | 🔄 Three forward tests running: value portfolio, 8-K monitor, CAPM benchmarks |
| 7 — Go / no-go decision | ⬜ Not started |

---

## 3. HARD CONSTRAINTS WE HAVE *MEASURED* (not assumed)

These are the facts that shape every future decision. They were discovered
empirically during Phase 2 and must not be forgotten.

**C1 — Commission drag is the binding constraint.**
At ~50,000 SEK (~$4,700 USD) with $1.00/order minimum commission (IBKR, the
broker used for every test and the original Phase 6 forward test):
- Monthly rebalance, 38 positions → **4.04%/year** drag
- Quarterly rebalance, 20 positions → **1.49%/year** drag
Any strategy must clear this hurdle *before* it beats SPY.
**Implication: low turnover is not a preference, it is a requirement.**

**UPDATE (2026-08-02) — moved to Alpaca; the $1/order floor above no longer
applies going forward.** Migration reason: a months-long forward test cannot
depend on a Gateway process staying up on a local machine indefinitely; Alpaca's
REST API needs no persistent connection. Alpaca is commission-free for US
equities — the only costs are small regulatory pass-throughs (SEC fee, sells
only, ~$20.60 per $1,000,000 of principal; FINRA TAF, sells only, ~$0.000195/share;
a small CAT fee on both legs), each rounded up to the nearest cent, confirmed both
from published rates and the account's own fee history. **Measured round-trip
cost on a ~$450 position: approximately $0.04, or ~0.009%** — roughly 50x smaller
than the ~$2.00 (~0.44%) the IBKR-era model above assumed for the same position.
Full derivation and the account-scale implications of the migration (Alpaca's
paper account is ~$98,550 vs. the original ~$4,700 — the forward test's ~$3,500
target investment stayed at its original size, deployed as a small slice of a
much larger account, to preserve continuity with everything already measured
against the small-account numbers above) are in `RESEARCH_LOG.md`'s Phase 6
section.

**Flagged, not reopened: Test 9's rejection was partly cost-driven and may
deserve re-examination under this new cost basis.** Test 9 (SEC 8-K event
study) was rejected on economic grounds specifically: a real, statistically
significant ~0.41% edge (item 5.02, 10-day horizon) was smaller than the
~0.63% round-trip cost assumed at the time. Under Alpaca's real ~0.009% cost,
that same edge would clear costs by roughly 40x instead of failing by ~35%.
This is **not** an instruction to rerun Test 9 now — doing so with the specific
foreknowledge that costs just got much cheaper is exactly the kind of post-hoc
lever-pulling this project's discipline exists to prevent, and Test 9's data
is not holdout-clean for a fresh look (see Evidence-Boundary status,
`RESEARCH_LOG.md`). It is recorded here as a trigger for a future, properly
pre-registered retest — on genuinely new data, or through a formally reopened
and pre-committed evaluation — not something to act on today.

**C2 — More frequent trading makes everything worse.**
Higher frequency → more trades → more commission drag → worse net returns.
This is why minute-level or intraday data is *not* an upgrade for this account.
Swing timescale (days–weeks) is the only economically viable band at this size.

**C3 — Price-only signals in liquid US equities are exhausted.**
Eight independent tests, all negative. This is consistent with theory: these are
the most heavily mined datasets in finance. Do not re-test this class without a
genuinely new angle.

**C4 — Passing a practice window does not mean an effect is real.**
Low-volatility passed every robustness check (t=2.47 → 3.18 crisis-excluded,
stable across sub-periods, no sector concentration) and then produced **−5.60%
alpha** in the sealed holdout. Holdouts exist because of exactly this.

**C5 — Adding money does not fix a negative-alpha strategy.**
Low-vol returned 6.81% *gross with zero costs* while SPY returned 16.77% over the
same window. Scaling a losing strategy scales the losses.

---

## 4. PHASE 3 — EVENT & SENTIMENT LAYER (current)

**Thesis:** price data is structurally blind to discrete corporate events — fraud,
executive departures, accounting restatements, bankruptcy. That information is
real, it moves stocks, and it is not in the price until after it appears.

**Where the edge could plausibly live** (not in speed — HFT owns that):
- **As a risk filter**, not an alpha source: avoid *holding* through a widening
  negative event. Bad news drifts over weeks.
- **In under-covered small caps**, where information diffuses slower. This is the
  original edge thesis of the whole project.

### Step 3.1 — 8-K item-code event study *(running)*
Structured SEC filing codes only. **No NLP.** Zero contamination risk.
- Data: SEC EDGAR, free, timestamped, 20+ years
- Key codes: 1.03 (bankruptcy), 2.02 (results), 4.02 (financials unreliable —
  accounting fraud), 5.02 (officer departure), 8.01 (other material)
- Measure: abnormal returns (stock minus SPY) at 1/5/10/20 days after filing
- Split by dollar-volume half to test the under-covered hypothesis
- **Evidence boundary: 2019+ only. Pre-2019 is now the sealed holdout for this
  hypothesis class.**

**Gate to proceed:** abnormal returns with |t| > 2 on at least one item code,
economically meaningful after costs.

### Step 3.2 — FinBERT on filing text *(only if 3.1 passes)*
Runs locally, free, unlimited. Trained on financial text, **not** on knowing
outcomes — so unlike a general LLM, it can be backtested honestly.
Purpose: separate "CFO retired after 20 years" from "CFO resigned amid probe."

### Step 3.3 — Combination *(only if 3.2 adds standalone value)*

**If Step 3.1 fails:** Phase 3 closes. Move to Section 8 (Decision Point).

---

## 5. PHASE 4–7 (unlocked only by a Phase 3 pass)

**Phase 4 — Combination.** Only signals with independently proven positive
expectancy get combined, and the *combination* is re-validated. A combo can be
worse than its parts.

**Phase 5 — Monitoring & decay detection.** Now high priority because of C4.
Continuously re-test live strategies; automatically flag and halt any strategy
whose rolling performance degrades below a pre-set threshold. Strategies die —
the system must notice before the account does.

**Phase 6 — Paper trading.** Minimum 30 trading days, maximum 40. Weekly reports:
total trades, win rate, avg win vs avg loss (expectancy), current drawdown %,
performance vs SPY. Non-negotiable per Rule 1.

**Phase 7 — Go / no-go.** Must beat SPY after **all** costs including commissions.
If it doesn't, it doesn't go live. Per Rule 5.

---

## 6. INFRASTRUCTURE BACKLOG
*Build when needed, not before. Premature infrastructure is procrastination.*

| Item | Build before | Status |
|---|---|---|
| Weekly report generator (Rule 2) | Phase 6 | Not built |
| Telegram alerts (entry/stop/target + circuit breaker fires) | Phase 6 | Not built |
| Decay monitoring | Phase 7 | Not built |
| Account snapshot logging (currently silently unused) | Phase 6 | **Bug — table exists, never written to** |

---

## 7. STANDING RULES OF PROCEDURE

1. **Pre-register before looking.** Declare the specification in writing before
   running it. No post-hoc parameter selection.
2. **One shot per holdout.** If unsure whether to spend one, don't.
3. **Log every rejection.** See `RESEARCH_LOG.md`. Never re-test something already
   rejected without genuinely new evidence or a new data source.
4. **Every cost model includes the $1/order commission floor.** A backtest without
   it is not measuring reality at this account size.
5. **Test each component alone before combining** (Rule 9).
6. **A negative result is a result.** Log it and move on.
7. **Rule 13 applies to us, not just to influencers.** If our own result looks
   too good, that is when to look hardest.

---

## 8. DECISION POINT (if Phase 3 fails)

If the event layer produces nothing, the honest options are:

**(a) Stop and index.** Nine rigorous tests, all negative, is strong evidence that
no retail-accessible edge exists in the data available. SPY returned 16.77%/yr in
the holdout window with Sharpe 1.00. Buying the index is the evidence-based move.

**(b) Repoint the machinery.** The testing pipeline — pre-registration, monkey
tests, evidence boundaries, sealed holdouts, cost modeling — is the durable asset
here and works on any hypothesis, not just trading. It is reusable indefinitely.

**(c) Come back with better data.** The binding limit has consistently been data
access, not method. If circumstances change (income, institutional access,
university enrollment), the questions can be re-asked with survivorship-bias-free
data.

None of these is a failure. Option (a) in particular is what the evidence
currently supports.

---

## 9. EXPLICITLY REJECTED — DO NOT REVISIT

**Tested and rejected** (details in `RESEARCH_LOG.md` — 19 total):
SMA crossover · RSI mean-reversion · 20-day z-score reversal (both formulations,
both universe sizes) · short-term reversal (revealed as market beta) ·
momentum 12-1 (alpha t=0.13) · low volatility (holdout alpha −5.60%) ·
fundamental value/quality (underpowered, t=1.52) · Fugazzi & Swedish CAPM
retests (positive but t<2) · pooled news-sentiment IC (clean null) ·
conditional shock-day sentiment (failed independence/liquidity checks) ·
volatility targeting (holdout Sharpe ratio 0.97, alpha t collapsed to 0.51) ·
regime classifier (+2.3pp vs +5pp bar) · RSI-conditioned-on-regime, 3x ·
RSI portfolio simulation (only 22% of signals tradeable, alpha −3.14%).
**Live/open, not rejected:** 8-K liquid-name event effect (Phase 6b forward
test, pre-registered, awaiting 100 declustered events) and value/B-M
(Phase 6 forward test, unproven — underpowered in backtest, not disproven).

**Rejected without testing, with reasons:**
- *Intraday / minute-level trading* — violates C1 and C2; competes with HFT;
  requires paid data we don't have
- *"Liquidity sweep" / ICT-style setups* — no published evidence base
- *Breakout & trend-continuation systems* — same hypothesis class as the SMA and
  momentum tests already rejected
- *Long-term reversal factor* — cannot be honestly tested on survivorship-biased
  free data (missing losers are exactly the ones that went to zero)
- *Levering the low-vol factor* — margin interest (~5–6%) exceeds the levered
  edge; amplifies drawdowns
- *Self-replicating / "survival mode" AI trading agents* — copying code does not
  create capital; the real-money version of this experiment lost 31–45% for two of
  six models tested
- *Pooled/free API-key repos* — ToS violations or stolen credentials

---

## 10. EXTERNAL TOOLS — STATUS AND TRIGGERS

Tools evaluated or adopted outside this repo's own code, what each is actually
for, and the specific condition that would make picking it back up worthwhile.
None of these are a substitute for a proven edge — most of them are plumbing
that only matters once one exists.

**ml4t** (cloned at `C:\Users\uwuzp\ml4t-study`) — a machine-learning-for-trading
curriculum/codebase. Its Information Coefficient methodology is already in use
throughout this project (`ic_analysis.py` and everything built on the same
per-stock-IC / cross-sectional-t-test pattern). Remaining chapters are largely
**moot**: much of the rest of the curriculum builds toward NLP/sentiment
feature engineering, and Tests 13-14 already tested that hypothesis class to
death (clean null on pooled IC, and the one significant conditional effect
failed independence and liquidity robustness checks). Revisit only if a
non-sentiment ML technique from later chapters (e.g. a specific factor-model
or portfolio-construction method) becomes relevant to a new, still-untested
hypothesis.

**NautilusTrader** and **QuantConnect Lean** — production-grade execution
engines. Both are **not currently relevant**: they solve the problem of
running a proven strategy reliably at scale, and this project doesn't have a
proven strategy to run. **Trigger: a proven edge (holdout-surviving, not just
practice-window-passing) PLUS an actual decision to go live with real,
non-paper capital.** Until both conditions hold, adopting either would be
infrastructure for a strategy that doesn't exist yet — the exact "premature
infrastructure is procrastination" trap Section 6 already warns against.

**TradingAgents** — a bull/bear multi-agent debate framework for
sentiment-driven decisions. **Moot.** Its entire premise is that structured
debate over sentiment signals adds value; Tests 13-14 tested sentiment
directly (FinBERT, pooled IC and a conditional shock-day design) and found
nothing that survives scrutiny. Layering a debate structure on top of a
signal that isn't there doesn't create one.

**Vibe-Trading** — an independent cross-check / second-opinion tool. Not
needed today because there is no candidate signal to check. **Trigger: if a
future test ever produces a genuine holdout-surviving result**, this is the
right tool to run as an independent adversarial check before treating that
result as real — the same spirit as Test 14's five-check robustness gauntlet,
from a second angle.

**alphalens-reloaded** — **rejected**, not merely deferred. It forces a
pandas downgrade, which conflicts directly with the hard `pandas 3.0.3`
requirement stated in every script's environment notes across this entire
project. Not worth the fragmentation of running two incompatible pandas
versions for one library's convenience functions, all of which (IC analysis,
quantile spreads, turnover) are already hand-built and working in this
codebase.

## 11. FLAGGED, NOT ACTIVE — CANDIDATE FUTURE HYPOTHESES

Ideas raised but deliberately not started. Listed so they aren't re-raised
as if new, and aren't forgotten either.

**Social media sentiment (StockTwits, not Twitter/Facebook).** Retail
sentiment can lead news-derived sentiment (Tests 13-14 tested news only,
both clean/failed). Twitter/X API is now paywalled for backtesting;
Facebook has no usable public API; building an unauthorized scraper is off
the table (ToS). StockTwits has a real, legal, stock-specific API and is
the standard academic proxy for retail sentiment. **Trigger to start:**
none yet — not scheduled, just the honest next candidate if sentiment is
revisited. Same rigor as Tests 13-14 required (declustering, SPY-drift
control) would still apply, plus a new confound to control for: pump-and-
dump/bot coordination can fake predictive-looking sentiment.

**Options-derived positioning (implied vol skew, put/call skew) — verdict:
MAYBE.** Confirmed after reading the full source (Aug 2026): construction is
`(OTM put IV - OTM call IV) / ATM IV` on delta-matched strikes, classified
into a 4-quadrant grid (price trend x skew) -- price and positioning
agreeing tells you nothing, price and positioning DISAGREEING (falling
price + upside options bid, or rising price + downside options bid) is the
only case worth testing. Worth pursuing because: (1) genuinely new data
source, none of the 19 rejected tests used options data; (2) real academic
grounding (informed options positioning preceding price moves is a
published finding -- verify citations independently, not confirmed here);
(3) the source material's own documented pitfalls (index put/call ratio is
unusable due to tail-strike windowing; skew must be raw vol points across
sectors, normalized within a sector; one bad options quote can invert an
entire name's reading; earnings-week skew is event premium, not
sentiment) are useful engineering warnings regardless of the source.
NOT confirmed: any actual backtested edge -- the source provides a
heuristic reading framework (4 quadrants + qualitative verdict), not a
measured expectancy, and its own performance claims are unverifiable
marketing for a paid subscription, not evidence. Do not adopt the
quadrant->action mapping as-is; it has to earn the same t-stat/monkey-
test/holdout bar as every other candidate.

Raised via a third-party "skew map" article/course with unverifiable
performance claims ($144k, then separately $65K->$300K+ / $129K / "9x
S&P" for a paid $69/mo community) — the marketing claims are not evidence
and are not being relied on. **What would actually need doing, not
started:** source real options data (not free like yfinance — CBOE, ORATS,
or similar), define a specific pre-registered signal (e.g. skew vs.
realized-vol divergence), and run it through the same pipeline (monkey
test, cost model, holdout) as every other candidate. **Trigger to start:**
a deliberate decision to spend on options data, not just interest in the
idea.

**Politician / corporate insider transaction disclosures (e.g. Quiver
Quant) — verdict: WEAK MAYBE.** Raised via a generic Claude-prompting
workflow guide, not a strategy source. Genuinely a data source none of
the 19 tests used, but weaker candidate than options skew: academic
evidence on congressional-trading predictive power is mixed, and the
signal is now heavily crowded (multiple ETFs already mirror disclosed
Congress trades), which tends to arbitrage away edge that existed.
**Trigger to start:** none — noted so it isn't re-raised as new, not
scheduled ahead of the options-skew candidate.

## 12. IDEAS WORTH STEALING (not hypotheses, not tests -- design/engineering
patterns noticed while reviewing external material, logged here so they
aren't lost or re-suggested as new)

**Notifications should default to silence, not report every cycle.**
Source: a third-party "Grok Bot Desk" guide (Aug 2026) -- unrelated to any
trading strategy, but the design principle is sound: most alert systems
train you to ignore them by messaging constantly. Its "head of desk" role
only sends a message when something clears a genuinely high bar, and
outputs nothing (not even a log spam line) otherwise. Applies directly to
our planned Discord webhook notifications: fills, stop-loss hits, and
circuit-breaker trips should alert; routine dry-run cycles and "nothing
happened" should not.

---

*Last updated: 2026-08-14, after Tests 16-19 (regime-switching + RSI) were
rejected and the engine build (bot/signals, bot/monitor/attribution,
bot/screener/universe) began. See ENGINE_ARCHITECTURE.md for the current
build. Combiner (bot/combiner/combiner.py) written next.*
