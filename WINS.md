# What This Research Program Actually Achieved

This document is for someone who has never seen this repository. It does not
contain a profitable trading strategy — fifteen hypothesis classes were
tested, pre-registered, and rejected. That is not a failure of the project;
it is the project. What follows is what the discipline behind that record
actually caught, in concrete terms, with the numbers.

For the full statistical detail behind every claim here, see
[`RESEARCH_LOG.md`](RESEARCH_LOG.md).

---

## 1. What the methodology caught

Two strategies passed practice-window testing — including every robustness
check specifically designed to break them — and were then killed by a sealed
holdout they had never seen. Both would have been deployed on real capital
without the holdout discipline. This section states that plainly, with the
before/after numbers.

### Low-volatility long-only (Test 8)

A textbook academic anomaly: long the least-volatile stocks, equal-weighted.

| | Practice window (2008–2018) | Holdout (2019–2026) |
|---|---|---|
| Annualized alpha | **+4.23%** | **-5.60%** |
| t-stat on alpha | **2.47** | -1.62 |
| Beta | 0.662 | 0.773 |
| Net Sharpe | 0.88 | 0.46 |
| SPY Sharpe (same window) | 0.64 | 1.00 |

Before the holdout, this strategy survived a full adversarial gauntlet:
sub-period stability (positive alpha in both halves of the practice window),
crisis exclusion (alpha got *stronger* without the 2008–09 window, t=3.18 —
clearing the strict Harvey-Liu-Zhu bar of 3.0), sector-concentration checks
(no single sector over 24% of the book), and position-count sensitivity
(stable from 10 to 38 holdings). Every check a diligent researcher would run
on a promising backtest, it passed.

The sealed 2019–2026 holdout didn't just weaken the effect — alpha **flipped
sign**. Even gross of every cost, the strategy returned 6.81%/yr while SPY
returned 16.77%/yr over the identical window.

### Volatility targeting (Test 15)

A different kind of strategy entirely — this one doesn't pick stocks, it
times exposure to the market (Moreira & Muir, *Journal of Finance*, 2017):
hold less when realized volatility is high, more when it's low.

| | Practice window (1993–2009) | Holdout (2010–2026) |
|---|---|---|
| Sharpe | 0.66 | 0.95 |
| Sharpe ratio vs. SPY | **1.217** | **0.970** |
| Annualized alpha | +2.26% | +0.76% |
| Alpha t-stat | 1.59 | 0.51 |
| Max drawdown | -39.03% | -19.07% |

In the practice window (1993–2009, 200 monthly observations — the largest
sample used anywhere in this project), the strategy beat SPY on Sharpe ratio
*and* beat a constant-leverage benchmark carrying the same average exposure
(1.244× ratio) — meaning the improvement was attributable to the
time-varying exposure itself, not just to carrying leverage. That is exactly
the claim the underlying paper makes.

In the sealed 2010–2026 holdout, the Sharpe ratio versus SPY fell to 0.970
— **the strategy no longer even matched SPY's risk-adjusted return**, and
alpha collapsed to statistical noise (t=0.51). See §2 for exactly why.

### The point

Both of these were genuine, well-powered, multi-check-surviving results in
their practice windows. Both died on data the strategy had never seen.
**Without the holdout discipline, both would have gone to paper trading and
then live capital.** Two out of fifteen tests following the identical
arc — real in practice, dead in holdout — is not a coincidence to explain
away. It is the single strongest argument this project makes for why a
sealed, one-shot holdout is not optional.

---

## 2. Mechanistic findings

Three results where the interesting part isn't just "it didn't work" but
*why*, in enough detail to be useful to someone building a different
strategy entirely.

### Why volatility targeting fails a V-shaped crash

Test 15's strategy rebalances monthly, sizing next month's exposure from the
*trailing 21-day realized volatility as of last month's close*. That one-month
lag is fine in a slow-building volatility regime. It is close to the worst
possible design for a crash that both falls and recovers inside a single
month — exactly what COVID-19 did:

| Month | Exposure | Vol-targeted return | SPY return | Difference |
|---|---|---|---|---|
| 2020-02 | **123.3%** | -9.69% | -7.77% | -1.92pp |
| 2020-03 | 61.7% | -10.12% | -16.35% | +6.24pp |
| 2020-04 | **16.6%** | +2.48% | +15.05% | **-12.57pp** |
| 2020-05 | 37.1% | +2.37% | +6.42% | -4.05pp |
| 2020-06 | 67.5% | +1.61% | +2.41% | -0.80pp |

Cumulative Feb–Jun 2020: the strategy returned **-13.47%** against SPY's
**-3.27%** — a 10-point loss during the exact five months it existed to
protect against.

Read the exposure column: February's 123% exposure (near the strategy's
1.5× cap) was set from January 2020's realized volatility — a calm,
pre-crash reading. The strategy walked into the fastest crash on record
*maximally levered*, because its only information about risk was already a
month stale. Then it overcorrected: March's extreme realized volatility
pushed April's exposure down to 17%, just in time to miss most of SPY's
sharp rebound (+15.05% in April; the strategy captured +2.48%).

**The lesson generalizes beyond this one strategy:** any exposure-sizing
rule driven by a trailing, backward-looking volatility estimate will be
late in both directions during a fast, V-shaped shock. It will be
over-exposed walking into the drop (because the drop itself hasn't shown up
in the trailing window yet) and under-exposed walking into the recovery
(because the drop *has* shown up, and the estimate hasn't caught up to
the fact that it's over). A faster lookback trades this for whipsaw
sensitivity in normal markets — there is no free parameter that fixes it.

### Why the benchmark you pick can fabricate an effect (Test 9)

The first version of the SEC 8-K event study (`event_study.py`, v1) measured
each stock's abnormal return after a filing as *(stock return − SPY
return)*. It found a broad, statistically strong negative drift after 8-K
filings and nearly treated that as a real information-processing effect.

The catch: the test universe underperformed SPY by roughly 5%/year as a
plain baseline, unrelated to any filing. Subtracting SPY's return from every
stock's return doesn't isolate the event's effect — it also bakes in
whichever direction the universe happens to be drifting relative to the
benchmark, for the whole sample period, every single time. A universe that
happens to underperform its benchmark will show a "negative event effect"
on *any* event you study in it, filing-related or not.

The fix (`event_study_v2.py`) replaced the benchmark with a matched
non-event control: each stock's own mean forward return on days *not*
within 20 trading days of any of its own 8-K filings, at the same horizon.
That differences out each stock's idiosyncratic drift without needing a
market benchmark at all. Most of the "effect" v1 found vanished once this
was in place — what remained was much smaller, and is discussed in §3.

**The lesson:** benchmark choice is not a neutral reporting decision. A
benchmark-relative measurement is only as clean as the assumption that your
test universe and your benchmark would have moved identically absent the
event — and that assumption is rarely true for a hand-selected universe.

### Pseudo-replication can manufacture significance out of clustering (Test 14)

Test 14 found what looked like a real one-day effect: stocks with an
unusual burst of negative-sentiment news headlines underperformed their own
baseline the next day by -1.69%, t=-3.94 — a very strong result on paper.

The problem: "shock days" cluster. A company doesn't get one isolated
negative headline; it gets a burst of five, ten, sometimes dozens of
articles about the same underlying event over a few days. The original test
counted each of those clustered days as an independent observation.

Declustering — keeping only the *first* shock day in any 10-trading-day
window per stock, discarding the rest as the same event re-counted — cut the
sample from 589 to 335 events and the effect from **t=-4.02 to t=-1.60**:
no longer significant. Most of the original significance was the same
handful of real events being counted five or six times each, not five or
six independent pieces of evidence.

**The lesson:** any test built on event-days needs an explicit independence
check before its t-statistics can be trusted. A large n built from clustered,
correlated observations produces exactly the over-confident significance
this project's whole methodology exists to catch — the same failure mode
this project's own low-volatility and volatility-targeting results warn
about, just showing up inside a single test instead of across a holdout
boundary.

---

## 3. Quantified constraints

### Commission drag is the binding constraint at this account size

At the ~$4,700 account size used throughout this project, with a flat
$1.00/order minimum:

| Rebalance frequency | Positions | Annual commission drag |
|---|---|---|
| Monthly | 38 | **4.04%/yr** |
| Monthly | 20 | 2.50%/yr |
| Quarterly | 20 | 1.49%/yr |

A real 4%/yr alpha — roughly what Test 8's low-volatility strategy showed in
its practice window — does not survive the monthly-38-position drag intact.
Cutting to a 20-stock quarterly rebalance nearly triples the net edge kept
(1.49%/yr drag instead of 4.04%), but the tradeability test found the signal
itself decays over a 3-month hold fast enough that net alpha collapsed
anyway (t=0.94) — cost reduction and signal decay pull in opposite
directions, and there is no rebalance frequency in between that avoids both.

The break-even account size for the strategies tested in this project — the
size at which commission drag stops being the dominant constraint — is
roughly **4× the capital actually available**.

### Statistical significance is not economic significance

The SEC 8-K event study produced t-statistics as extreme as **-12** on
measured effects of roughly **0.4%** — a result that would read as
overwhelming evidence to anyone looking only at the p-value. Against this
project's own account-size economics, a realistic position (~$470, per a
10-way split of a 50,000 SEK account) costs about **0.63%** round-trip
(commission plus spread) to trade. A 0.4% effect, however statistically
airtight, is a losing trade before the signal even has to be real.

This is not a subtle point, but it is the one most backtests skip: with
enough observations (Test 9 had 124,419 filing events), almost any nonzero
effect becomes "significant." Significance is a statement about whether an
effect is distinguishable from zero. Tradeability is a statement about
whether it is bigger than what it costs to capture — and at n>100,000, those
two questions have almost nothing to do with each other.

---

## 4. Data-quality bugs caught

None of these were found by looking for bugs. All four were found because a
downstream number looked implausible and got traced back.

**`CommonStockSharesOutstanding` corruption (Test 10 / Phase 6).** Building
the live forward test's stock-ranking script, several top-ranked "value"
picks showed Book-to-Market ratios in the hundreds of thousands — an
arithmetic impossibility for any real company (it implies a market cap of a
few dollars). The cause: SEC XBRL's `us-gaap:CommonStockSharesOutstanding`
tag is unreliable for a meaningful slice of filers — Up-C/holding-company
registrants, foreign private issuers on 20-F/6-K, and redomiciled entities
sometimes report a technical share class instead of total float (one
example found live: SPG showing "8,000 shares" against a real float near
325 million). Fixed with a data-integrity floor — reject any observation
implying more than 100%/day share turnover, which is not economically
plausible for a real company. Re-running the full historical backtest with
the fix barely moved the headline numbers (annualized alpha +5.28% → +5.55%,
t=1.46 → 1.52) — the bug hadn't materially contaminated the *historical*
top-decile picks, but it would have actively corrupted the *live* portfolio
the day it launched had it not been caught first.

**Stale, unverified CSVs sitting alongside verified data.** The project's
original 13-stock universe was built by `verify_universe.py`, which
mechanically checks each candidate against live broker data (price band,
average daily volume, minimum history) before it's allowed into the tested
universe — "no opinions, the broker's own data decides." An audit of the
`data/` directory while preparing this document found 46 per-ticker CSV
files on disk, but only 13 tickers ever passed verification and were used in
any test. The other 33 — including 24 tickers that don't even appear in
`verify_universe.py`'s candidate list at all — are orphaned artifacts from
superseded fetch attempts, never cleaned up. No test was contaminated by
this (every price-based test in this project uses an explicit, hardcoded
ticker list, never a directory glob), but it is exactly the kind of latent
risk that *would* silently corrupt a result the day someone writes
`glob("data/*.csv")` instead of checking the verified list — worth
documenting so it doesn't become that.

**A ~30-percentage-point hindsight bias in a prior project's stock picks
(Test 11).** Retesting an earlier CAPM stock-picking exercise, the original
six-stock portfolio (NVDA, AVGO, LLY, XOM, WMT, GOOGL) had been selected
*after* the tester already knew how all six performed through 2025 — not
just through the training window the original scripts claimed to use (which
itself had a separate bug: three of the four original test scripts
accidentally evaluated performance on the 2021–2023 *training* window
instead of the intended 2024–2025 walk-forward window). Rebuilding the
selection honestly — ranking all 30 original candidates by Jensen's alpha
using *only* data through end-2023, with no knowledge of what came after —
and running that honest six-stock portfolio through the same 2024–2025
window produced +116.23% total return against the original hindsight-picked
portfolio's +146.09%. The gap, **+29.86 percentage points**, is the
quantified size of the hindsight advantage baked into the original
selection — roughly a fifth of the portfolio's apparent outperformance had
nothing to do with skill.

---

## 5. Infrastructure verified

`ops/verify_system.py` runs five mechanical checks against the live trading
system — the broker connection, the trade journal, and the risk manager —
each answering a specific question a real account depends on getting right.

| # | Check | What it proves | Last verified result |
|---|---|---|---|
| 1 | Trade journal price integrity | No trade is logged with entry price == exit price (the original fake-fill bug this journal design was built to prevent) | **PASS** |
| 2 | Account snapshot logging | Equity snapshots exist and their timestamps advance monotonically | **PASS** |
| 3 | Risk-state persistence | Peak equity and start-of-day equity survive a process restart, read fresh from disk, not from memory | **PASS** |
| 4 | Circuit-breaker trigger logic | A fabricated equity value 25% below the real peak correctly halts trading, *and* the real stored state is left completely untouched by the test (verified by re-reading it after) | **PASS** |
| 5 | Broker/local position reconciliation | Every position and open order at the broker matches what the local system expects | **FAIL** |

Check 5's failure was real, not a test bug: it correctly caught two stale
`SELL 56.0 F` bracket-order legs resting at IBKR — leftovers from earlier,
unrelated testing on a different symbol — that had no corresponding record
in the local system. That is exactly the class of mismatch this check
exists to catch, and it caught it. **4 of 5 checks passing, with the 5th
correctly flagging a real state mismatch rather than false-passing, is a
better demonstration that the verification works than a clean 5-for-5 would
have been.** This document reports the check's last-verified result rather
than assuming it now reads differently; whether those orders have since been
cleared is worth re-checking before treating Check 5 as resolved.

---

## 6. What was NOT achieved

Stated as plainly as everything above: **no strategy tested in this project
beat SPY out of sample, across fifteen separate hypothesis classes, spanning
technical signals, cross-sectional factors, an SEC filing event study,
fundamental value and quality, two independent CAPM stock-picking retests,
two news-sentiment tests, and volatility targeting.**

Two of those fifteen (low-volatility, volatility targeting) looked like
genuine discoveries right up until the moment they were tested on data they
had never seen. Both times, the discipline of holding out that data and
spending it exactly once caught what every practice-window check had missed.

This repository does not contain a trading edge. It contains the record of
looking for one honestly, and the infrastructure that would be needed to
trade one safely if it were ever found. As of this document, the only
uncontaminated evidence stream remaining is the live forward test of the
Value (Book-to-Market) factor described in `RESEARCH_LOG.md`'s Phase 6
section — the one candidate from fifteen tests that was too underpowered to
reject *or* confirm in backtest, now accumulating real, un-mined monthly
observations one at a time.
