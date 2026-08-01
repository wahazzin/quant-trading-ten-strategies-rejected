# Research Log

Permanent record of every signal/factor test run in this project, in order,
with the key statistics and final verdict for each. **Read this before
proposing any new backtest.** If a configuration below is already tested
and rejected, do not re-test it without a genuinely new hypothesis, a
different universe/asset class, or materially different data — and never
treat the sealed 2019+ holdout on `data/yf_universe.parquet` as fresh data
for a new test on that same universe; it has been spent (see Test 8).

Project layout after reorganization:
- `research/data_fetch/` — data acquisition scripts (IBKR + yfinance)
- `research/signal_tests/` — short-horizon signal tests, 13/36-stock universes
- `research/factor_tests/` — long-horizon factor tests, 2,500-ticker universe
- `ops/` — live trading / account operations scripts
- `bot/` — the trading package (broker client, journal, risk manager)

## Data infrastructure notes

- **IBKR (paper account)**: source for the original 13-stock and 36-stock
  universes (`verify_universe.py`, `expand_universe.py`, `fetch_history.py`,
  `fetch_spy.py`).
- **Massive Market Data / Polygon API**: never obtained. No key was found in
  `.env`, environment variables, or any config file; the task was stopped
  and handed back rather than guessed at. Not pursued further after the
  project moved to free bulk data instead.
- **Stooq bulk downloads**: no longer freely accessible. `stooq.com/db/h/`
  (and even the per-ticker `/q/d/l/` CSV endpoint) now gate every request
  behind a client-side proof-of-work bot challenge, and the historical
  direct file URL (`static.stooq.com/db/h/d_us_txt.zip`) returns
  `401 Unauthorized` requiring credentials we don't have. Do not retry
  without new Stooq credentials — solving the challenge programmatically
  was deliberately not attempted (it exists specifically to block this).
- **yfinance**: adopted as the free, no-API-key data source for the
  large-universe factor tests. Rate limiting is cumulative over a session
  and gets progressively worse past roughly ~2,500 tickers in one run;
  a full 5,494-ticker attempt lost 837 tickers to `YFRateLimitError` in
  later chunks. Kept subsequent runs to ≤2,500 tickers with batching
  (50-500/batch), inter-batch sleeps, and one retry pass — this fetched
  cleanly with ~94% success and no throttling.

## Universe evolution

13 stocks (IBKR-verified, `verify_universe.py`) → 36 stocks (`expand_universe.py`,
2.8x expansion via mechanical filters, not the hoped-for 8x due to filter
attrition) → 600-ticker yfinance pilot (`fetch_yf_universe.py`) → **2,500-ticker
yfinance sample** (final universe used for all factor tests, `data/yf_universe.parquet`).

---

## Test 1 — SMA 20/50 crossover — **REJECTED**
- Script: `backtest_sma.py` (single stock F), `backtest_universe.py` (13 stocks)
- Monkey test (500-2000 simulated random traders, matched trade count/holding length): **44.2% average monkey-beat** across the 13-stock universe — worse than a coin flip.
- Verdict: no edge, indistinguishable from noise. Rejected.

## Test 2 — RSI(14) mean-reversion (oversold 30 / exit 55 / max hold 20d) — **REJECTED**
- Scripts: `backtest_rsi.py`, `validate_rsi.py`
- Universe: 13 stocks (IBKR)
- Initial monkey test looked promising: 63.4% avg monkey-beat, 7/13 stocks beat >60% of monkeys.
- Validation exposed it as noise: pooled t-stat on 71 trades = **0.344** (95% bootstrap CI
  **[-1.93%, +2.80%]**, includes zero); parameter-stability grid inconclusive (17/27 combos >55% but no clear signal); portfolio Sharpe **0.134**, max drawdown -7.95%.
- Verdict: statistically indistinguishable from zero once tested honestly. Rejected.

## Test 3 — IC diagnostic scan (8 signals × 4 horizons) — **DIAGNOSTIC** (led to Test 4)
- Script: `ic_analysis.py`
- Universe: 13 stocks
- Found a strong, consistent short-term reversal signal: RSI14, z-scores (MA20/MA50), momentum, and 5-day return all negative and significant (e.g. `Zscore_MA50` 1-day t=**-7.01**). 15/32 signal×horizon combos exceeded |t|>3.
- Verdict: real statistical pattern detected, but this was a diagnostic scan on raw daily observations, not a cost-aware, tradeable strategy test. Motivated Test 4.

## Test 4 — Quantile test: 20-day z-score reversal — **SPLIT VERDICT** (led to Test 5)
- Script: `quantile_test.py`
- Universe: 13 stocks
- Analysis A (pooled time-series quintile spread, Q1 minus Q5): survives 0.2% round-trip cost at 5/10/20-day horizons.
- Analysis B (cross-sectional, market-neutral long/short — the methodologically correct formulation given correlated stocks): **fails gross at every rebalance frequency** (Sharpe 0.04 to -0.46).
- Verdict: the divergence between A and B suggested the apparent edge in A was a market-timing artifact, not genuine stock-specific reversal. Motivated Test 5.

## Test 5 — Beta decomposition — **REJECTED** (confirmed market-timing artifact)
- Script: `beta_test.py`
- Universe: 13 stocks + SPY (IBKR)
- Concentration test: forward 10-day returns rise monotonically with the number of simultaneously-oversold stocks (-0.66% at 1-3 stocks oversold, up to +4.62% at 10+) — the signature of market-wide dip-buying, not stock selection.
- Long-only Q1 strategy vs SPY vs equal-weighted universe: **4.0% vs 47.6% vs 11.2%** total return — badly underperforms both benchmarks.
- Alpha regression: beta **1.035**, annualized alpha **-14.96%**, t=**-1.17**.
- Verdict: the entire "reversal edge" from Tests 3-4 was market beta exposure, not alpha. Rejected.

## Test 6 — Scale test: expanded universe (13→36 stocks) market-neutral reversal — **REJECTED** (confirmed at scale)
- Script: `scale_test.py` (+ `expand_universe.py` for the universe expansion)
- Universe: 36 stocks (2.8x expansion; 91/126 candidates failed mechanical price/ADV/history filters — an honest result, not a shortfall in effort)
- Decile long/short (bottom/top decile by 20-day z-score) at 1/5/10/20-day rebalance: **all |t-stat| < 0.4**, unprofitable gross at every frequency (total returns -23% to -83% over the practice window).
- Verdict: sample size was **not** the binding constraint — the null result held (if anything strengthened) at 2.8x scale. This closed out all short-horizon price-reversal testing on the IBKR-sourced 13/36-stock universe.

## Test 7 — Momentum 12-1 — **REJECTED** (beta, not alpha)
- Script: `factor_test.py`
- Universe: 600-ticker yfinance pilot (decile ~9.3 stocks — too small to trust), then 2,500-ticker yfinance sample (decile ~38 stocks), 2006-01 to 2018-10/12 practice window
- Long/short dollar-neutral: fails at both scales (t=-0.33 pilot, t=-0.44 at scale).
- Long-only decile: looks fine gross (+175.6% over the practice window) but underperforms both the equal-weighted universe and SPY on Sharpe.
- Alpha regression: beta **1.174**, annualized alpha **0.56%**, t=**0.13** — not remotely significant.
- Verdict: the long-only momentum decile's apparent outperformance is just elevated beta exposure. Rejected. **Not re-tested in the holdout** (already rejected pre-holdout, per pre-registration — do not spend holdout data on this).

## Test 8 — Low-volatility long-only — **PASSED practice window → DISAPPEARED in holdout (FINAL)**

**Practice window** (`factor_test.py`, 2008-01 to 2018-10, 2,500-ticker universe, ~38 stocks/decile):
- Long/short dollar-neutral: fails (t=-0.92) — volatility drag from a noisy (~9% monthly std) spread series masks a genuine long-only effect.
- Long-only decile: beta **0.662**, annualized alpha **+4.23%**, t=**2.47**, R²=0.78, net-of-0.2% Sharpe **0.88** vs SPY Sharpe 0.64. **Passed pre-registration.**

**Robustness checks** (`robustness_test.py`, practice window only, holdout still sealed):
- Sub-period stability: alpha positive in both 2008-2012 (t=1.26) and 2013-2018 (t=1.82) halves — not individually significant (reduced sample per half), but consistent sign/magnitude, not a crisis artifact.
- Crisis exclusion (2008-09 to 2009-06 dropped): alpha **strengthens** (t=**3.18**) — confirms not crisis-dependent.
- Sector concentration: not excessive — Financial Services leads at 24.1% average weight, diversified across 12 sectors (>40% single-sector concentration flag not triggered).
- Position-count sensitivity: N=38 passes (t=2.13); **N=20 and N=10 fail** (t=1.81, 1.77) — effect weakens below the full-decile size.
- Small-account commission drag ($4,700 account, $1/order minimum): 38 stocks ≈ 4.04%/yr drag, netting 6.20%/yr (vs SPY 8.97%/yr); 20 stocks ≈ 2.50%/yr drag, netting 7.48%/yr.

**Tradeability test** (`tradeable_test.py`, quarterly rebalance, 20 stocks, practice window):
- Cut commission drag to 1.49%/yr (from 4.04%), but fully-net alpha **collapsed** (t=0.94) — signal decay over the 3-month hold outweighed the cost savings.
- DIY bot (8.33% net annualized) underperformed both SPY (9.99%) and a hypothetical zero-trading-cost min-vol ETF proxy (10.00%, Sharpe 0.88).

**HOLDOUT TEST** (`holdout_test.py`, 2019-01 to 2026-04, one-shot, sealed since the project's first fetch):
- Alpha **flipped sign**: **-5.60%/yr** (t=**-1.62**, not significant), beta rose to **0.773**, net Sharpe fell to **0.46**.
- SPY returned **16.77%/yr** (Sharpe 1.00) over the identical window — the strategy badly underperformed simply holding the index.
- **VERDICT: the effect DISAPPEARED out of sample. Rejected as a tradeable strategy. The holdout is spent — do not re-run a holdout test on this universe/window.**

---

## Test 9 — SEC 8-K event study — **REJECTED** (real but uneconomic)
- Scripts: `event_study.py` (v1, confounded), `event_study_v2.py` (corrected); data via `fetch_edgar_cik_map.py`, `fetch_edgar_8k_bulk.py`, `fetch_edgar_8k_fill_gaps.py`
- Universe: 1,807 filers, 124,419 events with usable forward-return windows, 2019-01 onward
- **v1 methodology error, caught and corrected:** abnormal return was defined as stock return minus SPY return. Our universe underperformed SPY by ~5%/yr as a baseline, so the apparent "event effect" was largely the size premium running backwards. v2 replaced the benchmark with a matched non-event control (each stock's own mean forward return from days not within 20 trading days after any of its 8-K filings), differencing out idiosyncratic drift without a market benchmark.
- v2 pooled result: only the 10-day horizon significant (mean adjusted AR **-0.171%**, t=**-2.77**) — below the 0.2% round-trip cost bar.
- Best item code: **5.02** (executive/director departures), 10d **-0.406%** (t=-3.14), 20d **-0.412%** (t=-2.25) — clears the cost bar on paper.
- **Economic reality check at account size:** ~$470 position (50k SEK ÷ 10) → $2 commission = 0.43% + ~0.20% spread = **0.63% total cost vs 0.41% edge. Net negative before the signal even has to be real.** Would require ~200,000 SEK to become marginally economic.
- Size hypothesis **reversed**: the well-covered (high dollar-volume) half showed stronger and more persistent effects than the under-covered half — the opposite of the project's original small-cap edge thesis.
- Caveat: t-statistics are inflated by event clustering (8-Ks bunch in earnings season, so overlapping 20-day windows are far from independent) and by a control-group selection issue (frequent filers have few "clean" days, so their baseline is drawn from unusually quiet periods).
- Verdict: effects are real but too small to trade at this account size, and the methodology has unresolved confounds. Rejected.

## Test 10 — Value (Book-to-Market) + Quality (Gross Profitability) — **REJECTED** (underpowered, not disproven)
- Scripts: `fetch_edgar_fundamentals.py`, `fundamental_test.py`
- Data: SEC EDGAR XBRL bulk `companyfacts.zip` (1.30 GB), 1,043,209 fact-rows, 1,981/2,339 companies matched, filed dates 2009-04 onward
- **Lookahead protection:** every fact carries both its fiscal `end` date and its `filed` date; a fact is only usable from its filed date onward. Annual June formation, held 12 months, monthly returns measured on the fixed portfolio.
- Practice window 2010-06 to 2018-06 — 9 annual rebalances, 102 monthly observations

| | Value (B/M) | Quality (Gross Profitability) |
|---|---|---|
| Avg stocks/decile | 20.4 ⚠️ | 13.8 ⚠️ |
| Gross annualized | 21.15% | 14.15% |
| Net 0.2% + $1/order | 20.55% | 13.73% |
| Sharpe (fully net) | 1.20 | 0.81 |
| Beta | 1.159 | 0.979 |
| Annualized alpha | **+5.55%** | +1.94% |
| t-stat on alpha | **1.52** | 0.39 |

- Quality: rejected outright — t=0.39, and the long decile (14.15%) underperformed the equal-weighted universe (16.55%).
- Value: alpha is economically meaningful but misses the pre-registered t>2 bar. **Power analysis: detecting a ~5.5% alpha at this noise level requires ~192 monthly observations (~16 years). We have 102**, because XBRL tagging only phased in ~2009–2011. This is a hard data ceiling, not a fixable effort problem.
- Both deciles fall below the 25-stock trust threshold set in advance for this run.
- Verdict: rejected per pre-registration. Value is **unresolvable with available data** rather than disproven — the threshold was not loosened after seeing the result.

**Data-integrity correction (found building the Phase 6 forward test, applied retroactively):**
`us-gaap:CommonStockSharesOutstanding` is unreliable for a meaningful slice of filers — Up-C/holding-company registrants, foreign private issuers on 20-F/6-K, and redomiciled entities sometimes tag a technical share class rather than total float (e.g. SPG showed "8,000 shares" against a real ~325M-share count; several current-day top-B/M picks showed ratios in the hundreds of thousands, an impossibility for a real company). Fixed with a data-integrity floor in `value_signal`: reject any observation where shares outstanding is smaller than one day's average trading volume (implied >100%/day turnover is not economically plausible). Re-running the historical backtest with this fix barely moved the numbers — annualized alpha **+5.28% → +5.55%**, t-stat **1.46 → 1.52**, decile size 20.7 → 20.4 — so the original verdict stands; the bug happened not to contaminate the historical top-decile picks materially. It DID materially corrupt live/current-day ranking (see Phase 6 section below), which is why it was caught before being traded rather than after.

---

## Phase 6 — Forward Test (Value / Book-to-Market) — IN PROGRESS

**Start date:** 2026-07-31 — specification locked and 20 buy orders submitted same day. Markets were closed at submission time; orders were placed GTC and are resting (`PreSubmitted`), expected to fill at the next market open (2026-08-03). Inception equity and SPY baseline are captured by `ops/value_rebalance.py` only once real capital is confirmed deployed (an actual fill observed, or a pre-existing position found) — not merely on order submission — so the forward-test clock starts from real exposure, not paperwork.

**Why forward, not more backtesting:** Value (B/M) is the one unresolved candidate from Test 10 — alpha **+5.55%/yr** (post data-integrity fix below), t=**1.52**, economically meaningful but statistically underpowered (102 monthly observations vs the ~192 needed). The backtest dataset is fully spent (see Evidence-Boundary section below). Forward testing from today onward is the only remaining source of uncontaminated evidence.

**Exact specification (fixed as of today):**
- Universe: same filters as `fundamental_test.py` (price $5–100, monthly avg daily volume > 250,000, 24+ months history), applied to `data/yf_universe.parquet`'s latest available month.
- Signal: Book-to-Market = StockholdersEquity ÷ (price × CommonStockSharesOutstanding), using only fundamental facts filed by the rebalance date, with the data-integrity floor (shares outstanding ≥ one day's average volume).
- Selection: **top 20 by B/M** — a deliberate, pre-registered deviation from the backtest's ~20-stock decile, sized for this account's commission drag (Test 8: 4.04%/yr at 38 stocks vs 2.50%/yr at 20).
- Weighting: equal-weighted, ~90% of account equity invested (~10% cash buffer).
- Rebalance: annual, each June, held fixed for 12 months — identical mechanics to the backtest.
- Constraint: individual US common stocks only. PRIIPs/KID blocks ETF purchases on this account (confirmed by SPY's IBKR error 201 rejection during Phase 6 infrastructure validation) — immaterial here since Value is a stock-selection strategy, not an index proxy.

**Benchmark:** SPY total return over the identical window, tracked as a reference number via IBKR historical data (`ops/value_report.py`) — never bought, never held, because it can't be on this account.

**Success criterion:** the portfolio beats SPY, net of all costs (0.2% + $1/order commissions), over a multi-year horizon — specifically once ~192 monthly observations accumulate (~16 years at 12 monthly readings/year), the same power threshold the backtest fell short of. Interim monthly readings before that point are informative but not conclusive, and `ops/value_report.py` reports them as such every time.

**Commitment:** this specification — universe filters, B/M definition, top-20 cut, equal-weighting, the 90% equity target, and the annual June rebalance — is fixed as of 2026-07-31 and will not be modified in response to interim results. Changing methodology after seeing forward-test performance would reintroduce exactly the data-mining risk this whole research program has spent ten backtests learning to avoid.

Scripts: `ops/value_portfolio.py` (generate target), `ops/value_rebalance.py` (execute), `ops/value_report.py` (monthly tracking, since-inception performance, running observation count).

---

## Evidence-boundary status — IMPORTANT

**No untouched window remains in this dataset.** 2019+ was spent on the Test 8 low-volatility holdout, then used again as the Test 9 event-study analysis window. Pre-2019 served as the practice window for Tests 7, 8, and 10.

Beyond the data itself, the *designer* is contaminated: facts now known about 2019–2026 (SPY returned 16.77%/yr, low-volatility inverted, growth dominated value) inevitably leak into how any further test would be specified. **Any additional testing on this dataset is data-mining, regardless of how carefully the code is written.**

Resuming this research requires new data — see `ROADMAP.md` Section 8 for the two specific triggers.

---

## Bottom line

Ten hypothesis classes tested — SMA crossover, RSI mean-reversion, 20-day z-score
reversal (13 and 36-stock universes, both time-series and cross-sectional
formulations), momentum 12-1, low-volatility, SEC 8-K material events, and
fundamental value/quality — and every one rejected, either in-sample, in the
one-shot holdout after passing every practice-window check (low-volatility), on
economic grounds after clearing statistical ones (8-K events), or on
sample-size grounds that the available data cannot fix (value).

No signal from this research program is validated for live trading, and the
dataset can no longer support an honest eleventh test.
