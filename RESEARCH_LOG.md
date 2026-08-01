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

## Test 11 — Fugazzi retest (prior CAPM/Jensen's-alpha stock-picking exercise) — **REJECTED** (positive but not significant; ~20% hindsight-driven)
- Script: `fugazzi_retest.py`
- Fixed a bug in 3 of 4 original scripts: they tested on the 2021–2023 TRAINING window by mistake (START_DATE bug), not the intended 2024–2025 walk-forward window. SPY, which the original project never benchmarked against, is added here.
- Original portfolio: NVDA, AVGO, LLY, XOM, WMT, GOOGL, fixed weights [0.19, 0.205, 0.276, 0.211, 0.019, 0.10] — chosen with hindsight through 2025, not just through the training window.
- **Test 1 (original, OOS 2024–2025):** total return **+146.09%**, annualized **+49.60%**, Sharpe 1.71, max DD -24.96%.
- **Test 2 (SPY, same window):** total **+48.95%**, annualized **+21.42%**, Sharpe 1.31. Alpha regression: beta **1.393**, annualized alpha **+19.76%**, R²=0.622, **t=1.56** — positive but not significant.
- **Test 3 (honest reselection):** Jensen's alpha computed per stock using ONLY 2021–2023 data (beta from 2-year weekly returns vs ^GSPC, risk_free=0.045, market_ret=0.10), top 6 picked as of end-2023: NVDA, LLY, XOM, AVGO, CVX, AMD (4/6 overlap with the original). Equal-weighted: total **+116.23%**, annualized **+43.32%**, Sharpe 1.45. **Hindsight advantage: +29.86pp total return, +0.26 Sharpe** — roughly a fifth of the original's apparent edge came from picking winners after already knowing they'd won.
- **Test 4 (equal-weight all 30, no-skill baseline):** total **+68.24%**, annualized **+27.61%**, Sharpe 1.66. Both the original portfolio and the honest reselection beat this baseline too, so concentration wasn't the whole story.
- Verdict: directionally positive vs SPY but alpha not statistically significant (t=1.56); a real, quantifiable slice of the apparent edge was lookahead-driven. Rejected as a demonstrated-skill claim.

## Test 12 — Swedish CAPM retest (SPOT, ERIC, AZN, ALV, OTLY) — **REJECTED**
- Script: `swedish_retest.py`
- Same conventions as Test 11 (daily returns, no rebalancing, 2024–2025 OOS window).
- **(a) Equal-weighted:** total **+55.07%**, annualized **+24.59%**, Sharpe 1.11. Alpha regression vs SPY: beta 0.817, alpha **+7.09%**, **t=0.56**.
- **(b) CAPM-weighted** (pre-test 2-year weekly beta vs ^GSPC, expected_return = 0.045 + beta×0.055, weight = expected_return / Σexpected_return, computed only from 2022–2023 data): total **+51.03%**, annualized **+23.85%**, Sharpe 0.96. Alpha: beta 0.864, alpha **+5.34%**, **t=0.37**. Weights: SPOT 22.9%, ERIC 18.1%, AZN 12.1%, ALV 17.5%, **OTLY 29.3%** — the highest weight went to OTLY purely because of its high pre-test beta (2.31), despite OTLY then losing **56%** of its value in the test window from a confirmed-real (not split-artifact) decline.
- **(c) SPY:** total +48.95%, annualized +21.42%, Sharpe 1.31.
- Verdict: neither variant clears significance (t=0.56, t=0.37) — the modest edge over SPY is noise-level. Rejected.

## Test 13 — FinBERT sentiment IC scan (pooled, average-effect diagnostic) — **REJECTED** (clean null)
- Scripts: `check_alpaca_news_api.py` (coverage/rate-limit check), `sentiment_test.py`
- Universe: the 11 tickers from Tests 11–12 (6 US + 5 Swedish ADRs), 2021–2025, 40,209 unique Alpaca articles, every headline scored with FinBERT (`ProsusAI/finbert`, CPU).
- Article volume heavily skewed toward GOOGL (22,028) and NVDA (14,755) vs. the Swedish names (375–2,633) — a limitation that directly motivated Test 14's wider universe.
- IC analysis (Spearman, `ic_analysis.py`'s methodology): sentiment vs forward returns at 1/5/10/20 trading days, both a cross-stock t-test (11 independent per-stock ICs) and a pooled t-test.

| Horizon | Cross-stock mean IC | Cross-stock t | Pooled IC | Pooled t |
|---|---|---|---|---|
| 1d | +0.002 | 0.11 | -0.000 | -0.03 |
| 5d | -0.005 | -0.27 | -0.010 | -0.92 |
| 10d | -0.007 | -0.45 | -0.005 | -0.45 |
| 20d | -0.001 | -0.07 | -0.012 | -1.14 |

- Verdict: zero of 4 horizons clear |t|>2 on either measure — a flat null, not a close call. No detectable AVERAGE relationship between news sentiment and forward returns. Rejected.

## Test 14 — Conditional shock-day sentiment test — **REJECTED** (attention effect found but not sentiment-specific; the one sentiment-direction result failed independence/liquidity robustness checks)
- Scripts: `fetch_alpaca_news_wide.py`, `sentiment_shock_test.py`
- **Motivation:** Test 13 measured the AVERAGE effect across every day and came back null — but pooling dilutes toward zero any effect concentrated in rare high-impact events, by construction. This asks the conditional question Test 13's design cannot answer: does sentiment predict returns specifically when something significant happens?
- **Wider universe (Task 1):** 100 tickers randomly sampled (seed 42) from `data/yf_universe.parquet`, headlines+dates only, 2021 to present — 20,379 articles, 87/100 tickers with coverage, median 113 articles/ticker (far more balanced than Test 13's GOOGL/NVDA-dominated set; AMC was the max at 4,191).
- **Shock-day identification (Task 2):** a "shock day" = article count ≥ 3× that ticker's own median (over active-news days). **1,782 shock days** found across 83 tickers with a definable median (15.27% of active news-days, 0.72% of all ticker-trading-days). Only shock-day headlines were FinBERT-scored (8,848 unique; 571 reused from Test 13's cache, 8,277 freshly scored) — far cheaper than rescoring the full corpus.
- **Conditional event analysis (Task 3):** matched non-event control reused from `event_study_v2.py` — adjusted_AR = shock-day forward return − that ticker's own mean forward return on non-shock days. Terciles cut on the pooled shock-day sentiment distribution (594 events each).

| Bucket | n | 1d | 5d | 10d | 20d |
|---|---|---|---|---|---|
| Negative tercile | 594 | **-1.685%** (t=-3.94) | -1.369% (t=-1.29) | -1.024% (t=-0.56) | **-4.407%** (t=-2.52) |
| Positive tercile | 594 | +0.176% (t=0.28) | -0.196% (t=-0.19) | -0.939% (t=-0.66) | **-2.863%** (t=-3.14) |
| Neutral tercile | 594 | -0.605% (t=-1.47) | -0.518% (t=-0.60) | -0.856% (t=-0.75) | **-3.584%** (t=-2.33) |
| All pooled | 1,782 | **-0.710%** (t=-2.47) | -0.697% (t=-1.22) | -0.940% (t=-1.09) | **-3.623%** (t=-4.33) |

- **Key finding, and the important caveat:** at the 20-day horizon, ALL FOUR buckets show significant negative abnormal returns — including the POSITIVE-sentiment tercile (t=-3.14). This means the 20-day drift is not specific to bad news; it tracks unusual attention itself, regardless of whether FinBERT scored the coverage as positive, negative, or neutral. The 1-day horizon is the one place sentiment direction shows up cleanly: only the negative tercile is significant (t=-3.94), consistent with an immediate bad-news reaction that fades by day 5–10 before the (sentiment-independent) 20-day drift reappears.
- **Risk-filter test (Task 4):** for negative-sentiment shocks only, compared against each ticker's UNCONDITIONAL mean forward return (a different, simpler baseline than Task 3's non-shock-day control).

| Horizon | n | Mean diff | t-stat | vs 0.63% cost |
|---|---|---|---|---|
| 5d | 592 | -1.354% | -1.28 | not significant |
| 10d | 590 | -0.987% | -0.54 | not significant |
| 20d | 589 | **-4.296%** | **-2.46** | **exceeds 0.63% cost** |

- The 20-day negative-shock-vs-unconditional result is statistically significant and numerically large enough to clear the round-trip cost floor. **However, given Task 3's finding that the SAME 20-day negative drift also appears after positive-sentiment shocks, this result cannot be cleanly attributed to sentiment direction** — it may just be "this stock had an unusual news event," full stop, rather than "the news was bad." Treating this as a validated sentiment-based exit rule would overstate what was actually shown.
- **Sample-size honesty (Task 5):** every bucket cleared the 100-event floor (594 per tercile, 1,782 pooled) — none of the above is underpowered.
- Interim verdict: the 20-day pattern is an attention effect, not a sentiment-direction effect (see above). The 1-day horizon was the one place sentiment direction showed up cleanly (negative tercile t=-3.94, positive tercile t=0.28) — that result was carried forward to a dedicated robustness check rather than accepted at face value.

**Robustness check on the 1-day effect (five adversarial tests, script: `sentiment_shock_robustness.py`), reconstructed from cached data, no refetching/rescoring, tercile cutoffs and shock threshold held fixed throughout:**

| Check | n | Mean | t-stat | Result |
|---|---|---|---|---|
| Original (unrestricted) | 589 | -1.725% | -4.02 | baseline |
| 1. Declustered (≥10 trading days apart per ticker) | 335 | -0.769% | **-1.60** | **FAILS** |
| 2. Liquid half (60d $ volume, median split) | 266 | -0.184% | **-0.34** | **FAILS** |
| 2. Illiquid half | 298 | -2.136% | -3.49 | (survives, but untradeable) |
| 3. Ex-meme (AMC/SCHW/SE/VST/GH excluded) | 417 | -2.250% | -4.40 | survives |
| 4. Volatility-matched baseline (same ticker, same vol tercile) | 564 | -1.274% | -3.05 | survives (weaker) |

(Small ~1% drift in n/mean vs. the table above this line is a reproducibility artifact — a handful of shock days in the last few days of the fetched news window fall after the price panel's data cutoff and get excluded slightly differently by the two scripts. It doesn't change any conclusion.)

- **Check 1 (independence) is decisive on its own:** most of the original t=-3.94 significance was pseudo-replication — clustered, non-independent shock days (a company gets a burst of headlines, not one isolated article) were being counted as separate draws. Properly declustered, the effect nearly halves and loses significance (t=-1.60).
- **Check 2 (liquidity) independently confirms it isn't tradeable:** the liquid half shows essentially nothing (t=-0.34); whatever raw signal exists lives entirely in illiquid names.
- **Check 3 (meme contamination) is not the explanation** — excluding the 5 most-covered tickers *strengthens* the effect, meaning meme-stock noise was diluting it, not causing it.
- **Check 4 (volatility control)** shows a real, weaker residual effect after controlling for the fact that shock days are higher-volatility days generally (t=-4.02 → -3.05) — some of the raw effect, but not all of it, was just a volatility artifact.
- **Check 5 (implementation, using the declustered rate/size as the honest base case):** ~6.13 negative shocks/year on a 20-stock equal-weighted portfolio (the live value portfolio's size), net benefit per event **+0.14%** after the 0.63% round-trip cost, working out to **+0.04 percentage points/year at the whole-portfolio level** — negligible even setting aside that Check 1 already fails significance.
- **Final verdict: does NOT survive.** 2 of 4 adversarial checks fail outright, and one of the two failures (independence) is disqualifying by itself — a genuinely tradeable signal must be significant on correctly-counted independent observations, and this isn't. The other failure (liquidity) independently confirms it wouldn't be tradeable at real size even if it were. **Rejected — this does not reach a risk-filter layer on the live value portfolio.** No shock threshold, tercile cut, or additional horizon was altered to produce this conclusion.

## Test 15 — Volatility targeting (Moreira & Muir 2017) — **PRACTICE WINDOW: Sharpe improvement present, not statistically significant — HOLDOUT UNTOUCHED**
- Script: `vol_target_test.py`
- Different in kind from Tests 1–14: times exposure to the market rather than predicting direction. Disputed in the literature — Cederburg et al. found it fails out-of-sample for most factors. Prior was low going in; the project's record was 0/14.
- Data: SPY daily history from inception (1993-01-29), fetched fresh via yfinance — the deepest history used anywhere in this project (8,433 raw rows), specifically to span multiple volatility regimes (dot-com crash, 2008 financial crisis).
- Spec (fixed, no variants): monthly rebalance at the open, exposure = 15% target vol ÷ trailing-21-day realized vol, capped [0, 1.5×], 5%/yr margin cost on the borrowed portion, 0.05% round-trip cost on the exposure change.
- Practice window 1993 through 2009-12 — **200 usable monthly observations, the largest sample of any test in this project.**

| | Vol-targeted | SPY buy&hold | Constant leverage (avg 103.6%) |
|---|---|---|---|
| Total return | 268.15% | 230.00% | 231.55% |
| Annualized | 8.13% | 7.43% | 7.46% |
| Sharpe | **0.66** | 0.54 | 0.53 |
| Max drawdown | **-39.03%** | -51.30% | -52.76% |

- Average exposure 103.6%, average monthly turnover 18.7pp — small, as expected for a monthly single-instrument strategy.
- **Sharpe(vol-targeted)/Sharpe(SPY) = 1.217; Sharpe(vol-targeted)/Sharpe(constant leverage at the same average exposure) = 1.244.** The improvement beats a same-average-leverage benchmark too, not just plain SPY — meaning it isn't just "this exposure level happened to do well," it's specifically attributable to varying exposure over time, which is the actual Moreira & Muir claim (Sharpe ratio is leverage-invariant in the textbook case; only time-varying exposure should move it).
- Max drawdown improved substantially (-39% vs -51%) — consistent with the mechanism: exposure gets cut during high-vol regimes like 2008, exactly when drawdowns are worst.
- **Alpha regression** (vol-targeted, net of costs, vs SPY, monthly): beta **0.771**, annualized alpha **+2.26%**, R²=0.820, **t=1.59** — positive, but does not clear this project's t>2 significance bar.
- Verdict: **the first result in fifteen tests where the Sharpe-ratio comparison (the actual claim under test) holds up against two separate honest baselines** — but the formal alpha t-stat, on the largest sample in this project, is not significant at the conventional threshold. Positive, not proven — the same standard applied to every other borderline result here (Fugazzi t=1.56, Swedish t=0.56/0.37, Test 10 Value t=1.52). **The 2010+ holdout was not read anywhere in this script and remains sealed.** Whether to spend it on this one result is a deliberate, one-shot decision (see Test 8's precedent) — not made here.

---

## Evidence-boundary status — IMPORTANT

**No untouched window remains in this dataset.** 2019+ was spent on the Test 8 low-volatility holdout, then used again as the Test 9 event-study analysis window. Pre-2019 served as the practice window for Tests 7, 8, and 10.

Beyond the data itself, the *designer* is contaminated: facts now known about 2019–2026 (SPY returned 16.77%/yr, low-volatility inverted, growth dominated value) inevitably leak into how any further test would be specified. **Any additional testing on this dataset is data-mining, regardless of how carefully the code is written.**

Resuming this research requires new data — see `ROADMAP.md` Section 8 for the two specific triggers.

---

## Bottom line

Fifteen hypothesis classes tested — SMA crossover, RSI mean-reversion, 20-day
z-score reversal (13 and 36-stock universes, both time-series and
cross-sectional formulations), momentum 12-1, low-volatility, SEC 8-K material
events, fundamental value/quality, two independent CAPM stock-picking retests
(Fugazzi, Swedish ADRs), a pooled news-sentiment IC scan, a conditional
shock-day sentiment test, and volatility targeting. Fourteen of the fifteen
were rejected as a tradeable edge, either in-sample, in the one-shot holdout
after passing every practice-window check (low-volatility), on economic
grounds after clearing statistical ones (8-K events), on sample-size grounds
that the available data cannot fix (value), on significance grounds despite a
positive point estimate (Fugazzi, Swedish), as a clean null (pooled sentiment
IC), or as real-but-not-actually-about-sentiment once decomposed (Test 14's
20-day pattern tracked news attention generally, not sentiment direction,
once checked against the positive-sentiment bucket), or as a significant-
looking result that failed to survive adversarial robustness checks (Test
14's 1-day sentiment-direction effect lost significance once clustered shock
days were properly declustered, and separately showed no effect at all in
liquid, tradeable names).

**Test 15 (volatility targeting) is the one exception, and it is explicitly
NOT a validated result — it is an open practice-window finding with a sealed
holdout.** Sharpe improved over both plain SPY and a same-average-leverage
benchmark on the largest sample in this project (n=200 monthly
observations), but the formal alpha t-stat (1.59) does not clear this
project's own significance bar. It sits in the same "positive but not proven"
category as Fugazzi and the Swedish retest, with one difference: unlike
those, its holdout has not been touched. Whether to spend that one-shot
holdout is a decision for the user, not something this log resolves on its
own — see Test 15's entry for the precedent Test 8 set on why that decision
is treated as consequential.

No signal from this research program is validated for live trading. The
original backtest dataset (Tests 1–10) can no longer support an honest new
test on it; Tests 11–15 used independent, freshly-sourced data (fresh
per-ticker yfinance pulls, Alpaca news, SPY history from 1993) specifically
to keep generating uncontaminated evidence after that boundary was reached.
