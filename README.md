# Systematic Trading Research — Ten Strategies Tested, Ten Rejected

A complete, honestly-reported quantitative research record: ten trading strategies
built, tested under pre-registered specifications, and rejected on the evidence.

**This repository does not contain a profitable trading strategy.** It contains the
apparatus that proved ten strategies weren't — including one that passed every
robustness check before a sealed holdout killed it. That negative result, and the
discipline that produced it, is the point.

---

## Why a negative result is worth publishing

Anyone can post an equity curve. Almost every public "trading bot" repo shows a
backtest that looks good, because the ones that don't look good never get published.
That selection effect is why most of them are worthless.

This repo documents the opposite: what happens when you set the bar honestly and
report what you find. Every test here includes:

- **Pre-registration** — the exact specification (signal, parameters, costs,
  holding period) written down *before* the test ran. No post-hoc parameter picking.
- **Null-hypothesis testing** — strategies benchmarked against thousands of simulated
  random traders matched on trade count and holding period. Beating buy-and-hold
  isn't enough; you have to beat luck.
- **Evidence boundaries** — data split into a practice window and a sealed holdout,
  with the holdout spent exactly once, at the end, with the interpretation agreed in
  advance.
- **Realistic cost modeling** — including the $1.00/order commission floor, which at
  a small account size turned out to be the single most decisive constraint in the
  entire project.
- **Survivorship-bias accounting** — the direction of the bias is stated per factor,
  and one factor (long-term reversal) was dropped entirely because free data cannot
  test it honestly.

---

## The tests

| # | Strategy | Key statistic | Verdict |
|---|---|---|---|
| 1 | SMA 20/50 crossover | 44.2% avg monkey-beat (worse than random) | Rejected |
| 2 | RSI(14) mean-reversion | Pooled t = 0.34; +1.19% over 525 days; Sharpe 0.13 | Rejected |
| 3 | IC scan (8 signals × 4 horizons) | Found short-term reversal, mean IC ≈ −0.05 at 1 day | Diagnostic |
| 4 | Quantile / cost-survival test | Time-series spread survived costs; market-neutral version failed *gross* | Rejected |
| 5 | Beta decomposition | Alpha −14.96%, t = −1.17, beta 1.04 — the "edge" was market timing | Rejected |
| 6 | Scale test (36-stock universe) | All \|t\| < 0.4; unprofitable before costs | Rejected |
| 7 | Momentum 12-1 | Alpha 0.56%, t = 0.13, beta 1.17 | Rejected |
| 8 | **Low volatility (long-only)** | **Practice: alpha +4.23%, t = 2.47 → Holdout: alpha −5.60%, t = −1.62** | **Rejected in holdout** |
| 9 | SEC 8-K event study | Effects real but ≈0.4% vs ≈0.63% round-trip cost at account size | Rejected |
| 10 | Value (B/M) + Quality (gross profitability) | Value alpha +5.28% but t = 1.46; needs ~16yr of data, XBRL gives 8.5 | Rejected — underpowered |

Full details, statistics, and reasoning for each: **[`RESEARCH_LOG.md`](RESEARCH_LOG.md)**
Current status and decision framework: **[`ROADMAP.md`](ROADMAP.md)**

---

## The one that nearly made it

Test 8 is the interesting one.

Low-volatility long-only passed **every** check designed to break it:

- Practice window (2008–2018): alpha +4.23%/yr, t = 2.47, Sharpe 0.88 net of costs
- Sub-period stability: positive alpha in both halves
- Crisis exclusion: removing 2008–09 made it *stronger* (t = 3.18, clearing even the
  strict Harvey-Liu-Zhu bar of 3.0)
- Sector concentration: diversified across 12 sectors, no single-sector bet
- Position-count sensitivity: alpha stable from 10 to 38 holdings

Then the sealed 2019–2026 holdout: **alpha −5.60%, Sharpe 0.46, beta drifted 0.66 →
0.77.** The effect didn't weaken. It inverted.

Even gross of all costs, the strategy returned 6.81%/yr while SPY returned 16.77%/yr
over the same window.

**Without the holdout discipline, this would have gone to paper trading and then
live.** That's the entire argument for the methodology in one example.

---

## Findings that generalise beyond this project

**Commission drag is the binding constraint at small account sizes.** At ~$4,700
with a $1.00/order minimum: 4.04%/yr drag at monthly rebalancing across 38 positions,
1.49%/yr at quarterly across 20. A real 4% alpha does not survive that. The break-even
account size for the strategies tested here is roughly 4× the capital available.

**Statistical significance ≠ economic significance.** The 8-K event study produced
t-statistics as extreme as −12 on effects of 0.4%, against round-trip costs of 0.63%.
Significance is cheap at n > 100,000; tradeability is not.

**Benchmarks confound results.** The first version of the event study measured
abnormal returns against SPY, and appeared to find a broad negative post-filing drift.
It was measuring the size premium — the test universe underperformed SPY by ~5%/yr as
a baseline. Corrected with a matched non-event control, most of the effect vanished.

**Published edges decay.** The low-volatility anomaly was documented in the 1970s and
became investable via min-vol ETFs around 2011. Its failure in the 2019–2026 holdout
is consistent with an effect that has been widely harvested.

---

## Repository structure

```
bot/                        Core system
  broker/                   IBKR client, order execution, fill monitoring, FX
  journal/                  SQLite trade journal + persistent risk state
  risk/                     Circuit breaker, position sizing
research/
  data_fetch/               Universe construction, price/EDGAR data pipelines
  signal_tests/             Tests 1–6 (technical signals)
  factor_tests/             Tests 7–8 (factors, robustness, holdout)
  event_tests/              Test 9 (SEC 8-K event study)
ops/                        Cleanup, audit, snapshots, verification, reporting
RESEARCH_LOG.md             Every test, statistic, and verdict
ROADMAP.md                  Status, constraints, decision framework
```

**Infrastructure verification:** `ops/verify_system.py` runs a 5-point check —
trade journal price integrity, equity snapshot logging, risk-state persistence across
restarts, circuit-breaker trigger logic (tested in-memory without corrupting stored
state), and broker/local position reconciliation.

---

## Data

Price data is not committed (the universe panel alone is ~197MB). All of it is
regenerable from free sources:

- **Prices:** yfinance, ~2,500 randomly sampled US common stocks, 2006–2026
- **Universe:** NASDAQ Trader symbol directories (free, no auth)
- **Filings:** SEC EDGAR bulk submissions archive + per-company API
- **Benchmark:** SPY via yfinance

Run the scripts in `research/data_fetch/` to rebuild.

**Known limitation, stated plainly:** the universe is currently-listed tickers only,
so it carries survivorship bias. The bias direction is documented per factor —
conservative for momentum (the worst losers you would short are absent), minimal for
low volatility, and fatal for long-term reversal, which is why that factor was never
tested.

---

## Environment

Python 3.14, pandas 3.0.3, numpy 2.5.1, ib_async, yfinance, pyarrow, SQLAlchemy.

Requires `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` — numpy/OpenBLAS otherwise
fails to allocate memory on this Python version.

Broker scripts require IB Gateway running on a paper account. Research and backtest
scripts require no broker connection.

---

## Status

Concluded and paused. The decision framework, including the specific conditions under
which this research would be worth resuming, is in [`ROADMAP.md`](ROADMAP.md).

Built over several weeks in 2026 as a personal research project.
