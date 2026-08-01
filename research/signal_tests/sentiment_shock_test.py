"""
sentiment_shock_test.py -- Test 14: does news sentiment predict returns
specifically around unusual-attention ("shock") days? Test 13's pooled
FinBERT IC scan measured average predictive power across ALL days and
came back null (RESEARCH_LOG.md) -- but averaging dilutes toward zero
any effect concentrated in rare high-impact events, by construction.
This is the conditional test that design cannot run.

TASK 2 -- shock-day identification (cheap, no scoring):
For each of the 100 tickers in data/alpaca_news_wide.parquet, the
median daily article count is computed over days with >=1 article
(days with zero news are excluded from the median -- otherwise a
mostly-silent ticker's median would be 0, making "3x median" a
degenerate threshold of 0 that flags every single article as a
"shock"). A "shock day" is any (ticker, date) with article count >=
3x that ticker's own active-day median.

Only shock-day headlines are sent to FinBERT -- everything already
scored in data/alpaca_news_scored.parquet (Test 13's 11-ticker corpus)
is reused via an id lookup instead of being rescored.

TASK 3 -- conditional event analysis. Matched non-event control method
reused from research/event_tests/event_study_v2.py: each ticker's own
mean forward return on its NON-shock trading days is that ticker's
baseline at each horizon; adjusted_AR = shock-day forward return -
that baseline. Unlike event_study_v2.py's 8-K version, no post-event
exclusion window is applied beyond the shock day itself -- a "day" is
what was asked to be excluded, not a window around it. Forward returns
use ic_analysis.py's exact convention: entry at day-after-shock's open,
exit at (day-after-shock + horizon)'s open -- no lookahead into the
shock day's own reaction.

Shock-day sentiment = mean FinBERT score (P(positive) - P(negative))
across that (ticker, day)'s headlines. Terciles are cut on the POOLED
distribution of shock-day sentiment scores (not tuned, not per-ticker).

TASK 4 -- risk-filter test: for negative-sentiment shock days only, the
SAME shock-day forward return is compared against that ticker's
UNCONDITIONAL mean forward return (all trading days, not just non-shock
ones) -- a deliberately different, simpler baseline than Task 3's,
per spec. Checked against a 0.63% round-trip cost floor (a $470
position, $1/order + spread -- RESEARCH_LOG.md Test 9's account-size
economics).

TASK 5 -- any bucket under 100 events is flagged as underpowered rather
than reported as if its t-stat were reliable.

No strategy is built, no shock threshold is tuned, and no bucket is
selected as "best" -- every bucket specified is reported, full stop.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import random
import numpy as np
import pandas as pd

UNIVERSE_PATH = os.path.join("data", "yf_universe.parquet")
SAMPLE_SIZE = 100
SEED = 42
PRICE_PATH = os.path.join("data", "yf_universe.parquet")
NEWS_WIDE_PATH = os.path.join("data", "alpaca_news_wide.parquet")
SCORED_CACHE_PATH = os.path.join("data", "alpaca_news_scored.parquet")     # Test 13's already-scored corpus
SHOCK_SCORED_OUT_PATH = os.path.join("data", "alpaca_news_shock_scored.parquet")

TICKER_ALIASES = {"GOOG": "GOOGL"}
HORIZONS = [1, 5, 10, 20]
SHOCK_MULTIPLIER = 3.0
MIN_ACTIVE_DAYS_FOR_MEDIAN = 2   # need at least 2 active-news days to define a meaningful median
MIN_BUCKET_TRUST = 100           # Task 5's honesty floor
ROUNDTRIP_COST = 0.0063          # $470 position, $1/order + spread (RESEARCH_LOG.md Test 9)
SCORE_BATCH_SIZE = 32

pd.set_option("display.width", 140)


def forward_return(opens, horizon):
    """Verbatim from ic_analysis.py: fwd[N] = open[N+1+horizon] / open[N+1] - 1."""
    n = len(opens)
    fwd = np.full(n, np.nan)
    last_n = n - 1 - horizon
    if last_n <= 0:
        return fwd
    idx = np.arange(0, last_n)
    fwd[idx] = opens[idx + 1 + horizon] / opens[idx + 1] - 1
    return fwd


def t_stat(vals):
    n = len(vals)
    if n < 2:
        return n, float("nan"), float("nan")
    mean = float(vals.mean())
    std = float(vals.std(ddof=1))
    t = mean / (std / np.sqrt(n)) if std > 0 else float("nan")
    return n, mean, t


# ============================================================
# LOAD
# ============================================================
print("=" * 100)
print("LOAD")
print("=" * 100)
news_wide = pd.read_parquet(NEWS_WIDE_PATH)
print(f"Wide news corpus (100-ticker sample, unscored): {len(news_wide)} articles")

price_raw = pd.read_parquet(PRICE_PATH)
price_raw["date"] = pd.to_datetime(price_raw["date"])
print(f"Price panel: {price_raw['date'].min().date()} to {price_raw['date'].max().date()}, "
      f"{price_raw['ticker'].nunique()} tickers")

# Regenerate the EXACT same 100-ticker sample fetch_alpaca_news_wide.py drew (same seed, same
# source list) -- this is the fixed universe for shock-day analysis. Articles are frequently
# multi-tagged with OTHER companies' tickers too (co-mentions); explode() surfaces every one
# of those, so without this restriction the analysis would silently drift onto thousands of
# tickers for which we only ever fetched INCIDENTAL, non-representative news coverage.
universe_tickers = sorted(pd.read_parquet(UNIVERSE_PATH, columns=["ticker"])["ticker"].unique())
drawn_sample = random.Random(SEED).sample(universe_tickers, SAMPLE_SIZE)
print(f"Regenerated 100-ticker sample (seed={SEED}), matches fetch_alpaca_news_wide.py's draw.")

exploded = news_wide.explode("symbols").rename(columns={"symbols": "symbol"})
exploded["symbol"] = exploded["symbol"].replace(TICKER_ALIASES)
sample_tickers = sorted(set(drawn_sample) & set(price_raw["ticker"].unique()))
exploded = exploded[exploded["symbol"].isin(sample_tickers)].copy()
exploded["date"] = pd.to_datetime(exploded["created_at"]).dt.tz_localize(None).dt.normalize().astype("datetime64[ns]")
print(f"Sample tickers with both news and price coverage: {len(sample_tickers)}")


# ============================================================
# TASK 2: SHOCK-DAY IDENTIFICATION
# ============================================================
print()
print("=" * 100)
print("TASK 2: SHOCK-DAY IDENTIFICATION")
print("=" * 100)

daily_counts = exploded.groupby(["symbol", "date"]).size().rename("n_articles").reset_index()

shock_rows = []
tickers_with_median = 0
for ticker, g in daily_counts.groupby("symbol"):
    if len(g) < MIN_ACTIVE_DAYS_FOR_MEDIAN:
        continue  # can't define a meaningful "typical day" from 0-1 active days
    median_count = g["n_articles"].median()
    if median_count <= 0:
        continue
    tickers_with_median += 1
    threshold = SHOCK_MULTIPLIER * median_count
    shocks = g[g["n_articles"] >= threshold].copy()
    shocks["median_count"] = median_count
    shocks["threshold"] = threshold
    shock_rows.append(shocks)

shock_days = pd.concat(shock_rows, ignore_index=True) if shock_rows else pd.DataFrame(
    columns=["symbol", "date", "n_articles", "median_count", "threshold"])

total_active_days = len(daily_counts)
total_trading_day_cells = price_raw[price_raw["ticker"].isin(sample_tickers)].groupby("ticker").size().sum()
print(f"Tickers with a definable median (>= {MIN_ACTIVE_DAYS_FOR_MEDIAN} active-news days): {tickers_with_median}")
print(f"Shock days found: {len(shock_days)}")
print(f"  as % of active (ticker, day-with->=1-article) cells: {len(shock_days) / total_active_days * 100:.2f}% "
      f"({len(shock_days)} / {total_active_days})")
print(f"  as % of all (ticker, trading-day) cells in the window: {len(shock_days) / total_trading_day_cells * 100:.3f}% "
      f"({len(shock_days)} / {total_trading_day_cells})")

per_ticker_shocks = shock_days.groupby("symbol").size().reindex(sample_tickers, fill_value=0).sort_values(ascending=False)
print(f"\nShock days per ticker -- min {per_ticker_shocks.min()}, median {per_ticker_shocks.median():.0f}, "
      f"mean {per_ticker_shocks.mean():.2f}, max {per_ticker_shocks.max()} ({per_ticker_shocks.idxmax()})")
print(f"Tickers with zero shock days: {(per_ticker_shocks == 0).sum()} / {len(sample_tickers)}")
print("\nTop 15 tickers by shock-day count:")
print(per_ticker_shocks.head(15).to_string())


# ============================================================
# TASK 2 (cont.): SCORE ONLY SHOCK-DAY HEADLINES
# ============================================================
print()
print("=" * 100)
print("SCORING SHOCK-DAY HEADLINES ONLY")
print("=" * 100)

shock_marker = shock_days[["symbol", "date"]].drop_duplicates()
shock_marker["is_shock_day"] = True
shock_articles = exploded.merge(shock_marker, on=["symbol", "date"], how="inner")
shock_articles = shock_articles.drop(columns=["is_shock_day"]).drop_duplicates(subset=["id"])
print(f"Unique headlines on shock days needing a sentiment score: {len(shock_articles)}")

cache_hits = pd.DataFrame(columns=["id", "sentiment_score"])
if os.path.exists(SCORED_CACHE_PATH):
    cached = pd.read_parquet(SCORED_CACHE_PATH, columns=["id", "sentiment_score"])
    cache_hits = shock_articles[["id"]].merge(cached, on="id", how="inner")
    print(f"Reused from Test 13's already-scored cache ({SCORED_CACHE_PATH}): {len(cache_hits)}")

need_scoring = shock_articles[~shock_articles["id"].isin(cache_hits["id"])].copy()
print(f"Headlines requiring a fresh FinBERT pass: {len(need_scoring)}")

if os.path.exists(SHOCK_SCORED_OUT_PATH):
    prior_scored = pd.read_parquet(SHOCK_SCORED_OUT_PATH)
    already_done = need_scoring["id"].isin(prior_scored["id"])
    if already_done.any():
        print(f"Found {already_done.sum()} already scored in a prior run of this script -- reusing.")
    need_scoring = need_scoring[~already_done]
else:
    prior_scored = pd.DataFrame(columns=["id", "sentiment_score"])

if len(need_scoring):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print("Loading ProsusAI/finbert (CPU)...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()
    id2label = model.config.id2label

    headlines = need_scoring["headline"].fillna("").tolist()
    n = len(headlines)
    pos = np.zeros(n)
    neg = np.zeros(n)
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n, SCORE_BATCH_SIZE):
            batch = headlines[start:start + SCORE_BATCH_SIZE]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=64)
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).numpy()
            for i, label in id2label.items():
                if label.lower() == "positive":
                    pos[start:start + len(batch)] = probs[:, int(i)]
                elif label.lower() == "negative":
                    neg[start:start + len(batch)] = probs[:, int(i)]
            done = min(start + SCORE_BATCH_SIZE, n)
            if done % 500 < SCORE_BATCH_SIZE or done == n:
                rate = done / (time.time() - t0) if time.time() > t0 else 0
                print(f"  [{done}/{n}] {rate:.1f} headlines/sec", flush=True)

    new_scores = need_scoring[["id"]].copy()
    new_scores["sentiment_score"] = pos - neg
    prior_scored = pd.concat([prior_scored, new_scores], ignore_index=True).drop_duplicates(subset=["id"])
    prior_scored.to_parquet(SHOCK_SCORED_OUT_PATH, index=False)
    print(f"Scored {n} new headlines, saved to {SHOCK_SCORED_OUT_PATH}")

all_shock_scores = pd.concat([cache_hits, prior_scored[prior_scored["id"].isin(shock_articles["id"])]],
                              ignore_index=True).drop_duplicates(subset=["id"])
print(f"Total shock-day headlines with a sentiment score: {len(all_shock_scores)} / {len(shock_articles)}")

shock_articles = shock_articles.merge(all_shock_scores, on="id", how="left")
daily_sentiment = (
    shock_articles.dropna(subset=["sentiment_score"])
    .groupby(["symbol", "date"])["sentiment_score"].mean()
    .rename("sentiment").reset_index()
)
shock_days = shock_days.merge(daily_sentiment, on=["symbol", "date"], how="left")
n_no_sentiment = shock_days["sentiment"].isna().sum()
if n_no_sentiment:
    print(f"WARNING: {n_no_sentiment} shock days have no scored headline (scoring failure) -- excluded below.")
shock_days = shock_days.dropna(subset=["sentiment"]).reset_index(drop=True)


# ============================================================
# PRICE LOOKUPS + PER-TICKER FORWARD RETURNS (verbatim pattern from event_study_v2.py)
# ============================================================
print()
print("=" * 100)
print("PRICE LOOKUPS + FORWARD RETURNS")
print("=" * 100)
ticker_groups = {t: g.sort_values("date").reset_index(drop=True)
                  for t, g in price_raw[price_raw["ticker"].isin(sample_tickers)].groupby("ticker")}

stock_fwd = {}          # ticker -> {h: fwd array over price-date positions}
stock_dates_idx = {}    # ticker -> price-date numpy array (for searchsorted)
stock_nonshock_mean = {}   # ticker -> {h: mean fwd ret on non-shock trading days} (Task 3 baseline)
stock_unconditional_mean = {}  # ticker -> {h: mean fwd ret on ALL trading days} (Task 4 baseline)

shock_dates_by_ticker = shock_days.groupby("symbol")["date"].apply(set).to_dict()

for ticker, pdf in ticker_groups.items():
    dates = pdf["date"].to_numpy()
    opens = pdf["open"].to_numpy(dtype=float)
    n = len(dates)
    stock_dates_idx[ticker] = dates

    fwd = {h: forward_return(opens, h) for h in HORIZONS}
    stock_fwd[ticker] = fwd

    shock_pos = set()
    for d in shock_dates_by_ticker.get(ticker, set()):
        idx = np.searchsorted(dates, np.datetime64(d), side="left")
        if idx < n:
            shock_pos.add(idx)
    all_pos = np.arange(n)
    nonshock_pos = np.array([p for p in all_pos if p not in shock_pos])

    stock_nonshock_mean[ticker] = {}
    stock_unconditional_mean[ticker] = {}
    for h in HORIZONS:
        vals_all = fwd[h][~np.isnan(fwd[h])]
        stock_unconditional_mean[ticker][h] = float(vals_all.mean()) if len(vals_all) else np.nan
        vals_ns = fwd[h][nonshock_pos]
        vals_ns = vals_ns[~np.isnan(vals_ns)]
        stock_nonshock_mean[ticker][h] = float(vals_ns.mean()) if len(vals_ns) else np.nan

print(f"Per-ticker forward-return baselines built for {len(ticker_groups)} tickers.")


# ============================================================
# BUILD SHOCK-EVENT TABLE: event forward returns + both baselines
# ============================================================
event_rows = []
for row in shock_days.itertuples(index=False):
    ticker = row.symbol
    dates = stock_dates_idx.get(ticker)
    if dates is None:
        continue
    idx = np.searchsorted(dates, np.datetime64(row.date), side="left")
    fwd = stock_fwd[ticker]
    rec = {"ticker": ticker, "date": row.date, "sentiment": row.sentiment, "n_articles": row.n_articles}
    for h in HORIZONS:
        event_ret = fwd[h][idx] if idx < len(dates) else np.nan
        ns_base = stock_nonshock_mean[ticker].get(h, np.nan)
        uc_base = stock_unconditional_mean[ticker].get(h, np.nan)
        rec[f"event_ret_{h}"] = event_ret
        rec[f"adj_ar_{h}"] = event_ret - ns_base if pd.notna(event_ret) and pd.notna(ns_base) else np.nan
        rec[f"uncond_diff_{h}"] = event_ret - uc_base if pd.notna(event_ret) and pd.notna(uc_base) else np.nan
    event_rows.append(rec)

events = pd.DataFrame(event_rows)
print(f"\nShock events with usable price data: {len(events)} / {len(shock_days)}")

# Sentiment terciles -- cut on the POOLED shock-day sentiment distribution, not per-ticker, not tuned.
q33, q67 = events["sentiment"].quantile([1 / 3, 2 / 3])
events["tercile"] = np.select(
    [events["sentiment"] <= q33, events["sentiment"] >= q67],
    ["negative", "positive"], default="neutral",
)
print(f"Tercile cutoffs (pooled shock-day sentiment): bottom <= {q33:.4f}, top >= {q67:.4f}")
print(events["tercile"].value_counts().to_string())


# ============================================================
# TASK 3: CONDITIONAL EVENT ANALYSIS
# ============================================================
def report_bucket(label, df, col_prefix):
    print(f"\n{label} -- {len(df)} events")
    header = f"{'Horizon':<10}{'n':>8}{'Mean':>11}{'t-stat':>9}   {'Note'}"
    print(header)
    print("-" * len(header))
    for h in HORIZONS:
        vals = df[f"{col_prefix}_{h}"].dropna().to_numpy()
        n, mean, t = t_stat(vals)
        note = "" if n >= MIN_BUCKET_TRUST else f"UNDERPOWERED (n<{MIN_BUCKET_TRUST})"
        if n < 2:
            print(f"{str(h) + 'd':<10}{n:>8}{'n/a':>11}{'n/a':>9}   {note}")
            continue
        print(f"{str(h) + 'd':<10}{n:>8}{mean * 100:>10.3f}%{t:>9.2f}   {note}")


print()
print("=" * 100)
print("TASK 3: CONDITIONAL EVENT ANALYSIS (adjusted AR vs each ticker's non-shock-day baseline)")
print("=" * 100)
report_bucket("Shock days -- NEGATIVE sentiment (bottom tercile)", events[events["tercile"] == "negative"], "adj_ar")
report_bucket("Shock days -- POSITIVE sentiment (top tercile)", events[events["tercile"] == "positive"], "adj_ar")
report_bucket("Shock days -- NEUTRAL sentiment (middle tercile)", events[events["tercile"] == "neutral"], "adj_ar")
report_bucket("ALL shock days pooled (regardless of sentiment)", events, "adj_ar")


# ============================================================
# TASK 4: RISK-FILTER TEST (negative-sentiment shocks vs unconditional baseline)
# ============================================================
print()
print("=" * 100)
print("TASK 4: RISK-FILTER TEST -- negative-sentiment shocks vs each ticker's UNCONDITIONAL mean")
print("=" * 100)
print("Question: if you already hold a stock and it gets a negative news shock, does its forward")
print("return differ from what that stock does on average, unconditionally? (5/10/20-day horizons only)")
neg_events = events[events["tercile"] == "negative"]
print(f"\nNegative-sentiment shock events: {len(neg_events)}")
header = f"{'Horizon':<10}{'n':>8}{'MeanDiff':>11}{'t-stat':>9}   {'vs 0.63% cost'}"
print(header)
print("-" * len(header))
for h in [5, 10, 20]:
    vals = neg_events[f"uncond_diff_{h}"].dropna().to_numpy()
    n, mean, t = t_stat(vals)
    if n < 2:
        print(f"{str(h) + 'd':<10}{n:>8}{'n/a':>11}{'n/a':>9}   n/a")
        continue
    underpowered = " UNDERPOWERED" if n < MIN_BUCKET_TRUST else ""
    significant = (not np.isnan(t)) and abs(t) > 2
    if not significant:
        cost_verdict = "not significant -- cost comparison moot"
    elif abs(mean) > ROUNDTRIP_COST:
        cost_verdict = f"EXCEEDS {ROUNDTRIP_COST*100:.2f}% cost -- potentially actionable"
    else:
        cost_verdict = f"significant but BELOW {ROUNDTRIP_COST*100:.2f}% cost -- not actionable at this account size"
    print(f"{str(h) + 'd':<10}{n:>8}{mean * 100:>10.3f}%{t:>9.2f}   {cost_verdict}{underpowered}")


# ============================================================
# TASK 5: SAMPLE-SIZE HONESTY SUMMARY
# ============================================================
print()
print("=" * 100)
print("TASK 5: SAMPLE-SIZE HONESTY")
print("=" * 100)
bucket_counts = {
    "Negative tercile (Task 3)": (events["tercile"] == "negative").sum(),
    "Positive tercile (Task 3)": (events["tercile"] == "positive").sum(),
    "Neutral tercile (Task 3)": (events["tercile"] == "neutral").sum(),
    "All shocks pooled (Task 3)": len(events),
    "Negative tercile (Task 4)": len(neg_events),
}
for label, n in bucket_counts.items():
    flag = "OK" if n >= MIN_BUCKET_TRUST else f"UNDERPOWERED (< {MIN_BUCKET_TRUST})"
    print(f"  {label:<32} n={n:<6} {flag}")

print()
print("=" * 100)
print("This is a diagnostic scan only: the shock threshold (3x median), tercile cuts, and cost")
print("floor were fixed before results were seen. No strategy was built, no threshold was tuned,")
print("and no bucket was selected as \"best\" based on its outcome.")
