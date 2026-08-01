"""
sentiment_shock_robustness.py -- five adversarial checks on Test 14's
1-day negative-sentiment shock effect (adjusted AR -1.685%, t=-3.94).
Test 14's 20-day effect showed up in ALL terciles (including positive),
so that one is attention-driven, not sentiment-driven, and not pursued
further here. The 1-day effect is the one place sentiment direction
showed up cleanly -- this is what could actually survive to become a
risk-filter layer, if it survives being attacked. Only the 1-day
horizon is examined below, per instruction -- no other horizon, no
retuned shock threshold, no re-cut terciles (the ORIGINAL Test 14
tercile cutoffs, computed once from the full 1,782-event pooled
distribution, are reused unchanged throughout).

Reconstructs Test 14's shock-day event table from cached data (no
refetching, no rescoring -- data/alpaca_news_wide.parquet,
data/alpaca_news_scored.parquet, and data/alpaca_news_shock_scored.parquet
already contain everything needed).

Check 1 -- Independence: shock days cluster (a company doesn't get one
isolated headline, it gets a burst). Greedy declustering per ticker --
keep the first shock day, discard anything within 10 trading days of
the last KEPT shock day, then the next survivor starts a new window.
Tests whether the effect survives once dependent, clustered
observations stop being counted as if they were independent draws.

Check 2 -- Liquidity: median-split by trailing 60-day dollar volume
(the same SIZE_LOOKBACK_DAYS=60 proxy event_study_v2.py used). An
effect that only exists in illiquid names isn't tradeable at any
real size.

Check 3 -- Meme contamination: AMC alone contributed 4,191 of the
20,379 wide-corpus headlines. Exclude the 5 most-covered tickers
entirely (by total article count) and see if the effect survives on
the other 95.

Check 4 -- Volatility control: shock days are, almost by definition,
unusually volatile days. Test 14's baseline (mean forward return on
ALL non-shock days) doesn't account for that. This builds a stricter
baseline: for each ticker, days are bucketed into that ticker's OWN
trailing-20-day-realized-volatility terciles (computed over its full
history, not tuned), and each shock day's baseline is that ticker's
mean forward return on ITS non-shock days in the SAME volatility
tercile -- comparing like-volatility days to like-volatility days.

Check 5 -- Implementation: translates whatever survives Checks 1-4
into an annual portfolio-level estimate for a 20-stock equal-weighted
book (the same size as the live value portfolio), using the
DECLUSTERED (Check 1) event rate and effect size as the base case,
since declustering is what fixes the "how often does this actually,
independently fire" question -- the raw 1,782-count would overstate
firing frequency the same way it overstated significance.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import pandas as pd

UNIVERSE_PATH = os.path.join("data", "yf_universe.parquet")
NEWS_WIDE_PATH = os.path.join("data", "alpaca_news_wide.parquet")
SCORED_CACHE_PATH = os.path.join("data", "alpaca_news_scored.parquet")
SHOCK_SCORED_PATH = os.path.join("data", "alpaca_news_shock_scored.parquet")

SAMPLE_SIZE = 100
SEED = 42
TICKER_ALIASES = {"GOOG": "GOOGL"}
SHOCK_MULTIPLIER = 3.0
MIN_ACTIVE_DAYS_FOR_MEDIAN = 2
HORIZON = 1  # exclusively -- per instruction, no other horizon examined

DECLUSTER_WINDOW = 10          # Check 1: trading days
SIZE_LOOKBACK_DAYS = 60        # Check 2: matches event_study_v2.py
MIN_SIZE_LOOKBACK = 20
N_MEME_EXCLUDED = 5             # Check 3
VOL_LOOKBACK_DAYS = 20          # Check 4: trailing realized-vol window
MIN_VOL_MATCH_OBS = 10          # Check 4: minimum same-tercile non-shock days to trust a baseline

PORTFOLIO_SIZE = 20              # Check 5: matches the live value portfolio
ROUNDTRIP_COST = 0.0063          # Check 5: exit + re-entry, one round trip (RESEARCH_LOG.md Test 9)

pd.set_option("display.width", 140)


def forward_return_1(opens):
    """1-day-ahead version of ic_analysis.py's forward_return(): fwd[N] = open[N+2]/open[N+1] - 1."""
    n = len(opens)
    fwd = np.full(n, np.nan)
    last_n = n - 2
    if last_n <= 0:
        return fwd
    idx = np.arange(0, last_n)
    fwd[idx] = opens[idx + 2] / opens[idx + 1] - 1
    return fwd


def t_stat(vals):
    n = len(vals)
    if n < 2:
        return n, float("nan"), float("nan")
    mean = float(vals.mean())
    std = float(vals.std(ddof=1))
    t = mean / (std / np.sqrt(n)) if std > 0 else float("nan")
    return n, mean, t


def report_line(label, vals):
    n, mean, t = t_stat(vals)
    if n < 2:
        print(f"  {label:<45} n=  0   n/a")
        return n, mean, t
    print(f"  {label:<45} n={n:<5} mean={mean*100:>8.3f}%   t={t:>6.2f}")
    return n, mean, t


# ============================================================
# RECONSTRUCT TEST 14'S SHOCK-DAY EVENT TABLE (from cache, no refetch/rescore)
# ============================================================
print("=" * 100)
print("RECONSTRUCTION (from cached data -- no new fetching or scoring)")
print("=" * 100)
universe_tickers = sorted(pd.read_parquet(UNIVERSE_PATH, columns=["ticker"])["ticker"].unique())
sample = random.Random(SEED).sample(universe_tickers, SAMPLE_SIZE)

news_wide = pd.read_parquet(NEWS_WIDE_PATH)
price_raw = pd.read_parquet(UNIVERSE_PATH)
price_raw["date"] = pd.to_datetime(price_raw["date"])

exploded = news_wide.explode("symbols").rename(columns={"symbols": "symbol"})
exploded["symbol"] = exploded["symbol"].replace(TICKER_ALIASES)
sample_tickers = sorted(set(sample) & set(price_raw["ticker"].unique()))
exploded = exploded[exploded["symbol"].isin(sample_tickers)].copy()
exploded["date"] = pd.to_datetime(exploded["created_at"]).dt.tz_localize(None).dt.normalize().astype("datetime64[ns]")

# Check 3 needs total article counts per ticker across the FULL wide corpus (not just shock days)
total_article_counts = exploded.groupby("symbol").size().sort_values(ascending=False)
meme_tickers = total_article_counts.head(N_MEME_EXCLUDED).index.tolist()
print(f"Top {N_MEME_EXCLUDED} most-covered tickers (for Check 3): "
      f"{[(t, int(total_article_counts[t])) for t in meme_tickers]}")

daily_counts = exploded.groupby(["symbol", "date"]).size().rename("n_articles").reset_index()
shock_rows = []
for ticker, g in daily_counts.groupby("symbol"):
    if len(g) < MIN_ACTIVE_DAYS_FOR_MEDIAN:
        continue
    median_count = g["n_articles"].median()
    if median_count <= 0:
        continue
    shocks = g[g["n_articles"] >= SHOCK_MULTIPLIER * median_count].copy()
    shock_rows.append(shocks)
shock_days = pd.concat(shock_rows, ignore_index=True)
print(f"Shock days reconstructed: {len(shock_days)} (Test 14 found 1,782 -- must match)")
assert len(shock_days) == 1782, "Reconstruction mismatch vs Test 14 -- aborting rather than silently drifting."

# Sentiment: merge cached scores (Test 13's corpus + Test 14's shock-only scoring)
scored_cache = pd.read_parquet(SCORED_CACHE_PATH, columns=["id", "sentiment_score"])
shock_scored = pd.read_parquet(SHOCK_SCORED_PATH)
all_scores = pd.concat([scored_cache, shock_scored], ignore_index=True).drop_duplicates(subset=["id"])

shock_marker = shock_days[["symbol", "date"]].drop_duplicates()
shock_marker["is_shock_day"] = True
shock_articles = exploded.merge(shock_marker, on=["symbol", "date"], how="inner")
shock_articles = shock_articles.drop(columns=["is_shock_day"]).drop_duplicates(subset=["id"])
shock_articles = shock_articles.merge(all_scores, on="id", how="left")

daily_sentiment = (
    shock_articles.dropna(subset=["sentiment_score"])
    .groupby(["symbol", "date"])["sentiment_score"].mean()
    .rename("sentiment").reset_index()
)
shock_days = shock_days.merge(daily_sentiment, on=["symbol", "date"], how="left").dropna(subset=["sentiment"])
print(f"Shock days with a sentiment score: {len(shock_days)}")

# Price lookups + forward returns + non-shock baseline (Test 3's method, horizon=1 only)
ticker_groups = {t: g.sort_values("date").reset_index(drop=True)
                  for t, g in price_raw[price_raw["ticker"].isin(sample_tickers)].groupby("ticker")}

stock_dates = {}
stock_fwd1 = {}
stock_opens = {}
stock_closes = {}
stock_nonshock_mean1 = {}
shock_dates_by_ticker = shock_days.groupby("symbol")["date"].apply(set).to_dict()

for ticker, pdf in ticker_groups.items():
    dates = pdf["date"].to_numpy()
    opens = pdf["open"].to_numpy(dtype=float)
    closes = pdf["close"].to_numpy(dtype=float)
    volumes = pdf["volume"].to_numpy(dtype=float)
    n = len(dates)
    stock_dates[ticker] = dates
    stock_opens[ticker] = opens
    stock_closes[ticker] = closes
    fwd1 = forward_return_1(opens)
    stock_fwd1[ticker] = fwd1

    shock_pos = set()
    for d in shock_dates_by_ticker.get(ticker, set()):
        idx = np.searchsorted(dates, np.datetime64(d), side="left")
        if idx < n:
            shock_pos.add(idx)
    nonshock_mask = np.ones(n, dtype=bool)
    for p in shock_pos:
        nonshock_mask[p] = False
    vals = fwd1[nonshock_mask]
    vals = vals[~np.isnan(vals)]
    stock_nonshock_mean1[ticker] = float(vals.mean()) if len(vals) else np.nan

# Build the event table: one row per shock day with position index, event return, adjusted AR
event_rows = []
for row in shock_days.itertuples(index=False):
    ticker = row.symbol
    dates = stock_dates.get(ticker)
    if dates is None:
        continue
    idx = int(np.searchsorted(dates, np.datetime64(row.date), side="left"))
    if idx >= len(dates):
        continue
    event_ret = stock_fwd1[ticker][idx]
    baseline = stock_nonshock_mean1[ticker]
    adj_ar = event_ret - baseline if pd.notna(event_ret) and pd.notna(baseline) else np.nan
    event_rows.append({"ticker": ticker, "date": row.date, "pos": idx, "sentiment": row.sentiment,
                        "n_articles": row.n_articles, "event_ret_1": event_ret, "adj_ar_1": adj_ar})
events = pd.DataFrame(event_rows)

# ORIGINAL, FIXED tercile cutoffs -- computed once here (must match Test 14's -0.0429 / 0.2128)
q33, q67 = events["sentiment"].quantile([1 / 3, 2 / 3])
events["tercile"] = np.select([events["sentiment"] <= q33, events["sentiment"] >= q67],
                               ["negative", "positive"], default="neutral")
print(f"Tercile cutoffs: bottom <= {q33:.4f}, top >= {q67:.4f} (Test 14: -0.0429 / 0.2128 -- must match)")

baseline_neg = events[events["tercile"] == "negative"]
print(f"\nBaseline reproduction check -- negative tercile, 1-day, unrestricted:")
report_line("Original (Test 14) baseline", baseline_neg["adj_ar_1"].dropna().to_numpy())
print("(Should reproduce n=594, mean=-1.685%, t=-3.94 from Test 14 before any of the 5 checks below.)")


# ============================================================
# CHECK 1: INDEPENDENCE (declustering)
# ============================================================
print()
print("=" * 100)
print("CHECK 1: INDEPENDENCE -- greedy decluster, keep first shock/10 trading days per ticker")
print("=" * 100)
kept_idx = []
for ticker, g in events.sort_values("pos").groupby("ticker"):
    last_kept = -10**9
    for i, pos in zip(g.index, g["pos"]):
        if pos - last_kept >= DECLUSTER_WINDOW:
            kept_idx.append(i)
            last_kept = pos
declustered = events.loc[kept_idx].copy()
print(f"Events before declustering: {len(events)}  |  after: {len(declustered)} "
      f"({len(declustered) / len(events) * 100:.1f}% survive)")

decl_neg = declustered[declustered["tercile"] == "negative"]
print(f"Negative tercile: {len(baseline_neg)} -> {len(decl_neg)} after declustering")
check1_n, check1_mean, check1_t = report_line("Declustered negative tercile, 1-day",
                                                decl_neg["adj_ar_1"].dropna().to_numpy())
check1_survives = (not np.isnan(check1_t)) and abs(check1_t) > 2
print(f"Survives (|t|>2): {check1_survives}")


# ============================================================
# CHECK 2: LIQUIDITY (median dollar-volume split)
# ============================================================
print()
print("=" * 100)
print("CHECK 2: LIQUIDITY -- median split by trailing 60-day $ volume")
print("=" * 100)
size_proxies = []
for row in events.itertuples(index=False):
    ticker = row.ticker
    idx = row.pos
    closes = stock_closes[ticker]
    volumes = ticker_groups[ticker]["volume"].to_numpy(dtype=float)
    lo = max(0, idx - SIZE_LOOKBACK_DAYS)
    window_days = idx - lo
    proxy = float(np.median(closes[lo:idx] * volumes[lo:idx])) if window_days >= MIN_SIZE_LOOKBACK else np.nan
    size_proxies.append(proxy)
events["size_proxy"] = size_proxies

valid_size = events.dropna(subset=["size_proxy"])
median_proxy = valid_size["size_proxy"].median()
print(f"Events with a valid size proxy: {len(valid_size)} / {len(events)} "
      f"(median 60-day $ volume: ${median_proxy:,.0f})")

neg_valid = valid_size[valid_size["tercile"] == "negative"]
low_liq = neg_valid[neg_valid["size_proxy"] < median_proxy]
high_liq = neg_valid[neg_valid["size_proxy"] >= median_proxy]
check2_low_n, check2_low_mean, check2_low_t = report_line("Illiquid half (negative tercile), 1-day",
                                                             low_liq["adj_ar_1"].dropna().to_numpy())
check2_high_n, check2_high_mean, check2_high_t = report_line("Liquid half (negative tercile), 1-day",
                                                                high_liq["adj_ar_1"].dropna().to_numpy())
liquid_survives = (not np.isnan(check2_high_t)) and abs(check2_high_t) > 2
print(f"Survives in the LIQUID half (|t|>2): {liquid_survives} "
      f"{'-- tradeable at real size' if liquid_survives else '-- only in illiquid names, not tradeable at real size' if (not np.isnan(check2_low_t) and abs(check2_low_t) > 2) else ''}")


# ============================================================
# CHECK 3: MEME CONTAMINATION (exclude top-5 most-covered tickers)
# ============================================================
print()
print("=" * 100)
print(f"CHECK 3: MEME CONTAMINATION -- exclude the {N_MEME_EXCLUDED} most-covered tickers {meme_tickers}")
print("=" * 100)
ex_neg = baseline_neg[~baseline_neg["ticker"].isin(meme_tickers)]
print(f"Negative tercile: {len(baseline_neg)} -> {len(ex_neg)} after excluding meme-heavy tickers "
      f"({(baseline_neg['ticker'].isin(meme_tickers)).sum()} events removed)")
check3_n, check3_mean, check3_t = report_line("Ex-meme negative tercile, 1-day",
                                                 ex_neg["adj_ar_1"].dropna().to_numpy())
check3_survives = (not np.isnan(check3_t)) and abs(check3_t) > 2
print(f"Survives (|t|>2): {check3_survives}")


# ============================================================
# CHECK 4: VOLATILITY-MATCHED CONTROL
# ============================================================
print()
print("=" * 100)
print("CHECK 4: VOLATILITY CONTROL -- baseline = same ticker's non-shock days in the SAME "
      "trailing-20d realized-vol tercile (not ALL non-shock days)")
print("=" * 100)
vol_adj_ars = []
n_excluded_thin_vol_baseline = 0
for ticker, g in events.groupby("ticker"):
    closes = stock_closes[ticker]
    n = len(closes)
    daily_ret = np.full(n, np.nan)
    daily_ret[1:] = closes[1:] / closes[:-1] - 1
    roll_vol = pd.Series(daily_ret).rolling(VOL_LOOKBACK_DAYS).std().to_numpy()

    valid_vol_mask = ~np.isnan(roll_vol)
    if valid_vol_mask.sum() < 30:  # need enough days to even form vol terciles for this ticker
        continue
    # labels=False (integer bin codes) avoids a label-count/bin-count mismatch if duplicates="drop"
    # collapses fewer than 3 bins for a ticker with many identical rolling-vol values.
    vol_terc = pd.qcut(pd.Series(roll_vol[valid_vol_mask]), 3, labels=False, duplicates="drop")
    vol_tercile_by_pos = pd.Series(index=np.where(valid_vol_mask)[0], data=vol_terc.to_numpy())

    shock_positions_this_ticker = set(g["pos"])
    for row in g.itertuples(index=False):
        pos = row.pos
        if pos not in vol_tercile_by_pos.index:
            n_excluded_thin_vol_baseline += 1
            continue
        this_terc = vol_tercile_by_pos.loc[pos]
        same_terc_positions = vol_tercile_by_pos[vol_tercile_by_pos == this_terc].index
        baseline_positions = [p for p in same_terc_positions if p not in shock_positions_this_ticker]
        fwd = stock_fwd1[ticker]
        baseline_vals = fwd[baseline_positions]
        baseline_vals = baseline_vals[~np.isnan(baseline_vals)]
        if len(baseline_vals) < MIN_VOL_MATCH_OBS:
            n_excluded_thin_vol_baseline += 1
            continue
        vol_baseline_mean = float(baseline_vals.mean())
        event_ret = fwd[pos] if pos < len(fwd) else np.nan
        if pd.isna(event_ret):
            continue
        vol_adj_ars.append({"ticker": ticker, "pos": pos, "tercile": row.tercile,
                             "vol_adj_ar_1": event_ret - vol_baseline_mean, "vol_tercile": this_terc,
                             "n_baseline_days": len(baseline_vals)})

vol_events = pd.DataFrame(vol_adj_ars)
print(f"Events excluded (insufficient same-vol-tercile non-shock baseline, <{MIN_VOL_MATCH_OBS} days): "
      f"{n_excluded_thin_vol_baseline}")
vol_neg = vol_events[vol_events["tercile"] == "negative"]
check4_n, check4_mean, check4_t = report_line("Vol-matched negative tercile, 1-day",
                                                 vol_neg["vol_adj_ar_1"].dropna().to_numpy())
check4_survives = (not np.isnan(check4_t)) and abs(check4_t) > 2
print(f"Survives against a volatility-matched baseline (|t|>2): {check4_survives}")
print("(If this is much weaker than the unrestricted -1.685%/t=-3.94, part of Test 14's 1-day effect "
      "was already explainable by shock days simply being higher-volatility days, independent of sentiment.)")


# ============================================================
# CHECK 5: IMPLEMENTATION SIMULATION
# ============================================================
print()
print("=" * 100)
print("CHECK 5: IMPLEMENTATION -- 20-stock equal-weighted portfolio, using the DECLUSTERED "
      "(Check 1) rate and effect size as the base case")
print("=" * 100)
span_start = pd.Timestamp(news_wide["created_at"].min()[:10])
span_end = pd.Timestamp(news_wide["created_at"].max()[:10])
years_covered = (span_end - span_start).days / 365.25
n_tickers_with_shocks = shock_days["symbol"].nunique()
decl_neg_rate_per_ticker_per_year = len(decl_neg) / SAMPLE_SIZE / years_covered
expected_shocks_per_year_20stock = decl_neg_rate_per_ticker_per_year * PORTFOLIO_SIZE

print(f"Sample window: {years_covered:.2f} years, {SAMPLE_SIZE} tickers")
print(f"Declustered negative-tercile shock rate: {len(decl_neg)} events / {SAMPLE_SIZE} tickers / "
      f"{years_covered:.2f} years = {decl_neg_rate_per_ticker_per_year:.3f} per ticker per year")
print(f"Expected negative shocks per year, {PORTFOLIO_SIZE}-stock equal-weighted portfolio: "
      f"{expected_shocks_per_year_20stock:.2f}")

avoided_loss_per_event = abs(check1_mean) if not np.isnan(check1_mean) else float("nan")
net_benefit_per_event = avoided_loss_per_event - ROUNDTRIP_COST
net_annual_pp = expected_shocks_per_year_20stock * net_benefit_per_event / PORTFOLIO_SIZE * 100

print(f"\nDeclustered effect size (avoided loss per triggered event, if the position is exited): "
      f"{avoided_loss_per_event*100:.3f}%")
print(f"Round-trip cost of exiting + re-entering: {ROUNDTRIP_COST*100:.2f}%")
print(f"Net benefit per triggered event: {net_benefit_per_event*100:+.3f}%")
print(f"Portfolio-level annual benefit (net_benefit_per_event x expected_events / {PORTFOLIO_SIZE} positions): "
      f"{net_annual_pp:+.4f} percentage points/year")

if not check1_survives:
    verdict = "NOT WORTH IMPLEMENTING -- the effect did not survive declustering (Check 1) in the first place."
elif net_benefit_per_event <= 0:
    verdict = "NOT WORTH IMPLEMENTING -- avoided loss does not exceed the round-trip cost."
elif net_annual_pp < 0.1:
    verdict = ("MARGINAL / PROBABLY NOT WORTH IMPLEMENTING -- net benefit is positive but a fraction of a "
               "percentage point per year on the whole portfolio, for meaningfully added operational "
               "complexity (a live news-monitoring + FinBERT-scoring + same-day-exit pipeline).")
else:
    verdict = "Positive expected net benefit at the portfolio level -- see caveats above before implementing."
print(f"\nVERDICT: {verdict}")


# ============================================================
# SUMMARY
# ============================================================
print()
print("=" * 100)
print("SUMMARY -- does the 1-day negative-sentiment shock effect survive?")
print("=" * 100)
print(f"{'Original (Test 14, unrestricted)':<45} n={len(baseline_neg):<5} "
      f"mean={baseline_neg['adj_ar_1'].mean()*100:>8.3f}%   t={t_stat(baseline_neg['adj_ar_1'].dropna().to_numpy())[2]:>6.2f}")
print(f"{'Check 1: declustered':<45} n={check1_n:<5} mean={check1_mean*100:>8.3f}%   t={check1_t:>6.2f}   "
      f"{'SURVIVES' if check1_survives else 'FAILS'}")
print(f"{'Check 2: liquid half only':<45} n={check2_high_n:<5} mean={check2_high_mean*100:>8.3f}%   "
      f"t={check2_high_t:>6.2f}   {'SURVIVES' if liquid_survives else 'FAILS'}")
print(f"{'Check 3: ex-meme (top-5 excluded)':<45} n={check3_n:<5} mean={check3_mean*100:>8.3f}%   "
      f"t={check3_t:>6.2f}   {'SURVIVES' if check3_survives else 'FAILS'}")
print(f"{'Check 4: volatility-matched baseline':<45} n={check4_n:<5} mean={check4_mean*100:>8.3f}%   "
      f"t={check4_t:>6.2f}   {'SURVIVES' if check4_survives else 'FAILS'}")
n_survived = sum([check1_survives, liquid_survives, check3_survives, check4_survives])
print(f"\n{n_survived}/4 adversarial checks survived (|t|>2 maintained).")
print("No shock threshold, tercile cut, or additional horizon was altered to produce any of the above.")
