"""
sentiment_test.py -- diagnostic only: does news sentiment predict
anything at all? No strategy, no portfolio, no trading -- this is the
same "is there ANY information here" question ic_analysis.py asked of
price-based signals, asked here of FinBERT-scored Alpaca news headlines
instead. forward_return(), spearman_corr(), and the per-stock-IC ->
cross-sectional-t-test methodology are reused verbatim from
ic_analysis.py; a pooled (all stock-days combined) correlation and
t-stat is added alongside, since the brief asked for both.

Universe: the 11 stocks across both retests -- 6 US (fugazzi_retest.py:
NVDA, AVGO, LLY, XOM, WMT, GOOGL) + 5 Swedish ADRs (swedish_retest.py:
SPOT, ERIC, AZN, ALV, OTLY). Window: 2021-01-01 to 2025-12-31, matching
the combined training+test span of both retests, rather than pulling
Alpaca's full ~10-year archive (confirmed available back to at least
2015 by check_alpaca_news_api.py) -- five years x eleven tickers is
already a five/six-figure article count once fetched; going back
further would multiply fetch + FinBERT-scoring time for headlines
outside any window this project has actually price-tested.

Three checkpointed stages, each resumable independently by re-running
(existing parquet files are loaded instead of refetched/rescored):
  1. FETCH  -> data/alpaca_news_raw.parquet       (id, headline, created_at, symbols)
  2. SCORE  -> data/alpaca_news_scored.parquet     (+ finbert_pos/neg/neu, sentiment_score)
  3. IC analysis (in-memory, from the scored parquet + fresh yfinance prices)

Sentiment score per article = P(positive) - P(negative) from FinBERT
(ProsusAI/finbert), scored on the headline only (not summary/body), as
specified. An article tagged with multiple of our 11 tickers contributes
its score to EACH of them. Daily sentiment per ticker = mean article
score for that ticker on that UTC calendar date; dates are then snapped
forward to the next trading day (merge_asof, direction="forward") so
weekend/holiday news becomes "known" as of the next session -- the same
"known at day N's close" framing ic_analysis.py uses for its signals.

GOOGL note: Alpaca/other sources sometimes tag Alphabet articles with
"GOOG" (the other share class) instead of "GOOGL" -- both are treated
as the same underlying company and pooled under GOOGL.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

TICKERS = ["NVDA", "AVGO", "LLY", "XOM", "WMT", "GOOGL", "SPOT", "ERIC", "AZN", "ALV", "OTLY"]
TICKER_ALIASES = {"GOOG": "GOOGL"}  # other share class -> canonical ticker used everywhere below

START = "2021-01-01"
END = "2025-12-31"
HORIZONS = [1, 5, 10, 20]
MIN_STOCK_OBS = 20  # same defensive floor as ic_analysis.py

NEWS_RAW_PATH = os.path.join("data", "alpaca_news_raw.parquet")
NEWS_SCORED_PATH = os.path.join("data", "alpaca_news_scored.parquet")
FETCH_CHECKPOINT_EVERY_PAGES = 100
SCORE_CHECKPOINT_EVERY = 2000
SCORE_BATCH_SIZE = 32

NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
API_KEY = os.environ.get("APCA_API_KEY_ID")
API_SECRET = os.environ.get("APCA_API_SECRET_KEY")
if not API_KEY or not API_SECRET:
    raise SystemExit("APCA_API_KEY_ID / APCA_API_SECRET_KEY not found in .env.")
HEADERS = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": API_SECRET}


# ============================================================
# STAGE 1: FETCH (checkpointed, resumable)
# ============================================================
def fetch_all_news():
    if os.path.exists(NEWS_RAW_PATH):
        df = pd.read_parquet(NEWS_RAW_PATH)
        print(f"Found existing {NEWS_RAW_PATH} ({len(df)} articles) -- loading instead of refetching.")
        print("Delete this file first if you want a full refetch.")
        return df

    symbols_param = ",".join(TICKERS)
    all_rows = []
    seen_ids = set()
    page_token = None
    page_count = 0

    print(f"Fetching news for {TICKERS} from {START} to {END} (one combined multi-symbol stream)...")
    while True:
        params = {"start": f"{START}T00:00:00Z", "end": f"{END}T23:59:59Z",
                  "symbols": symbols_param, "limit": 50, "sort": "asc"}
        if page_token:
            params["page_token"] = page_token

        resp = requests.get(NEWS_URL, headers=HEADERS, params=params, timeout=30)
        if resp.status_code != 200:
            print(f"HTTP {resp.status_code} on page {page_count}: {resp.text[:300]}")
            time.sleep(2)
            continue

        data = resp.json()
        for a in data.get("news", []):
            if a["id"] in seen_ids:
                continue
            seen_ids.add(a["id"])
            all_rows.append({
                "id": a["id"],
                "headline": a["headline"],
                "created_at": a["created_at"],
                "symbols": list(a.get("symbols", [])),
            })

        page_token = data.get("next_page_token")
        page_count += 1
        if page_count % FETCH_CHECKPOINT_EVERY_PAGES == 0:
            pd.DataFrame(all_rows).to_parquet(NEWS_RAW_PATH, index=False)
            print(f"  [page {page_count}] checkpoint written -- {len(all_rows)} unique articles so far "
                  f"(latest: {all_rows[-1]['created_at']})", flush=True)

        if not page_token:
            break
        time.sleep(0.05)  # ~200/min headroom, comfortably under the observed 200/min limit

    df = pd.DataFrame(all_rows)
    df.to_parquet(NEWS_RAW_PATH, index=False)
    print(f"Fetch complete: {len(df)} unique articles across {page_count} pages, saved to {NEWS_RAW_PATH}")
    return df


# ============================================================
# STAGE 2: SCORE (FinBERT, batched, checkpointed)
# ============================================================
def score_all_news(news_df):
    if os.path.exists(NEWS_SCORED_PATH):
        scored = pd.read_parquet(NEWS_SCORED_PATH)
        if len(scored) == len(news_df):
            print(f"Found existing {NEWS_SCORED_PATH} matching the fetched article count -- loading.")
            return scored
        print(f"Existing {NEWS_SCORED_PATH} has {len(scored)} rows but {len(news_df)} articles are fetched -- "
              f"rescoring from scratch.")

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print("Loading ProsusAI/finbert (CPU)...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()
    # FinBERT label order (config.id2label): 0=positive, 1=negative, 2=neutral
    id2label = model.config.id2label
    print(f"Label mapping: {id2label}")

    headlines = news_df["headline"].fillna("").tolist()
    n = len(headlines)
    pos = np.zeros(n)
    neg = np.zeros(n)
    neu = np.zeros(n)

    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n, SCORE_BATCH_SIZE):
            batch = headlines[start:start + SCORE_BATCH_SIZE]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=64)
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).numpy()
            for i, label in id2label.items():
                col = {"positive": pos, "negative": neg, "neutral": neu}[label.lower()]
                col[start:start + len(batch)] = probs[:, int(i)]

            done = min(start + SCORE_BATCH_SIZE, n)
            if done % SCORE_CHECKPOINT_EVERY < SCORE_BATCH_SIZE:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta_min = (n - done) / rate / 60 if rate > 0 else float("nan")
                out = news_df.iloc[:done].copy()
                out["finbert_pos"] = pos[:done]
                out["finbert_neg"] = neg[:done]
                out["finbert_neu"] = neu[:done]
                out["sentiment_score"] = out["finbert_pos"] - out["finbert_neg"]
                out.to_parquet(NEWS_SCORED_PATH, index=False)
                print(f"  [{done}/{n}] checkpoint written -- {rate:.1f} headlines/sec, "
                      f"ETA {eta_min:.1f} min", flush=True)

    out = news_df.copy()
    out["finbert_pos"] = pos
    out["finbert_neg"] = neg
    out["finbert_neu"] = neu
    out["sentiment_score"] = out["finbert_pos"] - out["finbert_neg"]
    out.to_parquet(NEWS_SCORED_PATH, index=False)
    print(f"Scoring complete: {n} headlines, saved to {NEWS_SCORED_PATH}")
    return out


# ============================================================
# STAGE 3: AGGREGATE + PRICE DATA + IC ANALYSIS
# ============================================================
def spearman_corr(x, y):
    """Verbatim from ic_analysis.py."""
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def forward_return(opens, horizon):
    """Verbatim from ic_analysis.py: fwd[N] = open[N+1+horizon] / open[N+1] - 1 --
    signal known at day N's close predicts the return from day N+1's open to
    day N+1+horizon's open."""
    n = len(opens)
    fwd = np.full(n, np.nan)
    last_n = n - 1 - horizon
    if last_n <= 0:
        return fwd
    idx = np.arange(0, last_n)
    fwd[idx] = opens[idx + 1 + horizon] / opens[idx + 1] - 1
    return fwd


def pooled_t_stat(rho, n):
    if np.isnan(rho) or n <= 2 or abs(rho) >= 1:
        return float("nan")
    return float(rho * np.sqrt((n - 2) / (1 - rho ** 2)))


print("=" * 100)
print("STAGE 1: FETCH")
print("=" * 100)
news_df = fetch_all_news()

print()
print("=" * 100)
print("STAGE 2: SCORE (FinBERT)")
print("=" * 100)
scored_df = score_all_news(news_df)

print()
print("=" * 100)
print("STAGE 3: AGGREGATE TO DAILY PER-TICKER SENTIMENT")
print("=" * 100)
exploded = scored_df.explode("symbols").rename(columns={"symbols": "symbol"})
exploded["symbol"] = exploded["symbol"].replace(TICKER_ALIASES)
exploded = exploded[exploded["symbol"].isin(TICKERS)].copy()
exploded["date"] = pd.to_datetime(exploded["created_at"]).dt.tz_localize(None).dt.normalize().astype("datetime64[ns]")

daily_sentiment = (
    exploded.groupby(["symbol", "date"])["sentiment_score"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "sentiment", "count": "n_articles"})
    .reset_index()
)
print(f"Scored articles: {len(scored_df)}  |  (ticker, date) sentiment observations: {len(daily_sentiment)}")
print(daily_sentiment.groupby("symbol")["n_articles"].sum().sort_values(ascending=False)
      .rename("total_article_mentions").to_string())

print()
print("=" * 100)
print("STAGE 4: PRICE DATA + FORWARD RETURNS")
print("=" * 100)
px_raw = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)
opens_all = px_raw["Open"]
missing_px = [t for t in TICKERS if t not in opens_all.columns or opens_all[t].dropna().empty]
if missing_px:
    print(f"WARNING: no price data for {missing_px} -- excluded from the IC analysis.")

per_stock_ic = {h: {} for h in HORIZONS}
pooled_records = {h: [] for h in HORIZONS}

for ticker in TICKERS:
    if ticker in missing_px:
        continue
    opens = opens_all[ticker].dropna()
    price_dates = pd.to_datetime(opens.index).tz_localize(None).astype("datetime64[ns]")
    price_df = pd.DataFrame({"date": price_dates, "open": opens.to_numpy()}).sort_values("date")

    sent = daily_sentiment[daily_sentiment["symbol"] == ticker][["date", "sentiment"]].sort_values("date")
    if sent.empty:
        print(f"{ticker}: no scored news in window -- skipped")
        continue

    # Snap each article date forward to the next available trading day --
    # weekend/holiday news becomes "known" as of the next session's open.
    snapped = pd.merge_asof(sent, price_df[["date"]], on="date", direction="forward")
    snapped = snapped.dropna(subset=["date"])
    merged = price_df.merge(snapped.rename(columns={"date": "trading_date"}),
                             left_on="date", right_on="trading_date", how="left")
    # multiple news-days can snap to the same trading day (e.g. Sat+Sun -> Monday) -- average them
    daily_signal = merged.groupby("date").agg(open=("open", "first"), sentiment=("sentiment", "mean"))
    daily_signal = daily_signal.reset_index().sort_values("date").reset_index(drop=True)

    opens_arr = daily_signal["open"].to_numpy(dtype=float)
    sig_arr = daily_signal["sentiment"].to_numpy(dtype=float)

    for h in HORIZONS:
        fwd = forward_return(opens_arr, h)
        mask = ~np.isnan(sig_arr) & ~np.isnan(fwd)
        n_valid = int(mask.sum())
        if n_valid < MIN_STOCK_OBS:
            continue
        rho = spearman_corr(sig_arr[mask], fwd[mask])
        if not np.isnan(rho):
            per_stock_ic[h][ticker] = rho
            pooled_records[h].append(pd.DataFrame({"sentiment": sig_arr[mask], "fwd": fwd[mask]}))

print()
print("=" * 100)
print("IC ANALYSIS -- news sentiment vs forward returns (diagnostic only, no strategy)")
print("=" * 100)

summary_rows = []
for h in HORIZONS:
    ics = np.array(list(per_stock_ic[h].values()))
    k = len(ics)
    mean_ic = float(ics.mean()) if k > 0 else float("nan")
    std_ic = float(ics.std(ddof=1)) if k > 1 else float("nan")
    cross_t = mean_ic / (std_ic / np.sqrt(k)) if (k > 1 and std_ic > 0) else float("nan")

    if pooled_records[h]:
        pooled_df = pd.concat(pooled_records[h], ignore_index=True)
        pooled_rho = spearman_corr(pooled_df["sentiment"].to_numpy(), pooled_df["fwd"].to_numpy())
        pooled_n = len(pooled_df)
        pooled_t = pooled_t_stat(pooled_rho, pooled_n)
    else:
        pooled_rho, pooled_n, pooled_t = float("nan"), 0, float("nan")

    summary_rows.append({
        "horizon_days": h, "n_stocks": k, "mean_ic_cross_stock": mean_ic, "std_ic": std_ic,
        "cross_stock_t": cross_t, "pooled_ic": pooled_rho, "pooled_n": pooled_n, "pooled_t": pooled_t,
    })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

print()
print("--- Per-stock IC breakdown ---")
detail_header = f"{'Horizon':>8}  " + "".join(f"{t:>8}" for t in TICKERS)
print(detail_header)
print("-" * len(detail_header))
for h in HORIZONS:
    row = f"{h:>7}d  "
    for t in TICKERS:
        val = per_stock_ic[h].get(t)
        row += f"{val:>8.3f}" if val is not None else f"{'--':>8}"
    print(row)

print()
print("=" * 100)
print("Interpretation")
print("=" * 100)
n_sig_cross = sum(1 for r in summary_rows if not np.isnan(r["cross_stock_t"]) and abs(r["cross_stock_t"]) > 2)
n_sig_pooled = sum(1 for r in summary_rows if not np.isnan(r["pooled_t"]) and abs(r["pooled_t"]) > 2)
print(f"{len(HORIZONS)} horizons tested. Cross-stock t-stat exceeds |t|>2 at {n_sig_cross} horizon(s); "
      f"pooled t-stat exceeds |t|>2 at {n_sig_pooled} horizon(s).")
print("Pooled t-stats treat all stock-days as independent, which they are not (same-day news across "
      "correlated names, autocorrelated sentiment) -- they will read as more significant than they "
      "really are. The cross-stock t-test (11 independent per-stock ICs) is the more defensible number.")
print("This is a diagnostic scan only: no strategy was built, no threshold was selected, and no")
print("configuration was tuned based on these results.")
