"""
Sentiment shadow-veto -- SCORES AND LOGS ONLY, NEVER BLOCKS A TRADE.

Per RESEARCH_LOG Tests 13/14 (pooled and shock-conditional news sentiment
both failed as standalone signals), this module does NOT feed sentiment
into the combiner or capacity gate. Its only job is to attach a same-day
sentiment score to every candidate the orchestrator considers, so
trade_attribution.py can later ask, honestly and retrospectively: "would
vetoing on sentiment have helped or hurt, if we had used it?" That
question can only be answered from logged history -- never by assuming
sentiment matters and wiring the veto in live first.

Reuses the exact FinBERT scoring convention from
research/signal_tests/sentiment_test.py: sentiment_score = P(positive) -
P(negative) from ProsusAI/finbert, scored on the headline only, mean
across headlines when multiple exist for the same ticker/window. This is
a deliberate reuse, not a reimplementation -- the methodology already
has a track record (2 completed tests) and there's no reason to risk
introducing a subtly different scoring convention here.

FinBERT is loaded ONCE per SentimentVeto instance (expensive to load,
cheap to reuse) -- never re-instantiate this class per ticker or per
cycle. Persisting scores to the journal for later attribution is the
CALLER's responsibility (e.g. whatever script runs the orchestrator on a
schedule) -- this class only computes and returns a score, it does not
touch the database.
"""
import os
from typing import Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

NEWS_URL = "https://data.alpaca.markets/v1beta1/news"


class SentimentVeto:
    def __init__(self, lookback_days: int = 1):
        self.lookback_days = lookback_days
        api_key = os.environ.get("APCA_API_KEY_ID")
        api_secret = os.environ.get("APCA_API_SECRET_KEY")
        if not api_key or not api_secret:
            raise RuntimeError(
                "APCA_API_KEY_ID / APCA_API_SECRET_KEY not found in .env -- "
                "the shadow-veto needs Alpaca news access to fetch anything."
            )
        self._headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret}
        self._model = None
        self._tokenizer = None
        self._id2label = None

    def _load_model(self):
        if self._model is not None:
            return
        import torch  # noqa: F401 (imported for side effect of availability check)
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        print("  [sentiment_veto] loading ProsusAI/finbert (first call only, then cached)...")
        self._tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self._model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self._model.eval()
        self._id2label = self._model.config.id2label

    def _fetch_headlines(self, ticker: str, as_of: pd.Timestamp) -> list:
        start = (as_of - pd.Timedelta(days=self.lookback_days)).strftime("%Y-%m-%dT00:00:00Z")
        end = as_of.strftime("%Y-%m-%dT23:59:59Z")
        params = {"start": start, "end": end, "symbols": ticker, "limit": 50, "sort": "desc"}
        try:
            resp = requests.get(NEWS_URL, headers=self._headers, params=params, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  [sentiment_veto] news fetch failed for {ticker}: {e}")
            return []
        return [a["headline"] for a in resp.json().get("news", [])]

    def _score_headlines(self, headlines: list) -> Optional[float]:
        """Exposed separately from score() so it can be tested without
        needing a live news fetch."""
        if not headlines:
            return None
        self._load_model()
        import torch
        with torch.no_grad():
            inputs = self._tokenizer(
                headlines, return_tensors="pt", padding=True, truncation=True, max_length=64
            )
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).numpy()

        pos_idx = next(i for i, lbl in self._id2label.items() if lbl.lower() == "positive")
        neg_idx = next(i for i, lbl in self._id2label.items() if lbl.lower() == "negative")
        scores = probs[:, pos_idx] - probs[:, neg_idx]
        return float(np.mean(scores))

    def score(self, ticker: str, ticker_data: pd.DataFrame) -> Optional[float]:
        """
        Matches the Orchestrator's sentiment_fn signature exactly:
        (ticker, ticker_data) -> Optional[float]. `ticker_data` (price
        rows) is accepted but unused for fetching -- news is pulled
        independently by ticker + as_of, not derived from price data.
        Returns None (never 0.0) when there's no news to score, so "no
        news" and "neutral news" are never confused downstream.

        LOGGING ONLY. The Orchestrator's contract guarantees this never
        filters `ranked`. Do not wire this into the combiner or capacity
        gate without first running trade_attribution.analyze() on
        accumulated logged history and getting a real answer -- the same
        bar every other signal had to clear.
        """
        if ticker_data.empty or "date" not in ticker_data.columns:
            as_of = pd.Timestamp.utcnow()
        else:
            as_of = pd.Timestamp(ticker_data["date"].max())

        headlines = self._fetch_headlines(ticker, as_of)
        return self._score_headlines(headlines)
