"""
social_sentiment_veto.py -- StockTwits shadow-veto. SCORES AND LOGS ONLY,
NEVER BLOCKS A TRADE. Same contract as bot/monitor/sentiment_veto.py
(news/FinBERT) -- this is the social-media counterpart flagged in
ROADMAP.md Section 11, built as a shadow-logger only, not a validated
signal.

WHY THIS STAYS A LOGGER, NOT A SIGNAL, UNTIL PROVEN OTHERWISE:
  - Twitter/X was ruled out (paywalled historical data for backtesting).
  - Facebook was ruled out (no usable public API since Cambridge Analytica).
  - StockTwits is the legitimate remaining option -- a real, free, public
    API, stock-specific by design.
  - BUT: turning this into something the combiner is allowed to use
    requires the same rigor Tests 13/14 needed for news sentiment (years
    of history, declustering, SPY-drift control) PLUS a new problem those
    tests never faced -- pump-and-dump/bot coordination can fake
    real-looking bullish sentiment. None of that validation work has been
    done. This module only computes and logs a score.

SCORING CONVENTION: StockTwits lets users self-tag each message as
"Bullish", "Bearish", or leave it untagged. Score = (bullish_count -
bearish_count) / tagged_count, range -1 to +1. Untagged messages are
excluded from the denominator entirely (not counted as neutral) --
excluding them is a real design choice, not a neutral default, since
retail sentiment tools built this way commonly report only a small
minority of messages as tagged; this may need revisiting once real data
volume is seen. Returns None (not 0.0) when there are zero TAGGED
messages, so "nobody expressed an opinion" and "opinions cancelled out"
are never confused.

API NOTE -- VERIFY BEFORE TRUSTING: this uses StockTwits' public,
unauthenticated symbol-stream endpoint. I have not been able to verify
current rate limits, terms of service, or endpoint stability against
live documentation in this session (no search access) -- confirm this
still matches StockTwits' current API behavior before relying on it for
anything beyond this shadow-logging test.
"""
from typing import Optional

import pandas as pd
import requests

STOCKTWITS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"


class SocialSentimentVeto:
    def __init__(self, limit: int = 30, timeout: int = 15):
        self.limit = limit
        self.timeout = timeout

    def _fetch_messages(self, ticker: str) -> list:
        url = STOCKTWITS_URL.format(symbol=ticker)
        try:
            resp = requests.get(url, params={"limit": self.limit}, timeout=self.timeout)
            if resp.status_code == 429:
                print(f"  [social_sentiment] rate limited fetching {ticker} -- skipping this cycle")
                return []
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  [social_sentiment] fetch failed for {ticker}: {e}")
            return []
        return resp.json().get("messages", [])

    def _score_messages(self, messages: list) -> Optional[float]:
        bullish = 0
        bearish = 0
        for msg in messages:
            entities = msg.get("entities") or {}
            sentiment = entities.get("sentiment")
            if not sentiment:
                continue  # untagged -- excluded from denominator, not counted as neutral
            basic = sentiment.get("basic")
            if basic == "Bullish":
                bullish += 1
            elif basic == "Bearish":
                bearish += 1

        tagged_total = bullish + bearish
        if tagged_total == 0:
            return None
        return (bullish - bearish) / tagged_total

    def score(self, ticker: str, ticker_data: pd.DataFrame) -> Optional[float]:
        """
        Matches the Orchestrator's sentiment_fn signature exactly:
        (ticker, ticker_data) -> Optional[float]. `ticker_data` (price
        rows) is accepted but unused -- messages are fetched independently
        by ticker, not derived from price data.

        LOGGING ONLY. Never wire this into the combiner or capacity gate
        without first running the same declustered, SPY-drift-controlled,
        bot-manipulation-checked analysis Tests 13/14 required for news
        sentiment -- the same bar every other signal had to clear.
        """
        messages = self._fetch_messages(ticker)
        return self._score_messages(messages)
