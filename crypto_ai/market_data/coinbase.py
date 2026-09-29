"""
coinbase.py -- thin client for Coinbase Exchange PUBLIC market data (no API key, read-only).

CONCEPTS:
  * ticker  = the live top-of-book: best bid, best ask, last price, rolling 24h volume.
  * candle  = OHLCV bar for a fixed period ("granularity" in seconds: 3600 = 1h, 86400 = 1d).
              Coinbase returns rows [time, low, high, open, close, volume], NEWEST FIRST,
              where `time` is when the candle STARTED.
  * A candle is only "closed" once start + granularity <= now. The newest row is usually still
    forming -- using it would leak information that did not exist yet at decision time.
    closed_only() removes it. Every feature is built from closed candles only.

Any network/data problem raises DataFault. The runner treats that as "skip this cycle and log
it", never "guess".
"""
import time
from datetime import datetime

import requests

BASE = "https://api.exchange.coinbase.com"
HEADERS = {"User-Agent": "crypto-ai-research/0.1 (paper-trading research, read-only)"}


class DataFault(Exception):
    pass


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


class CoinbaseClient:
    def __init__(self, base=BASE, pause=0.12, session=None):
        self.base = base
        self.pause = pause          # stay well under the public rate limit
        self.s = session or requests.Session()

    def _get(self, path, params=None, tries=3):
        last = None
        for i in range(tries):
            try:
                r = self.s.get(self.base + path, params=params, headers=HEADERS, timeout=15)
                if r.status_code == 429:
                    time.sleep(1.5 * (i + 1))
                    continue
                r.raise_for_status()
                time.sleep(self.pause)
                return r.json()
            except Exception as e:
                last = e
                time.sleep(1 + i)
        raise DataFault(f"{path}: {last}")

    def ticker(self, product):
        j = self._get(f"/products/{product}/ticker")
        try:
            return {"price": float(j["price"]), "bid": float(j["bid"]), "ask": float(j["ask"]),
                    "volume_base": float(j["volume"]), "time": parse_time(j["time"])}
        except (KeyError, ValueError) as e:
            raise DataFault(f"{product} ticker malformed: {e}")

    def candles(self, product, granularity):
        rows = self._get(f"/products/{product}/candles", {"granularity": granularity})
        out = [{"t": int(r[0]), "low": float(r[1]), "high": float(r[2]), "open": float(r[3]),
                "close": float(r[4]), "volume": float(r[5])} for r in rows]
        out.sort(key=lambda c: c["t"])
        return out


def closed_only(candles, granularity, now_ts):
    """Drop any candle that has not finished yet (start + granularity > now)."""
    return [c for c in candles if c["t"] + granularity <= now_ts]
