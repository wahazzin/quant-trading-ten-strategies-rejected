"""
Universe screener -- dynamic eligibility, rebuilt every rebalance.

WHY THIS EXISTS
The prior CAPM bot could only ever trade 6 hardcoded tickers, so it could
never be better than the initial guess. It wasn't deciding anything; it was
executing a decision already made months earlier. This screener rebuilds
the tradeable universe from scratch each rebalance.

CRITICAL RULE -- FILTERS ARE STRUCTURAL, NEVER PERFORMANCE-BASED
Eligibility depends only on price, liquidity, and available history. It
NEVER depends on past returns, momentum, or any outcome measure. Filtering
on outcomes bakes hindsight into selection -- the exact error measured in
Test 11, where picking the "best 6 of 30" by Jensen's alpha (computed with
data through 2025, then tested on 2021-2025) produced 30 percentage points
of pure hindsight advantage.

POINT-IN-TIME
Every filter uses only data dated strictly before the decision date. The
screener is handed a pre-sliced frame; it must never read raw files itself.
"""
from dataclasses import dataclass

import pandas as pd


@dataclass
class ScreenConfig:
    price_min: float = 5.0          # below this, spreads and penny-stock
    price_max: float = 100.0        # artifacts dominate (Test 17b: 130% sd)
    min_adv: float = 250_000        # shares/day, 20-day average
    min_history_days: int = 252     # need a year before a stock is eligible
    adv_window: int = 20


class UniverseScreener:
    """Rebuilds the eligible universe at a given decision date."""

    def __init__(self, config: ScreenConfig | None = None):
        self.config = config or ScreenConfig()

    def screen(self, data: pd.DataFrame, as_of) -> pd.DataFrame:
        """
        Args:
            data:  long-format frame with ticker, date, close, volume
            as_of: decision date. ONLY rows strictly before this are used.

        Returns:
            One row per eligible ticker, with the latest pre-as_of values
            and the metrics each filter was judged on -- so any inclusion
            can be audited after the fact.
        """
        cfg = self.config
        as_of = pd.Timestamp(as_of)

        # HARD point-in-time cut. Everything downstream is blind to the future.
        hist = data[data["date"] < as_of].copy()
        if hist.empty:
            return pd.DataFrame()

        hist = hist.sort_values(["ticker", "date"])

        # rolling average volume, per ticker
        hist["adv"] = (hist.groupby("ticker")["volume"]
                       .transform(lambda s: s.rolling(cfg.adv_window,
                                                      min_periods=cfg.adv_window)
                                  .mean()))

        # history length available as of the decision date
        hist["n_obs"] = hist.groupby("ticker").cumcount() + 1

        latest = hist.groupby("ticker").tail(1).set_index("ticker")

        eligible = latest[
            (latest["close"] >= cfg.price_min)
            & (latest["close"] <= cfg.price_max)
            & (latest["adv"] >= cfg.min_adv)
            & (latest["n_obs"] >= cfg.min_history_days)
            & latest["close"].notna()
            & latest["adv"].notna()
        ].copy()

        eligible["as_of"] = as_of
        return eligible[["date", "close", "volume", "adv", "n_obs", "as_of"]]


    def report(self, data: pd.DataFrame, as_of) -> dict:
        """
        Diagnostic: how many tickers each filter removed. Run this whenever
        the universe size looks surprising -- a silently shrinking universe
        is how a backtest turns into a small-sample illusion.
        """
        cfg = self.config
        as_of = pd.Timestamp(as_of)
        hist = data[data["date"] < as_of].copy()
        if hist.empty:
            return {"error": "no data before as_of"}

        hist = hist.sort_values(["ticker", "date"])
        hist["adv"] = (hist.groupby("ticker")["volume"]
                       .transform(lambda s: s.rolling(cfg.adv_window,
                                                      min_periods=cfg.adv_window)
                                  .mean()))
        hist["n_obs"] = hist.groupby("ticker").cumcount() + 1
        latest = hist.groupby("ticker").tail(1)

        total = len(latest)
        fail_price = (~latest["close"].between(cfg.price_min,
                                               cfg.price_max)).sum()
        fail_liq = (latest["adv"] < cfg.min_adv).sum()
        fail_hist = (latest["n_obs"] < cfg.min_history_days).sum()
        passed = len(self.screen(data, as_of))

        return {
            "as_of": str(as_of.date()),
            "tickers_with_data": int(total),
            "failed_price_filter": int(fail_price),
            "failed_liquidity_filter": int(fail_liq),
            "failed_history_filter": int(fail_hist),
            "eligible": int(passed),
            "eligible_pct": round(passed / total * 100, 1) if total else 0.0,
        }
