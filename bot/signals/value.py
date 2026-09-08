"""
Value signal -- book-to-market ranking.

EVIDENCE STATUS: unproven (underpowered, NOT disproven)
    Test 10 measured annualized alpha +5.28% at t=1.46 on 2010-2018.
    Economically meaningful, statistically short of the t>2 bar. The
    reason is a hard data ceiling, not a weak effect: SEC XBRL fundamentals
    only start ~2010, giving 102 monthly observations where ~192 are needed
    to resolve an effect this size. It is the ONLY candidate in this project
    that failed on data availability rather than on evidence against it.
    Corrected for the CommonStockSharesOutstanding bug it became +5.55%,
    t=1.52 -- verdict unchanged.

    Currently running as a live forward test (Phase 6, since 2026-08-03).

LOOKAHEAD PROTECTION
    Every fundamental fact carries BOTH its fiscal period end AND its
    `filed` date. A fact may only be used from its FILED date onward --
    a Q4 2015 figure filed in March 2016 is unknowable in January 2016.
    This is enforced in latest_known_value() below and must never be
    relaxed for convenience.
"""
import numpy as np
import pandas as pd

from bot.signals.base import Signal


def latest_known_value(facts: pd.DataFrame, ticker: str,
                       concept: str, as_of: pd.Timestamp):
    """
    Most recent value for `concept` that was ALREADY FILED as of `as_of`.
    Returns None when nothing qualifies -- never falls back to a later
    filing, which would be lookahead.
    """
    sub = facts[(facts["ticker"] == ticker)
                & (facts["concept"] == concept)
                & (facts["filed"] <= as_of)]
    if sub.empty:
        return None
    return sub.sort_values("filed").iloc[-1]["value"]


class ValueSignal(Signal):
    """
    Book-to-market: StockholdersEquity / (price * shares outstanding).
    Higher = cheaper relative to book value = more attractive.
    """

    name = "value_bm"
    evidence = "unproven"
    required_columns = ("ticker", "date", "close")

    def __init__(self, facts: pd.DataFrame, as_of: pd.Timestamp,
                 min_share_ratio: float = 1.0):
        """
        Args:
            facts: long frame with ticker, concept, value, filed
            as_of: decision date -- only facts filed before this are used
            min_share_ratio: shares outstanding must be at least this
                multiple of one day's average volume. Guards the
                CommonStockSharesOutstanding data bug found in Test 10,
                where Up-C/holdco structures reported absurd share counts
                (AMCR appeared to have 13,001 total shares, giving a
                book-to-market of 20,672x and topping the "cheapest" list).
        """
        self.facts = facts
        self.as_of = pd.Timestamp(as_of)
        self.min_share_ratio = min_share_ratio

    def score(self, data: pd.DataFrame) -> pd.Series:
        out = {}
        for ticker, grp in data.groupby("ticker"):
            price = grp.sort_values("date").iloc[-1]["close"]
            if not np.isfinite(price) or price <= 0:
                continue

            equity = latest_known_value(self.facts, ticker,
                                        "StockholdersEquity", self.as_of)
            shares = latest_known_value(self.facts, ticker,
                                        "CommonStockSharesOutstanding",
                                        self.as_of)
            if equity is None or shares is None:
                continue
            if equity <= 0 or shares <= 0:
                continue      # negative book value -> not a value stock

            # sanity floor against corrupt share counts
            if "adv" in grp.columns:
                adv = grp.sort_values("date").iloc[-1].get("adv", np.nan)
                if np.isfinite(adv) and shares < adv * self.min_share_ratio:
                    continue

            market_cap = price * shares
            if market_cap <= 0:
                continue
            out[ticker] = equity / market_cap

        return pd.Series(out, name=self.name, dtype=float)
