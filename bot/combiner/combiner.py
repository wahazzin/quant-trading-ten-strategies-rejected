"""
Signal combiner -- turns per-signal scores into one ranked list.

DESIGN (locked, see ENGINE_ARCHITECTURE.md and RESEARCH_LOG.md Test 19):
    Default combiner is a transparent, equal-weighted average of
    cross-sectionally z-scored signal outputs. NO ML combiner is built
    here -- with only one signal (value_bm) in the registry today,
    training any combiner would fit noise to n=1 input. An ML slot
    activates only once >=2 signals have INDEPENDENTLY demonstrated edge
    (each on its own pre-registered test, each surviving its own holdout)
    -- never before that, and never as a substitute for that.

This module owns no capacity logic (Test 19's lesson lives in the
capacity gate, not here) and no risk logic -- it only turns a DataFrame
of per-signal scores into one ranked Series.
"""
import numpy as np
import pandas as pd

from bot.signals.base import Signal, SignalRegistry


def zscore(s: pd.Series) -> pd.Series:
    """
    Cross-sectional z-score of one signal's scores across tickers.
    NaN-safe: a ticker a signal didn't score stays NaN, never becomes 0 --
    zero would silently mean "average," which is not what a missing score
    means.
    """
    mu = s.mean(skipna=True)
    sigma = s.std(skipna=True)
    if not np.isfinite(sigma) or sigma == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sigma


class Combiner:
    """
    Combines multiple already-computed signal scores into one ranking.

    weights: optional {signal_name: weight}. Any signal not listed
    defaults to weight 1.0 -- i.e. equal-weighted unless told otherwise.
    A weight is a deliberate override, not a promotion; nothing here
    changes evidence status.
    """

    def __init__(self, weights: dict | None = None):
        self.weights = weights or {}

    def combine(self, signal_scores: pd.DataFrame) -> pd.Series:
        """
        signal_scores: DataFrame indexed by ticker, one column per signal
        (the exact shape SignalRegistry.score_all() returns).

        Returns a single Series indexed by ticker, sorted descending --
        higher = more attractive, same convention as every Signal.
        A ticker missing from every signal is dropped, not scored as 0.
        """
        if signal_scores.empty:
            return pd.Series(dtype=float)

        z = signal_scores.apply(zscore)

        w = pd.Series({c: self.weights.get(c, 1.0) for c in z.columns})

        weighted = z.mul(w, axis=1)
        present = z.notna().mul(w, axis=1)

        numer = weighted.sum(axis=1, skipna=True)
        denom = present.sum(axis=1)

        combined = numer / denom
        combined[denom == 0] = np.nan

        return combined.dropna().sort_values(ascending=False)

    def combine_from_registry(
        self,
        registry: SignalRegistry,
        data: pd.DataFrame,
        allowed_evidence=("unproven", "validated"),
    ) -> pd.Series:
        """
        Convenience path: score every active signal, drop any whose
        evidence status isn't in allowed_evidence (rejected signals stay
        registered for comparison/logging per base.py's contract, but
        must never silently feed a live ranking), then combine.
        """
        scores = registry.score_all(data)

        blocked = [
            sig.name for sig in registry.active()
            if sig.evidence not in allowed_evidence and sig.name in scores.columns
        ]
        if blocked:
            print(f"  [combiner] excluding non-{allowed_evidence} signals: {blocked}")
            scores = scores.drop(columns=blocked)

        return self.combine(scores)
