"""
capm2_allocation.py -- pure allocation logic for Test 20 / Phase 6d
(RESEARCH_LOG.md). No broker calls here on purpose -- these functions are
easy to unit-test in isolation, exactly like combiner.py and
capacity_gate.py were tonight. ops/capm2_weekly_rebalance.py and
ops/capm2_hourly_crash_watch.py are what actually call a broker.
"""
import numpy as np

MAX_WEIGHT_PER_NAME = 0.40  # hard cap, see RESEARCH_LOG.md Test 20 spec


def compute_weights(trailing_returns: dict) -> dict:
    """
    trailing_returns: {ticker: trailing_3mo_return} for ONE group (US or
    Swedish -- never mix groups, they never compete for weight against
    each other per the pre-registered spec).

    Returns {ticker: weight}. Weights sum to AT MOST 1.0, not always
    exactly 1.0 -- see the cap-shortfall note below. All 0.0 for every
    name is an explicit valid state (nobody beat the group average this
    cycle), not an error.

    Method (fixed, pre-registered): z-score within the group, floored at
    0 (an underperformer gets cut to literal ZERO, not just reduced),
    then normalized to sum to 1.0, then capped at MAX_WEIGHT_PER_NAME.

    CAP-SHORTFALL RULE (fixed here, before any live use -- this was an
    undecided edge case caught by testing, not discovered live): if there
    are too few winning names to reach 100% while respecting the cap
    (e.g. only 2 winners at a 40% cap maxes out at 80%), the shortfall is
    left as CASH, never forced into a zero-floored underperformer and
    never allowed to push a winner over the cap. A strategy that only
    trusts a couple of names should hold less, not fake full investment
    by diluting into names it just rejected.
    """
    tickers = list(trailing_returns.keys())
    values = np.array([trailing_returns[t] for t in tickers], dtype=float)

    mu = values.mean()
    sigma = values.std()
    if sigma == 0 or not np.isfinite(sigma):
        z = np.zeros_like(values)
    else:
        z = (values - mu) / sigma

    floored = np.maximum(z, 0.0)
    total = floored.sum()
    if total == 0:
        # every name is at-or-below the group average -- no winners to tilt
        # toward this cycle. Explicit all-zero state, not an error.
        return {t: 0.0 for t in tickers}

    weights = floored / total
    weights = _apply_cap(dict(zip(tickers, weights)))
    return weights


def _apply_cap(weights: dict, cap: float = MAX_WEIGHT_PER_NAME) -> dict:
    """
    Iteratively caps any weight above `cap`, redistributing the excess
    proportionally among OTHER WINNERS still under the cap (weight > 0
    AND < cap) -- NEVER toward a name already at exactly 0 (a
    zero-floored underperformer). If no such recipient exists, the
    excess is left unallocated (cash), not force-fed anywhere.
    """
    weights = dict(weights)
    for _ in range(len(weights) + 1):  # bounded iterations, always converges
        over = {t: w for t, w in weights.items() if w > cap}
        if not over:
            break
        excess = sum(w - cap for w in over.values())
        for t in over:
            weights[t] = cap

        under = {t: w for t, w in weights.items() if 0 < w < cap}
        under_total = sum(under.values())
        if under_total == 0:
            break  # no eligible winner left to redistribute into -- stays cash
        for t in under:
            weights[t] += excess * (under[t] / under_total)

    return weights


def check_crash(daily_return: float, sentiment_score, drop_threshold: float = -0.08,
                 sentiment_threshold: float = -0.3) -> bool:
    """
    Hourly crash-watch trigger, per RESEARCH_LOG.md Test 20 spec.
    Fires when BOTH conditions hold: an unusually large single-day drop
    AND a negative shadow-sentiment reading. Requiring both avoids
    reacting to a big drop with no bad-news confirmation, and avoids
    reacting to bad sentiment with no actual price impact yet.

    sentiment_score: None means no sentiment data available -- treated as
    NOT triggering (never assume bad news from missing data).
    drop_threshold/sentiment_threshold: placeholders, not backtested --
    see RESEARCH_LOG.md's TODO to set these once, deliberately, before
    relying on this in a real cycle.
    """
    if sentiment_score is None:
        return False
    return daily_return <= drop_threshold and sentiment_score <= sentiment_threshold
