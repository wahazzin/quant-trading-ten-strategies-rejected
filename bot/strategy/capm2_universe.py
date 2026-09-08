"""
capm2_universe.py -- the ONE place the Test 20 / Phase 6d ticker lists and
account credentials live, so ops/capm2_weekly_rebalance.py and
ops/capm2_hourly_crash_watch.py can never drift apart on which symbols
belong to which group or which .env keys to use.

Deliberately IDENTICAL to Phase 6c's static CAPM groups (RESEARCH_LOG.md
Test 20 spec) -- the allocation METHOD is the variable being tested, not
the universe.
"""

GROUPS = {
    "US": {
        "label": "CAPM2 US (Test 20 reallocation)",
        "tickers": ["NVDA", "AVGO", "LLY", "WMT", "XOM", "GOOGL"],
        "key_env": "CAPM2_US_KEY_ID",
        "secret_env": "CAPM2_US_SECRET_KEY",
    },
    "SE": {
        "label": "CAPM2 Swedish (Test 20 reallocation)",
        "tickers": ["SPOT", "ERIC", "AZN", "ALV", "OTLY"],
        "key_env": "CAPM2_SE_KEY_ID",
        "secret_env": "CAPM2_SE_SECRET_KEY",
    },
}
