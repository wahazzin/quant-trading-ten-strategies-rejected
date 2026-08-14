"""
guard.py -- prevents a script from running against the wrong broker by
accident. Added after an incident where 20 pre-migration GTC orders,
left resting at IBKR instead of being cancelled before the switch to
Alpaca, filled AFTER the migration was already complete -- producing
duplicate exposure to the same 20-stock portfolio at both brokers
simultaneously (see RESEARCH_LOG.md, Phase 6). The root cause wasn't a
code bug; it was a script that would have happily kept talking to IBKR
even after this project had moved on, with nothing to stop it.

ACTIVE_BROKER in .env is the single source of truth for which broker is
currently live. Every script that connects to a broker calls
require_broker(name) with the ONE broker it is written for, before
doing anything else. This function does not infer, guess, or fall
back -- a mismatch is always a hard, loud failure.
"""
import os
from dotenv import load_dotenv

load_dotenv()


def require_broker(expected):
    active = os.environ.get("ACTIVE_BROKER")
    if active != expected:
        raise SystemExit(
            f"REFUSING TO RUN: this script is written for broker '{expected}', but "
            f"ACTIVE_BROKER in .env is '{active}'. If you really mean to run this "
            f"script, set ACTIVE_BROKER={expected} in .env first -- otherwise it is "
            f"about to talk to the wrong broker, which is exactly the incident this "
            f"check exists to prevent."
        )
