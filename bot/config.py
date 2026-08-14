import os
from dotenv import load_dotenv

load_dotenv()

# RETIRED (2026-08-03): IBKR is fully decommissioned for this project -- see
# RESEARCH_LOG.md's Phase 6 section for the incident that made this final
# (pre-migration GTC orders filled at IBKR after the Alpaca migration was
# already live, producing duplicate exposure to the same portfolio at both
# brokers). All IBKR-dependent scripts now live in ops/legacy_ibkr/, each
# guarded by bot/broker/guard.py's require_broker("ibkr") -- none will run
# unless ACTIVE_BROKER=ibkr is explicitly set for that one invocation. Kept
# only because ops/legacy_ibkr/*.py still imports these constants.
IB_HOST = os.getenv("IB_HOST", "127.0.0.1")
IB_PORT = int(os.getenv("IB_PORT", 4002))
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", 1))