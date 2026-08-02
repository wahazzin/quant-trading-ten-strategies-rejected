import os
from dotenv import load_dotenv

load_dotenv()

# LEGACY (2026-08-02): the project moved the live forward test from IBKR to
# Alpaca -- see ROADMAP.md constraint C1 and RESEARCH_LOG.md's Phase 6
# section for why (no locally-running Gateway process for a months-long
# forward test, and a much cheaper real cost structure). Nothing in ops/
# imports these anymore; kept only for any one-off IBKR diagnostic script
# still in the repo (e.g. ops/audit.py) that hasn't been retired.
IB_HOST = os.getenv("IB_HOST", "127.0.0.1")
IB_PORT = int(os.getenv("IB_PORT", 4002))
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", 1))