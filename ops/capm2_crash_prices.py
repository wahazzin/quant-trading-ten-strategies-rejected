"""Prior-close helper for ops/capm2_hourly_crash_watch.py (fix 2026-11-23, weekly review W48).

During market hours Alpaca's 1Day bars request with end=<today> already contains TODAY's
in-progress bar, whose "close" is the latest trade. The crash watch used bars[-1]["c"] as
"the most recent prior close", so it compared the live price with itself and read ~0.00%
every day (W48 logs: -0.24%..+0.11%), i.e. the pre-registered -8% crash rule could never fire.
This picks the last bar dated BEFORE today's New York trading date instead.
Rule, thresholds and sentiment veto are unchanged.
"""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")


def ny_today(now=None):
    now = now or datetime.now(timezone.utc)
    return now.astimezone(NY).date().isoformat()


def prior_close(bars, today_ny):
    """Close of the last daily bar strictly before `today_ny` (YYYY-MM-DD, New York date).

    Alpaca daily bar timestamps are midnight New York time in UTC ("2026-11-20T05:00:00Z"),
    so the first 10 characters are the bar's New York trading date. Returns None if no
    earlier bar exists."""
    earlier = [b for b in bars if str(b["t"])[:10] < today_ny]
    if not earlier:
        return None
    return float(earlier[-1]["c"])
