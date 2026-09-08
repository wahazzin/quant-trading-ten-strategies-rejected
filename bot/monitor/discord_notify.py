"""
discord_notify.py -- posts trade alerts to Discord via webhook.

SILENCE BY DEFAULT (see ROADMAP.md Section 12). This module only has
functions for genuinely notable events: a fill, a stop-loss hit, a
target hit, a circuit-breaker trip. There is deliberately NO "cycle
completed" or "here's what I'm thinking" function -- routine dry-run
cycles and capacity/screening details stay in the terminal log, not
Discord. If a new event type is added later, ask "would I want a phone
buzz for this specifically" before adding it here.
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")


def _post(content: str, color_emoji: str = ""):
    if not WEBHOOK_URL:
        print(f"  [discord_notify] DISCORD_WEBHOOK_URL not set -- would have sent: {content}")
        return False
    try:
        resp = requests.post(WEBHOOK_URL, json={"content": f"{color_emoji} {content}"}, timeout=10)
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"  [discord_notify] failed to post to Discord: {e}")
        return False


def notify_fill(symbol, side, quantity, price):
    return _post(f"**Fill:** {side.upper()} {quantity} {symbol} @ ${price:.2f}", "\U0001F7E2")


def notify_stop_hit(symbol, quantity, stop_price, entry_price):
    pnl_pct = (stop_price - entry_price) / entry_price * 100
    return _post(
        f"**Stop-loss hit:** {symbol} closed {quantity} shares @ ${stop_price:.2f} "
        f"(entry ${entry_price:.2f}, {pnl_pct:+.1f}%)",
        "\U0001F534",
    )


def notify_target_hit(symbol, quantity, target_price, entry_price):
    pnl_pct = (target_price - entry_price) / entry_price * 100
    return _post(
        f"**Target hit:** {symbol} closed {quantity} shares @ ${target_price:.2f} "
        f"(entry ${entry_price:.2f}, {pnl_pct:+.1f}%)",
        "\U0001F3AF",
    )


def notify_circuit_breaker(reason: str):
    return _post(f"**CIRCUIT BREAKER TRIPPED:** {reason} -- no new entries until reset.", "\U0001F6D1")


def notify_crash_cut(symbol, quantity, daily_return_pct, sentiment_score):
    return _post(
        f"**Crash-watch cut:** {symbol} sold {quantity} shares -- "
        f"{daily_return_pct:+.1f}% today AND sentiment {sentiment_score:+.2f} "
        f"both crossed the trigger. Cut to zero outside the weekly cycle.",
        "\U000026A0",
    )
