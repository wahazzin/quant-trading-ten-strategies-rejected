"""
capm2_status.py -- READ-ONLY snapshot of the CAPM2 forward test (US + SE Alpaca paper accounts) so the weekly
review (a scheduled Claude session without broker keys) can read it from GitHub.

Only GET requests: account, positions, portfolio history, SPY bars. It never places, changes or cancels
orders (tests/test_capm2_status.py checks this file contains no POST/PATCH/DELETE).
Output: status/capm2_status.json + status/capm2_status.md
"""
import json
import os
import time
from datetime import datetime, timedelta, timezone

import requests

PAPER = "https://paper-api.alpaca.markets"
DATA = "https://data.alpaca.markets"
GROUPS = {"US": ("CAPM2_US_KEY_ID", "CAPM2_US_SECRET_KEY"), "SE": ("CAPM2_SE_KEY_ID", "CAPM2_SE_SECRET_KEY")}


def get(url, h, **params):
    r = requests.get(url, headers=h, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def snapshot():
    out = {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"), "accounts": {}}
    spy_h = None
    for g, (k, s) in GROUPS.items():
        if not os.environ.get(k):
            out["accounts"][g] = {"error": f"{k} not set"}; continue
        h = {"APCA-API-KEY-ID": os.environ[k], "APCA-API-SECRET-KEY": os.environ[s]}
        spy_h = spy_h or h
        try:
            a = get(f"{PAPER}/v2/account", h)
            pos = get(f"{PAPER}/v2/positions", h)
            hist = get(f"{PAPER}/v2/account/portfolio/history", h, period="3M", timeframe="1D")
            eq = [(t, e) for t, e in zip(hist.get("timestamp", []), hist.get("equity", [])) if e]
            out["accounts"][g] = {
                "equity": float(a["equity"]), "last_equity": float(a["last_equity"]), "cash": float(a["cash"]),
                "positions": [{"symbol": p["symbol"], "qty": float(p["qty"]), "market_value": float(p["market_value"]),
                               "unrealized_plpc": float(p["unrealized_plpc"])} for p in pos],
                "equity_history": eq}
        except Exception as e:
            out["accounts"][g] = {"error": f"{type(e).__name__}: {str(e)[:150]}"}
    if spy_h:
        try:
            start = (datetime.now(timezone.utc) - timedelta(days=95)).strftime("%Y-%m-%d")
            bars = get(f"{DATA}/v2/stocks/SPY/bars", spy_h, timeframe="1Day", start=start, feed="iex", limit=200)
            out["spy"] = [(b["t"][:10], b["c"]) for b in bars.get("bars", [])]
        except Exception as e:
            out["spy_error"] = type(e).__name__
    return out


def markdown(s):
    L = [f"# CAPM2 status ({s['generated_utc']} UTC) — read-only snapshot", ""]
    for g, a in s["accounts"].items():
        if "error" in a:
            L += [f"## {g}: ERROR {a['error']}", ""]; continue
        eq = a["equity_history"]
        def ret(days):
            if len(eq) < 2:
                return None
            cutoff = eq[-1][0] - days * 86400
            base = next((e for t, e in eq if t >= cutoff), eq[0][1])
            return a["equity"] / base - 1
        r7, r30 = ret(7), ret(30)
        f = lambda x: "—" if x is None else f"{x:+.2%}"
        peak = max([e for _, e in eq] + [a["equity"]])
        L += [f"## {g} account", f"- Equity {a['equity']:,.2f} · cash {a['cash']:,.2f} · 7d {f(r7)} · 30d {f(r30)} · "
              f"drawdown from peak {a['equity'] / peak - 1:+.2%}",
              "- Positions: " + (", ".join(f"{p['symbol']} {p['market_value']:,.0f} ({p['unrealized_plpc']:+.1%})" for p in a["positions"]) or "none"), ""]
    if s.get("spy"):
        sp = s["spy"]
        def sret(days):
            cutoff = (datetime.strptime(sp[-1][0], "%Y-%m-%d") - timedelta(days=days)).strftime("%Y-%m-%d")
            base = next((c for d, c in sp if d >= cutoff), sp[0][1])
            return sp[-1][1] / base - 1
        L += [f"SPY (IEX closes): 7d {sret(7):+.2%} · 30d {sret(30):+.2%}"]
    return "\n".join(L) + "\n"


def message(text):
    """Post a plain text to this repo's Discord channel (weekly review). Not a broker call."""
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if url:
        requests.post(url, json={"content": text[:1900]}, timeout=15)   # Discord webhook only


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["message"]:
        message(os.environ.get("TEXT", ""))
        raise SystemExit(0)
    s = snapshot()
    os.makedirs("status", exist_ok=True)
    json.dump(s, open("status/capm2_status.json", "w"), indent=1)
    md = markdown(s)
    open("status/capm2_status.md", "w").write(md)
    print(md)
