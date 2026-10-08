"""
capm2_shadow_stop.py -- Test 21 (pre-registered in RESEARCH_LOG.md, 2026-10-08): a SHADOW copy of CAPM2 with
one extra rule -- a 15% trailing stop per holding. READ-ONLY on the broker (GET only); no orders, no extra account.

Rule (fixed): each trading day after the close, for every name the shadow holds: peak = highest daily close since
the shadow bought it; if today's close <= 85% of that peak, the shadow sells it at today's close and holds the cash
until the next weekly rebalance. Every Monday (after CAPM2's real rebalance) the shadow re-copies CAPM2's real
weights with its own equity. Costs: 0.0045% per side (Alpaca ~0.009% round trip, as CAPM2's own cost model).
State: status/shadow_stop.json (capm-status branch). Report: status/shadow_stop.md
"""
import json
import os
from datetime import date, datetime, timedelta, timezone

import requests

PAPER, DATA = "https://paper-api.alpaca.markets", "https://data.alpaca.markets"
GROUPS = {"US": ("CAPM2_US_KEY_ID", "CAPM2_US_SECRET_KEY"), "SE": ("CAPM2_SE_KEY_ID", "CAPM2_SE_SECRET_KEY")}
STOP = 0.85
FEE = 0.000045
STATE = "status/shadow_stop.json"


def get(url, h, **p):
    r = requests.get(url, headers=h, params=p, timeout=30)
    r.raise_for_status()
    return r.json()


def closes(h, symbols, start):
    out = {}
    for s in symbols:
        bars = get(f"{DATA}/v2/stocks/{s}/bars", h, timeframe="1Day", start=start, feed="iex", limit=1000).get("bars", [])
        out[s] = {b["t"][:10]: b["c"] for b in bars}
    return out


def mirror(g, real_positions, real_equity, equity, px_day, prices):
    """Re-copy CAPM2's real weights with the shadow's own equity (fees charged on the turnover)."""
    old = {s: v["shares"] * prices.get(s, {}).get(px_day, v["entry"]) for s, v in g["holdings"].items()}
    new_h, cost = {}, 0.0
    for p in real_positions:
        s, w = p["symbol"], float(p["market_value"]) / real_equity
        px = prices.get(s, {}).get(px_day) or float(p["current_price"])
        val = equity * w
        cost += abs(val - old.get(s, 0.0)) * FEE
        new_h[s] = {"shares": val / px, "peak": px, "entry": px}
    for s, v in old.items():
        if s not in new_h:
            cost += v * FEE
    invested = sum(v["shares"] * v["entry"] for v in new_h.values())
    g["holdings"], g["cash"] = new_h, equity - invested - cost


def run(today=None):
    today = today or datetime.now(timezone.utc).date()
    st = json.load(open(STATE)) if os.path.exists(STATE) else {"start": today.isoformat(), "groups": {}, "log": []}
    for gname, (k, sk) in GROUPS.items():
        if not os.environ.get(k):
            continue
        h = {"APCA-API-KEY-ID": os.environ[k], "APCA-API-SECRET-KEY": os.environ[sk]}
        acct = get(f"{PAPER}/v2/account", h)
        pos = get(f"{PAPER}/v2/positions", h)
        real_eq = float(acct["equity"])
        g = st["groups"].setdefault(gname, {"cash": 0.0, "holdings": {}, "last_day": None, "last_sync": None,
                                             "history": []})
        syms = sorted(set(g["holdings"]) | {p["symbol"] for p in pos})
        start = (date.fromisoformat(g["last_day"]) if g["last_day"] else today - timedelta(days=7)).isoformat()
        prices = closes(h, syms, start) if syms else {}
        days = sorted({d for s in prices.values() for d in s if (not g["last_day"] or d > g["last_day"])})
        if not g["last_day"]:                                    # first run: exact copy of the real account today
            day = days[-1] if days else today.isoformat()
            mirror(g, pos, real_eq, real_eq, day, prices)
            g["last_day"], g["last_sync"] = day, day
            st["log"].append({"day": day, "group": gname, "event": "START", "equity": real_eq})
            g["history"].append({"day": day, "shadow": real_eq, "real": real_eq})
            continue
        for day in days:
            eq = g["cash"] + sum(v["shares"] * prices.get(s, {}).get(day, v["peak"]) for s, v in g["holdings"].items())
            is_monday = date.fromisoformat(day).weekday() == 0
            if is_monday and day == days[-1] and g["last_sync"] != day:
                mirror(g, pos, real_eq, eq, day, prices)           # weekly: copy CAPM2's new weights
                g["last_sync"] = day
                st["log"].append({"day": day, "group": gname, "event": "WEEKLY_SYNC", "equity": eq})
            for s, v in list(g["holdings"].items()):
                c = prices.get(s, {}).get(day)
                if c is None:
                    continue
                v["peak"] = max(v["peak"], c)
                if c <= STOP * v["peak"]:
                    val = v["shares"] * c
                    g["cash"] += val * (1 - FEE)
                    del g["holdings"][s]
                    st["log"].append({"day": day, "group": gname, "event": "STOP", "symbol": s, "close": c,
                                      "peak": v["peak"], "value": val})
            eq = g["cash"] + sum(v["shares"] * prices.get(s, {}).get(day, v["peak"]) for s, v in g["holdings"].items())
            g["history"].append({"day": day, "shadow": eq, "real": real_eq if day == days[-1] else None})
            g["last_day"] = day
    os.makedirs("status", exist_ok=True)
    json.dump(st, open(STATE, "w"), indent=1)
    L = [f"# Test 21 — CAPM2 with a 15% trailing stop (shadow) vs real CAPM2", "",
         f"Started {st['start']}. Pre-registered in RESEARCH_LOG.md. Interim numbers are informational only.", ""]
    for gname, g in st["groups"].items():
        hist = g["history"]
        if not hist:
            continue
        first = hist[0]
        last_real = next((x["real"] for x in reversed(hist) if x["real"]), None)
        L.append(f"- **{gname}**: shadow {hist[-1]['shadow']:,.2f} ({hist[-1]['shadow'] / first['shadow'] - 1:+.2%}) vs "
                 f"real CAPM2 {last_real:,.2f} ({last_real / first['real'] - 1:+.2%}) since start; "
                 f"shadow holds {', '.join(g['holdings']) or 'cash only'}")
    stops = [x for x in st["log"] if x["event"] == "STOP"]
    L += ["", f"Stops fired: {len(stops)}" + "".join(f"\n- {x['day']} {x['group']} {x['symbol']} at {x['close']:.2f} (peak {x['peak']:.2f})" for x in stops)]
    open("status/shadow_stop.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    run()
