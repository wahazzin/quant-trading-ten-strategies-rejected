"""
capm2_shadow_stop.py -- Test 21 (pre-registered in RESEARCH_LOG.md, 2026-10-08): a SHADOW copy of CAPM2 with
one extra rule -- a 15% trailing stop per holding. READ-ONLY on the broker (GET only); no orders, no extra account.

Rule (fixed): each trading day after the close, for every name the shadow holds: peak = highest daily close since
the shadow bought it; if today's close <= 85% of that peak, the shadow sells it at today's close and holds the cash
until the next weekly rebalance. Once per week, at the close of the first day on which CAPM2's real rebalance has
filled (Monday normally, Tuesday if Monday was a holiday or the job ran late; Wednesday at the latest), it re-copies
CAPM2's real holdings as of that close (share counts + cash, scaled to its own equity; rebuilt from the account's
FILL activities, GET only). A trading day whose session has not closed yet is never processed (partial bar). Costs: 0.0045% per side (Alpaca ~0.009% round trip, as CAPM2's own cost model).
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


def session_open_day(clock):
    """The exchange date (New York) whose session has NOT finished yet, or None. Alpaca's daily bar for that day is
    partial (fixed 2026-10-26: the Monday 06:20 snapshot started 7h late, at 13:37 UTC = 7 minutes after the open,
    and the shadow booked the 2026-10-19 opening price as that day's 'close' and never revisited it)."""
    if not clock:
        return None
    ts, nc = str(clock.get("timestamp", ""))[:10], str(clock.get("next_close", ""))[:10]
    return ts if ts and ts == nc else None


def fills_since(h, start_day):
    """Real-account FILL activities from start_day 00:00 UTC on (GET only; account activities, not orders)."""
    out, token = [], None
    while True:
        p = {"activity_types": "FILL", "after": f"{start_day}T00:00:00Z", "direction": "asc", "page_size": 100}
        if token:
            p["page_token"] = token
        page = get(f"{PAPER}/v2/account/activities", h, **p)
        out += page
        if len(page) < 100:
            return out
        token = page[-1]["id"]


def real_as_of(day, positions, cash, fills, prices):
    """Real account (positions as dicts, cash, equity) at `day`'s close, rebuilt from the CURRENT positions/cash by
    undoing every fill made after that day. Lets the weekly re-copy happen on the right day even when it is
    processed later (fixed 2026-10-26: CAPM2's rebalance ran on Tuesday 2026-10-20 because GitHub started both Monday
    slots after the close, but the shadow had already re-copied on Monday 10-19, i.e. last week's holdings, and kept
    them all week)."""
    qty = {p["symbol"]: float(p["qty"]) for p in positions}
    cash = float(cash)
    for f in fills:
        if f["transaction_time"][:10] <= day:
            continue
        q, px = float(f["qty"]), float(f["price"])
        sign = 1 if f["side"] == "buy" else -1               # undo: a later buy added shares and spent cash
        qty[f["symbol"]] = qty.get(f["symbol"], 0.0) - sign * q
        cash += sign * q * px
    pos, eq = [], cash
    for s, q in sorted(qty.items()):
        if abs(q) < 1e-9:
            continue
        px = prices.get(s, {}).get(day)
        if px is None:
            px = next((float(p["current_price"]) for p in positions if p["symbol"] == s), 0.0)
        pos.append({"symbol": s, "qty": str(q), "market_value": str(q * px), "current_price": str(px)})
        eq += q * px
    return pos, cash, eq


def week_start(day):
    d = date.fromisoformat(day)
    return (d - timedelta(days=d.weekday())).isoformat()


def rebalanced_by(day, fills):
    """True once the real account has traded in `day`'s ISO week (on or before `day`), i.e. CAPM2's weekly
    rebalance has run; from Wednesday on we sync anyway (a week where the rebalance needed no orders)."""
    ws = week_start(day)
    return date.fromisoformat(day).weekday() >= 2 or any(ws <= f["transaction_time"][:10] <= day for f in fills)


def mirror(g, real_positions, real_cash, real_equity, equity, px_day, prices):
    """Re-copy CAPM2's real account with the shadow's own equity: the real share counts and cash, scaled by
    equity / real_equity (fees charged on the turnover).
    Fixed 2026-10-12: it used to derive shares from weights (market_value / equity, at the broker's live price)
    divided by the last daily close -- when those two prices differ (the 2026-10-08 start ran pre-market at
    10:27 UTC), the 'copy' held different share counts than the real account, so shadow and real drifted apart
    with no stop involved (US +0.18%, SE -0.43% after two days)."""
    scale = equity / real_equity
    old = {s: v["shares"] * prices.get(s, {}).get(px_day, v["entry"]) for s, v in g["holdings"].items()}
    new_h, cost = {}, 0.0
    for p in real_positions:
        s = p["symbol"]
        px = prices.get(s, {}).get(px_day) or float(p["current_price"])
        qty = float(p["qty"]) if p.get("qty") is not None else float(p["market_value"]) / float(p["current_price"])
        shares = qty * scale
        cost += abs(shares * px - old.get(s, 0.0)) * FEE
        new_h[s] = {"shares": shares, "peak": px, "entry": px}
    for s, v in old.items():
        if s not in new_h:
            cost += v * FEE
    g["holdings"], g["cash"] = new_h, float(real_cash) * scale - cost


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
        try:
            clock = get(f"{PAPER}/v2/clock", h)
        except Exception:  # noqa: BLE001
            clock = None
        unfinished = session_open_day(clock)
        start = (date.fromisoformat(g["last_day"]) if g["last_day"] else today - timedelta(days=7)).isoformat()
        try:
            fills = fills_since(h, week_start(start)) if g["last_day"] else []
        except Exception as e:  # noqa: BLE001  -- skip this run; the next one catches up (last_day not advanced)
            print(f"WARNING {gname}: could not read fills ({type(e).__name__}); shadow not updated this run")
            continue
        syms = sorted(set(g["holdings"]) | {p["symbol"] for p in pos} | {f["symbol"] for f in fills})
        prices = closes(h, syms, start) if syms else {}
        days = sorted({d for s in prices.values() for d in s if (not g["last_day"] or d > g["last_day"])
                       and d != unfinished and d <= today.isoformat()})
        if not g["last_day"]:                                    # first run: exact copy of the real account today
            day = days[-1] if days else today.isoformat()
            mirror(g, pos, acct["cash"], real_eq, real_eq, day, prices)
            g["last_day"], g["last_sync"] = day, day
            st["log"].append({"day": day, "group": gname, "event": "START", "equity": real_eq})
            g["history"].append({"day": day, "shadow": real_eq, "real": real_eq})
            continue
        for day in days:
            eq = g["cash"] + sum(v["shares"] * prices.get(s, {}).get(day, v["peak"]) for s, v in g["holdings"].items())
            r_pos, r_cash, r_eq = real_as_of(day, pos, acct["cash"], fills, prices)
            # weekly re-copy: once per ISO week, at the close of the first day on which CAPM2's real rebalance has
            # run (Monday normally; Tuesday when Monday was a holiday or GitHub started the job too late)
            new_week = date.fromisoformat(day).isocalendar()[:2] != date.fromisoformat(g["last_sync"]).isocalendar()[:2]
            if new_week and rebalanced_by(day, fills):
                mirror(g, r_pos, r_cash, r_eq, eq, day, prices)            # weekly: copy CAPM2's new weights
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
            g["history"].append({"day": day, "shadow": eq, "real": round(r_eq, 2)})
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
