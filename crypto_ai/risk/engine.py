"""
engine.py -- the deterministic risk engine (rules R1-R14 in SPEC_v0.1.md section 8).

THE MOST IMPORTANT DESIGN RULE: no LLM, no randomness, no network in this file. The AI proposes
target weights; this code decides what is actually allowed. Same inputs always give the same
outputs, so it can be unit-tested exhaustively (tests/test_crypto_ai_risk.py).

Each proposal comes back as one decision dict:
    status  APPROVED  -> executed exactly as asked
            CLIPPED   -> executed but smaller/tighter than asked (codes say which rules bit)
            REJECTED  -> not executed (codes say why)
            NOOP      -> HOLD / DO_NOTHING, nothing to do
Sells are evaluated BEFORE buys, and every rule sees the running hypothetical portfolio, so two
individually-fine buys cannot jointly break a cap.

Rules that REDUCE risk (sells, stops, breaker exits) are never blocked by the turnover limit:
a risk limit that can trap you in a position would be worse than no limit.
"""
from dataclasses import dataclass
from datetime import timedelta

from crypto_ai.journal import iso, parse_iso

EPS = 1e-9


@dataclass
class Proposal:
    arm: str
    asset: str
    action: str                    # BUY ADD HOLD REDUCE SELL EXIT DO_NOTHING
    target_weight: float = None    # desired share of total equity AFTER the trade
    stop_loss_pct: float = None    # optional; tighten-only
    source: str = "ai"


def _dec(p, status, codes, **kw):
    d = {"arm": p.arm, "asset": p.asset, "action": p.action, "status": status,
         "requested_weight": p.target_weight, "codes": list(codes), "side": None,
         "notional_usd": 0.0, "final_weight": None, "stop_pct": None, "full_exit": False}
    d.update(kw)
    return d


def update_breakers(pf, mids, now, cfg):
    """Circuit breakers. Returns (liquidate_everything: bool, events: list)."""
    rk, events, liquidate = cfg["risk"], [], False
    eq = pf.equity(mids)
    if pf.halted_until and parse_iso(pf.halted_until) <= now:
        pf.halted_until = None
        pf.peak_equity = eq                     # fresh baseline so we don't instantly re-trip
        events.append({"type": "BREAKER_RESUME", "arm": pf.name, "equity": eq})
    if pf.no_buys_until and parse_iso(pf.no_buys_until) <= now:
        pf.no_buys_until = None
    pf.peak_equity = max(pf.peak_equity, eq)

    today = now.strftime("%Y-%m-%d")
    if pf.day_start_date != today:
        pf.day_start_date, pf.day_start_equity = today, eq
    day_ret = eq / pf.day_start_equity - 1 if pf.day_start_equity > 0 else 0.0
    if day_ret <= -rk["daily_loss_limit_pct"] / 100 and not pf.no_buys_until:
        pf.no_buys_until = iso(now + timedelta(hours=24))
        events.append({"type": "R9_DAILY_LOSS", "arm": pf.name, "day_return": day_ret})

    dd = eq / pf.peak_equity - 1 if pf.peak_equity > 0 else 0.0
    if dd <= -rk["drawdown_halt_pct"] / 100 and not pf.halted_until:
        pf.halted_until = iso(now + timedelta(days=rk["halt_lockout_days"]))
        liquidate = True
        events.append({"type": "R10_DRAWDOWN_HALT", "arm": pf.name, "drawdown": dd,
                       "halted_until": pf.halted_until})
    return liquidate, events


def evaluate(proposals, pf, snap, cfg, now):
    rk, assets = cfg["risk"], snap["assets"]
    majors, fee = set(cfg["majors"]), cfg["costs"]["fee_bps"] / 1e4
    mids = {a: v["mid"] for a, v in assets.items()}
    eq = pf.equity(mids)
    out = []
    if eq <= 0:
        return [_dec(p, "REJECTED", ["R14_NO_EQUITY"]) for p in proposals]

    run = dict(pf.weights(mids))               # running weights as approvals accumulate
    traded = pf.traded_last_24h(now)
    cash_left = pf.cash
    halted = bool(pf.halted_until and parse_iso(pf.halted_until) > now)
    nobuy = bool(pf.no_buys_until and parse_iso(pf.no_buys_until) > now)
    seen = set()

    sells, buys = [], []
    for p in proposals:
        if p.action in ("HOLD", "DO_NOTHING"):
            out.append(_dec(p, "NOOP", []))
        elif p.action in ("BUY", "ADD"):
            buys.append(p)
        elif p.action in ("REDUCE", "SELL", "EXIT"):
            sells.append(p)
        else:
            out.append(_dec(p, "REJECTED", ["R12_UNKNOWN_ACTION"]))

    # ---------------- sells first ----------------
    for p in sells:
        a = p.asset
        if a not in assets:
            out.append(_dec(p, "REJECTED", ["R11_NO_QUOTE"])); continue
        if a in seen:
            out.append(_dec(p, "REJECTED", ["R7_DUPLICATE_ORDER"])); continue
        seen.add(a)
        cw = run.get(a, 0.0)
        if cw <= EPS:
            out.append(_dec(p, "REJECTED", ["R14_NOT_HELD"])); continue
        full = p.action in ("SELL", "EXIT")
        tw = 0.0 if full else p.target_weight
        if tw is None or not (0 <= tw <= 1):
            out.append(_dec(p, "REJECTED", ["R12_BAD_TARGET"])); continue
        if not full and tw >= cw - EPS:
            out.append(_dec(p, "REJECTED", ["R12_REDUCE_NOT_LOWER"])); continue
        if tw <= EPS:
            full, tw = True, 0.0
        delta = cw * eq if full else (cw - tw) * eq
        if not full and delta < rk["min_order_usd"]:
            out.append(_dec(p, "REJECTED", ["R7_MIN_ORDER"])); continue
        traded += delta
        run[a] = tw
        cash_left += delta * (1 - fee)
        out.append(_dec(p, "APPROVED", [], side="sell", notional_usd=delta,
                        final_weight=tw, full_exit=full))

    # ---------------- then buys ----------------
    for p in buys:
        a, codes = p.asset, []
        if a not in assets:
            out.append(_dec(p, "REJECTED", ["R11_NO_QUOTE"])); continue
        if a in seen:
            out.append(_dec(p, "REJECTED", ["R7_DUPLICATE_ORDER"])); continue
        seen.add(a)
        cw = run.get(a, 0.0)
        if p.target_weight is None or not (0 <= p.target_weight <= 1):
            out.append(_dec(p, "REJECTED", ["R12_BAD_TARGET"])); continue
        if p.action == "BUY" and cw > 1e-4:
            out.append(_dec(p, "REJECTED", ["R12_BUY_ON_HELD_USE_ADD"])); continue
        if p.action == "ADD" and cw <= 1e-4:
            out.append(_dec(p, "REJECTED", ["R12_ADD_NOT_HELD_USE_BUY"])); continue
        if halted:
            out.append(_dec(p, "REJECTED", ["R13_HALTED"])); continue
        if nobuy:
            out.append(_dec(p, "REJECTED", ["R9_DAILY_LOSS_NO_BUYS"])); continue
        if assets[a]["spread_bps"] > rk["max_spread_bps"]:
            out.append(_dec(p, "REJECTED", ["R8_SPREAD_TOO_WIDE"])); continue

        req = rk["max_stop_pct"] if p.stop_loss_pct is None else p.stop_loss_pct
        sp = min(max(req, rk["min_stop_pct"]), rk["max_stop_pct"])
        if abs(sp - req) > EPS:
            codes.append("R5_STOP_ADJUSTED")

        cap_asset = rk["max_weight_major"] if a in majors else rk["max_weight_alt"]
        cap_risk = rk["max_risk_per_position_equity_pct"] / sp
        tw = p.target_weight
        if tw > min(cap_asset, cap_risk) + EPS:
            codes.append("R2_LOSS_AT_STOP_CAP" if cap_risk < cap_asset else "R1_ASSET_CAP")
            tw = min(cap_asset, cap_risk)
        if a not in majors:
            alt_others = sum(w for k, w in run.items() if k not in majors and k != a)
            room = rk["max_alt_cluster"] - alt_others
            if tw > room + EPS:
                codes.append("R4_ALT_CLUSTER"); tw = max(room, 0.0)
        gross_others = sum(w for k, w in run.items() if k != a)
        room = rk["max_gross_exposure"] - gross_others
        if tw > room + EPS:
            codes.append("R3_GROSS_EXPOSURE"); tw = max(room, 0.0)
        by_cash = cw + cash_left / (1 + fee) / eq
        if tw > by_cash + EPS:
            codes.append("R14_CASH"); tw = by_cash
        delta = (tw - cw) * eq
        if delta <= EPS * eq:
            out.append(_dec(p, "REJECTED", codes + ["NO_ROOM_TO_ADD"])); continue
        vol_cap = rk["max_order_pct_of_24h_volume"] * assets[a]["volume_24h_usd"]
        if delta > vol_cap:
            codes.append("R8_LIQUIDITY"); delta = vol_cap
        remaining = rk["max_turnover_24h"] * eq - traded
        if remaining <= 0:
            out.append(_dec(p, "REJECTED", codes + ["R6_TURNOVER_EXHAUSTED"])); continue
        if delta > remaining:
            codes.append("R6_TURNOVER"); delta = remaining
        if delta < rk["min_order_usd"]:
            out.append(_dec(p, "REJECTED", codes + ["R7_MIN_ORDER"])); continue
        run[a] = cw + delta / eq
        traded += delta
        cash_left -= delta * (1 + fee)
        out.append(_dec(p, "CLIPPED" if codes else "APPROVED", codes, side="buy",
                        notional_usd=delta, final_weight=run[a], stop_pct=sp))
    return out
