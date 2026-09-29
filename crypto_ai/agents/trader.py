"""
trader.py -- the AI trader: builds the prompt, calls the LLM, validates, returns proposals and
thesis updates.

WHAT IT SEES each cycle (the "information menu" -- v0.1 is price/volume only):
  1. market features for every asset (closed candles + live quote)
  2. its own portfolio: equity, cash, weights, P&L, stop prices, breaker state, turnover left
  3. its open theses (persistent beliefs) including confidence history
  4. last cycle's feedback: what the risk engine approved/clipped/rejected and why, plus any
     stop-outs or breaker events since
The same function will later accept richer menus (news, sentiment) for NEW arms -- that's how the
ablation arms will be built without touching ai_pv.

WHAT IT CANNOT DO: size orders, bypass limits, or see the future. It returns target weights;
risk/engine.py decides what actually happens.
"""
import hashlib
import json
import os

from crypto_ai.agents.schemas import validate
from crypto_ai.risk.engine import Proposal

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURE_KEYS = ["ret_1h", "ret_6h", "ret_24h", "ret_7d", "ret_30d", "vol_7d_ann", "vol_30d_ann",
                "volume_ratio", "dist_sma20", "dist_sma50", "drawdown_from_30d_high",
                "rsi14_daily", "corr_btc_30d"]


def render_system_prompt(cfg):
    with open(os.path.join(PKG_DIR, cfg["llm"]["prompt_file"]), encoding="utf-8") as f:
        tpl = f.read()
    rk, c = cfg["risk"], cfg["costs"]
    rules = "\n".join([
        f"- Max weight per asset: {rk['max_weight_major']:.0%} for BTC/ETH, {rk['max_weight_alt']:.0%} for others.",
        f"- Every position has a stop-loss, max {rk['max_stop_pct']:.0f}% below average cost; you may set a tighter one (min {rk['min_stop_pct']:.0f}%).",
        f"- A position's loss at its stop may not exceed {rk['max_risk_per_position_equity_pct']:.0f}% of equity (wide stop => smaller max weight).",
        f"- Max total invested {rk['max_gross_exposure']:.0%}; at least {1 - rk['max_gross_exposure']:.0%} stays in USD.",
        f"- Max {rk['max_alt_cluster']:.0%} combined in assets other than BTC and ETH.",
        f"- Max turnover {rk['max_turnover_24h']:.0%} of equity per rolling 24h (sells/stops always allowed).",
        f"- Min order ${rk['min_order_usd']:.0f}. One order per asset per cycle.",
        f"- Daily loss of {rk['daily_loss_limit_pct']:.0f}% blocks new buys for 24h. Drawdown of {rk['drawdown_halt_pct']:.0f}% from peak liquidates everything and locks buying for {rk['halt_lockout_days']} days.",
    ])
    costs = (f"{c['fee_bps']:.0f} bps fee per side, plus the bid/ask spread, plus slippage "
             f"(~{c['slippage_bps_major']:.0f} bps BTC/ETH, ~{c['slippage_bps_alt']:.0f} bps others); "
             f"a round trip costs roughly {2 * c['fee_bps'] / 100:.1f}% or more")
    return (tpl.replace("{{UNIVERSE}}", ", ".join(cfg["universe"]))
               .replace("{{COSTS}}", costs).replace("{{RISK_RULES}}", rules))


def _pct(x):
    return None if x is None else round(100 * x, 2)


def build_user_prompt(cycle_id, snapshot, pf, theses, feedback, cfg, now):
    assets = snapshot["assets"]
    mids = {a: v["mid"] for a, v in assets.items()}
    eq = pf.equity(mids)
    market = {}
    for a, v in assets.items():
        f = v["features"]
        row = {"price": round(v["mid"], 6), "spread_bps": v["spread_bps"],
               "volume_24h_usd_M": round(v["volume_24h_usd"] / 1e6, 2)}
        for k in FEATURE_KEYS:
            val = f.get(k)
            row[k] = (round(val, 3) if k in ("rsi14_daily", "volume_ratio", "corr_btc_30d")
                      else _pct(val)) if val is not None else None
        market[a] = row
    positions = {}
    for a, p in pf.positions.items():
        positions[a] = {"weight_pct": round(100 * p.qty * mids[a] / eq, 2),
                        "avg_cost": round(p.avg_cost, 6),
                        "unrealized_pnl_pct": round(100 * (mids[a] / p.avg_cost - 1), 2),
                        "stop_price": round(p.stop_price, 6), "stop_pct": p.stop_pct,
                        "opened_cycle": p.opened_cycle}
    rk = cfg["risk"]
    portfolio = {"equity_usd": round(eq, 2), "cash_usd": round(pf.cash, 2),
                 "cash_pct": round(100 * pf.cash / eq, 2) if eq > 0 else 0,
                 "drawdown_from_peak_pct": round(100 * pf.drawdown(mids), 2),
                 "positions": positions,
                 "turnover_used_24h_pct": round(100 * pf.traded_last_24h(now) / eq, 2) if eq > 0 else 0,
                 "turnover_limit_24h_pct": 100 * rk["max_turnover_24h"],
                 "buys_blocked_until": pf.no_buys_until, "halted_until": pf.halted_until}
    payload = {"cycle_id": cycle_id, "time_utc": cycle_id,
               "note": "Returns/vol/distances are in PERCENT. Features use closed candles only.",
               "market": market, "portfolio": portfolio, "open_theses": theses,
               "last_cycle_feedback": feedback}
    return ("Current state (JSON). Decide and respond with the JSON object only.\n\n"
            + json.dumps(payload, indent=1, sort_keys=True))


def decide(llm, cfg, cycle_id, snapshot, pf, theses, feedback, now):
    """Returns a result dict. Never raises for bad model output -- it degrades to DO_NOTHING."""
    system = render_system_prompt(cfg)
    user = build_user_prompt(cycle_id, snapshot, pf, theses, feedback, cfg, now)
    mids = {a: v["mid"] for a, v in snapshot["assets"].items()}
    ctx = {"cycle_id": cycle_id, "snapshot": snapshot, "weights": pf.weights(mids)}
    attempts, parsed, errors = [], None, []
    for _ in range(1 + cfg["llm"]["parse_retries"]):
        resp = llm.complete(system, user, ctx)
        parsed, errors = validate(resp["text"], cfg, cycle_id)
        attempts.append({"raw": resp["text"], "errors": errors, "model_id": resp.get("model_id"),
                         "provider": resp.get("provider"), "endpoint_errors": resp.get("endpoint_errors"),
                         "stop_reason": resp.get("stop_reason"), "tokens_in": resp.get("tokens_in"),
                         "tokens_out": resp.get("tokens_out"), "latency_s": resp.get("latency_s")})
        if parsed is not None:
            break
    proposals = []
    if parsed is not None:
        for d in parsed["decisions"]:
            proposals.append(Proposal("ai_pv", d["asset"], d["action"], d.get("target_weight"),
                                      d.get("stop_loss_pct"), "ai"))
    return {"ok": parsed is not None, "parsed": parsed, "errors": errors, "attempts": attempts,
            "proposals": proposals, "user_prompt": user,
            "system_sha256": hashlib.sha256(system.encode()).hexdigest(),
            "prompt_sha256": hashlib.sha256((system + user).encode()).hexdigest()}


def apply_thesis_updates(theses, parsed, cycle_id):
    """Returns (new_theses, history_rows). Closed/invalidated theses leave the open set but stay
    in the history log forever."""
    theses = json.loads(json.dumps(theses))
    conf = {d["asset"]: d.get("confidence") for d in parsed["decisions"]}
    history = []
    for u in parsed["thesis_updates"]:
        a = u["asset"]
        history.append({"cycle_id": cycle_id, **u, "confidence": conf.get(a)})
        if u["status"] in ("closed", "invalidated"):
            theses.pop(a, None)
            continue
        t = theses.get(a, {"opened_cycle": cycle_id, "confidence_history": []})
        t.update({"status": u["status"], "updated_cycle": cycle_id, "summary": u.get("summary", ""),
                  "reasons": u.get("reasons", []), "invalidation": u.get("invalidation", [])})
        if conf.get(a) is not None:
            t["confidence_history"] = (t["confidence_history"] + [{"cycle": cycle_id, "confidence": conf[a]}])[-20:]
        theses[a] = t
    return theses, history
