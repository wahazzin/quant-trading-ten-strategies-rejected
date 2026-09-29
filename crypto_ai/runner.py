"""
runner.py -- runs ONE decision cycle. Scheduled every 6 hours; safe to run more often (idempotent).

ORDER OF A CYCLE (and why this order):
  1. verify the spec lock           -> no quiet goalpost moving
  2. cycle_id = UTC time floored to the 6h slot; already done? exit
  3. snapshot market data           -> one shared view of the world for ALL arms (fairness)
  4. per arm: stops -> breakers -> (liquidate if halted)   -> risk is handled before new ideas
  5. reference benchmarks rebalance (no risk engine)
  6. AI decides (LLM call)          -> slow; prices move while it thinks
  7. fetch FRESH quotes             -> fills happen at post-decision prices, never the ones the AI saw
  8. trend_quant + AI proposals -> risk engine -> paper exchange
  9. theses, equity, feedback, state -> saved atomically; everything appended to the journal

Failure policy: bad/missing market data => whole cycle skipped and logged (DATA_FAULT). Bad LLM
output => AI does nothing this cycle (AI_FAULT); baselines still run. Nothing is ever guessed.
If an asset is delisted every cycle becomes DATA_FAULT until an amendment handles it -- loud on
purpose (PREREGISTRATION.md section 8).

CLI:
  python -m crypto_ai.runner --state-dir crypto_ai_state_dryrun --dry-run      # mock LLM, no lock
  python -m crypto_ai.runner --state-dir crypto_ai_state                       # the real thing
"""
import argparse
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_ai.agents.llm import LLMError, make_llm
from crypto_ai.agents.trader import apply_thesis_updates, decide
from crypto_ai.execution.paper_exchange import execute, replay_stops
from crypto_ai.journal import Journal, iso, now_utc, parse_iso
from crypto_ai.lock import load_config, verify_lock
from crypto_ai.market_data.coinbase import CoinbaseClient, DataFault
from crypto_ai.market_data.features import build_snapshot, fresh_quotes
from crypto_ai.portfolio.baselines import rebalance_direct, reference_targets, trend_proposals
from crypto_ai.portfolio.state import Portfolio
from crypto_ai.risk.engine import evaluate, update_breakers

BANNER = "INTERIM -- INFORMATIONAL ONLY. Interim performance is not evidence of edge."


def cycle_id_for(now):
    return now.replace(hour=(now.hour // 6) * 6, minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:00Z")


def _alert(msg, dry_run):
    if dry_run:
        return
    try:
        from bot.monitor.discord_notify import notify_circuit_breaker
        notify_circuit_breaker(f"[crypto_ai] {msg}")
    except Exception:
        pass                                   # alerts are best-effort, never block a cycle


def _load_arms(j, cfg):
    arms = {}
    for name in cfg["arms"]:
        d = j.load_json(f"arms/{name}.json")
        arms[name] = Portfolio.from_dict(d) if d else Portfolio(name, cfg["capital"]["initial_cash_usd"])
    return arms


def _execute_decisions(pf, decs, quotes, cfg, now, cycle_id, cause):
    fills = []
    for d in decs:
        if d["status"] not in ("APPROVED", "CLIPPED"):
            continue
        q = quotes.get(d["asset"])
        if q is None:
            d["status"], d["codes"] = "REJECTED", d["codes"] + ["R11_NO_FRESH_QUOTE"]
            continue
        f = execute(pf, d["asset"], d["side"], d["notional_usd"], q, cfg, now, cause, cycle_id,
                    stop_pct=d["stop_pct"], full_exit=d["full_exit"])
        if f:
            fills.append(f)
    return fills


def run_cycle(state_dir, cfg, client, llm, now=None, dry_run=False, require_lock=True):
    t0 = time.time()
    now = now or now_utc()
    j = Journal(state_dir)
    cid = cycle_id_for(now)

    if require_lock:
        ok, problems = verify_lock(state_dir)
        if not ok:
            j.append("events.jsonl", {"ts": iso(now), "cycle_id": cid, "type": "LOCK_MISMATCH",
                                      "problems": problems})
            _alert("LOCK MISMATCH -- refusing to run: " + "; ".join(problems), dry_run)
            return {"cycle_id": cid, "status": "LOCK_MISMATCH", "problems": problems}

    if any(r.get("cycle_id") == cid and r.get("status") != "DATA_FAULT" for r in j.read("cycles.jsonl")):
        return {"cycle_id": cid, "status": "ALREADY_DONE"}

    # ---- 3. snapshot -------------------------------------------------------------------------
    try:
        snap, hourly = build_snapshot(cfg, client, now)
    except DataFault as e:
        j.append("events.jsonl", {"ts": iso(now), "cycle_id": cid, "type": "DATA_FAULT", "error": str(e)})
        j.append("cycles.jsonl", {"cycle_id": cid, "started_at": iso(now), "status": "DATA_FAULT",
                                  "error": str(e), "dry_run": dry_run})
        return {"cycle_id": cid, "status": "DATA_FAULT", "error": str(e)}
    j.append("snapshots.jsonl", {"cycle_id": cid, "ts": iso(now), "assets": snap["assets"]})
    mids = {a: v["mid"] for a, v in snap["assets"].items()}
    snap_quotes = {a: {"mid": v["mid"], "bid": v["bid"], "ask": v["ask"],
                       "volume_24h_usd": v["volume_24h_usd"]} for a, v in snap["assets"].items()}

    arms = _load_arms(j, cfg)
    fills, events = [], []

    # ---- 4. stops and breakers (constrained arms only) ----------------------------------------
    for name, pf in arms.items():
        if not cfg["arms"][name]["constrained"]:
            continue
        since = parse_iso(pf.last_ts).timestamp() if pf.last_ts else None
        for f in replay_stops(pf, hourly, snap["assets"], cfg, now, cid, since_ts=since):
            fills.append(f)
            events.append({"type": "STOP_TRIGGERED", "arm": name, "asset": f["asset"],
                           "price": f["price"], "stop_price": f["stop_price"]})
        liquidate, evs = update_breakers(pf, mids, now, cfg)
        events += evs
        if liquidate:
            for a in list(pf.positions):
                f = execute(pf, a, "sell", 0, snap_quotes[a], cfg, now, "breaker", cid, full_exit=True)
                if f:
                    fills.append(f)
            _alert(f"{name}: drawdown halt, liquidated, buys locked until {pf.halted_until}", dry_run)

    # ---- 5. reference benchmarks --------------------------------------------------------------
    for name, pf in arms.items():
        if cfg["arms"][name]["constrained"]:
            continue
        targets = reference_targets(name, pf, cfg, now)
        if targets:
            fills += rebalance_direct(pf, targets, snap_quotes, cfg, now, cid)

    # ---- 6. AI decision ------------------------------------------------------------------------
    ai = arms["ai_pv"]
    theses = j.load_json("theses.json", {})
    feedback = j.load_json("ai_feedback.json", {"note": "first cycle, no feedback yet"})
    ai_row = {"cycle_id": cid, "ts": iso(now), "arm": "ai_pv"}
    ai_decs, result = [], None
    try:
        result = decide(llm, cfg, cid, snap, ai, theses, feedback, now)
        last = result["attempts"][-1]
        j.append("prompts.jsonl", {"cycle_id": cid, "system_sha256": result["system_sha256"],
                                   "prompt_sha256": result["prompt_sha256"],
                                   "user_prompt": result["user_prompt"], "attempts": result["attempts"]})
        ai_row.update({"ok": result["ok"], "errors": result["errors"], "model_id": last["model_id"],
                       "tokens_in": sum(a["tokens_in"] or 0 for a in result["attempts"]),
                       "tokens_out": sum(a["tokens_out"] or 0 for a in result["attempts"]),
                       "latency_s": last["latency_s"], "n_attempts": len(result["attempts"])})
        if not result["ok"]:
            events.append({"type": "AI_FAULT", "errors": result["errors"]})
    except LLMError as e:
        ai_row.update({"ok": False, "errors": [str(e)]})
        events.append({"type": "AI_FAULT", "errors": [str(e)]})

    # ---- 7. fresh quotes, then 8. constrained execution -----------------------------------------
    quotes = fresh_quotes(cfg, client, cfg["universe"], now_utc() if not dry_run else now)
    tq = arms["trend_quant"]
    tq_decs = evaluate(trend_proposals("trend_quant", tq, snap, cfg, now), tq, snap, cfg, now)
    fills += _execute_decisions(tq, tq_decs, quotes, cfg, now, cid, "baseline")
    if result and result["ok"]:
        ai_decs = evaluate(result["proposals"], ai, snap, cfg, now)
        fills += _execute_decisions(ai, ai_decs, quotes, cfg, now, cid, "ai")
        theses, hist = apply_thesis_updates(theses, result["parsed"], cid)
        for h in hist:
            j.append("theses_history.jsonl", h)
        j.save_json("theses.json", theses)
        p = result["parsed"]
        ai_row.update({"outlook": p["outlook"], "decisions": p["decisions"],
                       "thesis_updates": p["thesis_updates"], "portfolio_note": p["portfolio_note"],
                       "fence_stripped": p["_fence_stripped"]})
    ai_row["risk_decisions"] = ai_decs
    j.append("decisions.jsonl", ai_row)
    j.append("decisions.jsonl", {"cycle_id": cid, "ts": iso(now), "arm": "trend_quant", "risk_decisions": tq_decs})

    # ---- 9. persist -----------------------------------------------------------------------------
    for f in fills:
        j.append("orders.jsonl", f)
    for e in events:
        j.append("events.jsonl", {"ts": iso(now), "cycle_id": cid, **e})
    for name, pf in arms.items():
        eq = pf.equity(mids)
        j.append("equity.jsonl", {"cycle_id": cid, "ts": iso(now), "arm": name, "equity": eq,
                                  "cash": pf.cash, "gross_exposure": 1 - pf.cash / eq if eq > 0 else 0,
                                  "drawdown": pf.drawdown(mids), "peak": pf.peak_equity,
                                  "fees_usd": pf.fees_usd, "spread_usd": pf.spread_usd,
                                  "slippage_usd": pf.slippage_usd, "traded_usd": pf.traded_notional_usd,
                                  "n_positions": len(pf.positions)})
        pf.last_ts = iso(now)
        j.save_json(f"arms/{name}.json", pf.to_dict())
    j.save_json("ai_feedback.json", {
        "from_cycle": cid,
        "risk_engine": [{k: d[k] for k in ("asset", "action", "status", "codes", "requested_weight", "final_weight")}
                        for d in ai_decs],
        "your_fills": [{k: f[k] for k in ("asset", "side", "qty", "price", "cause")} for f in fills if f["arm"] == "ai_pv"],
        "events": [e for e in events if e.get("arm") in (None, "ai_pv")],
    })
    status = "OK" if ai_row.get("ok") else "OK_AI_FAULT"
    j.append("cycles.jsonl", {"cycle_id": cid, "started_at": iso(now), "finished_at": iso(now_utc()),
                              "duration_s": round(time.time() - t0, 1), "status": status,
                              "model_id": ai_row.get("model_id"), "tokens_in": ai_row.get("tokens_in"),
                              "tokens_out": ai_row.get("tokens_out"), "n_fills": len(fills),
                              "n_events": len(events), "dry_run": dry_run})
    return {"cycle_id": cid, "status": status, "fills": len(fills), "events": events,
            "ai_errors": ai_row.get("errors")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--dry-run", action="store_true", help="mock LLM, no lock required, no alerts")
    a = ap.parse_args()
    cfg = load_config()
    llm = make_llm(cfg, "mock" if a.dry_run else None)
    out = run_cycle(a.state_dir, cfg, CoinbaseClient(), llm, dry_run=a.dry_run, require_lock=not a.dry_run)
    print(BANNER)
    print({k: v for k, v in out.items() if k != "events"})
    for e in out.get("events", []) or []:
        print("  event:", e)
    sys.exit(0 if out["status"] in ("OK", "OK_AI_FAULT", "ALREADY_DONE") else 1)
