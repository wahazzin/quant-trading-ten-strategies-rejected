"""
baselines.py -- the deterministic arms the AI must beat. None of them look at the AI.

TWO KINDS (this distinction is the whole point of having baselines):
  * REFERENCE benchmarks (btc_hold, eth_hold, ew_basket): "what if you'd just held it?"
    They bypass the risk engine, have no stops, and are fully invested. Rebalanced directly.
  * CONSTRAINT-MATCHED challenger (trend_quant): produces Proposals that go through the SAME
    risk engine, stops, and costs as the AI. If the AI only beats the reference benchmarks but
    not trend_quant, the risk engine deserves the credit, not the AI.

trend_quant rule (fixed in PREREGISTRATION.md): every cycle, hold equal weight in every asset
whose last daily close is above its 50-day simple moving average. The target per
asset is 80% / (number eligible), so the portfolio aims at the same 80% gross cap the AI faces;
the risk engine still clips it like any other arm. Trades only when the gap to target > 5pp.
"""
from crypto_ai.execution.paper_exchange import execute
from crypto_ai.risk.engine import Proposal


def rebalance_direct(pf, targets, quotes, cfg, now, cycle_id):
    """Unconstrained rebalance to exact target weights (reference benchmarks only).
    Sells first so their cash funds the buys. Returns fill rows."""
    mids = {a: q["mid"] for a, q in quotes.items()}
    eq = pf.equity(mids)
    cur = pf.weights(mids)
    fills = []
    for a in list(pf.positions):
        tw = targets.get(a, 0.0)
        if cur.get(a, 0.0) > tw + 1e-6:
            f = execute(pf, a, "sell", (cur[a] - tw) * eq, quotes[a], cfg, now, "baseline",
                        cycle_id, full_exit=tw <= 1e-9)
            if f:
                fills.append(f)
    for a, tw in targets.items():
        diff = tw - cur.get(a, 0.0)
        if diff > 1e-6:
            f = execute(pf, a, "buy", diff * eq, quotes[a], cfg, now, "baseline", cycle_id)
            if f:
                fills.append(f)
    return fills


def reference_targets(arm, pf, cfg, now):
    """Target weights for a reference benchmark THIS cycle, or None if it should not trade."""
    kind = cfg["arms"][arm]["kind"]
    if kind == "buy_hold":
        if pf.extra.get("bought"):
            return None
        pf.extra["bought"] = True
        return {cfg["arms"][arm]["asset"]: 1.0}
    if kind == "equal_weight":
        month = now.strftime("%Y-%m")
        if pf.extra.get("last_rebalance_month") == month:
            return None
        pf.extra["last_rebalance_month"] = month
        n = len(cfg["universe"])
        # 1/n minus a hair so fees on the buys don't make the last order fail for lack of cash
        return {a: (1.0 / n) * 0.995 for a in cfg["universe"]}
    raise ValueError(f"not a reference arm: {arm}")


def trend_proposals(arm, pf, snapshot, cfg, now):
    """Proposals for trend_quant. Evaluated EVERY cycle: the signal uses daily closes, so it can
    only change once per day, and the band prevents churn. (An earlier draft evaluated only at
    00:00 UTC; the dry run showed that, combined with the rolling 24h turnover window, it would
    take ~8 days to build positions vs ~4 for the AI -- an unfair handicap on the baseline.)"""
    spec = cfg["arms"][arm]
    assets = snapshot["assets"]
    eligible = [a for a in cfg["universe"]
                if assets[a]["features"]["daily_close"] > assets[a]["features"]["sma50"]]
    target = cfg["risk"]["max_gross_exposure"] / len(eligible) if eligible else 0.0
    mids = {a: v["mid"] for a, v in assets.items()}
    cur = pf.weights(mids)
    band = spec["band"]
    props = []
    for a in cfg["universe"]:
        w = cur.get(a, 0.0)
        if a in eligible:
            if w <= 1e-4:
                props.append(Proposal(arm, a, "BUY", target, None, "baseline"))
            elif target - w > band:
                props.append(Proposal(arm, a, "ADD", target, None, "baseline"))
            elif w - target > band:
                props.append(Proposal(arm, a, "REDUCE", target, None, "baseline"))
        elif w > 1e-4:
            props.append(Proposal(arm, a, "EXIT", None, None, "baseline"))
    return props
