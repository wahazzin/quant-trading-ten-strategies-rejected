"""
schemas.py -- validates the LLM's raw text before anything is allowed to act on it.

CONCEPT: an LLM is a text generator. It can return invalid JSON, invent assets, skip required
fields, or contradict itself. This file is the border control between "text the model wrote"
and "actions the system takes". It never guesses or repairs meaning; the only tolerance is
stripping a markdown code fence, and even that is recorded (`fence_stripped`).

validate() returns (parsed_or_None, errors). Any error => the whole response is rejected, the
runner retries once, then falls back to DO_NOTHING and logs AI_FAULT.
"""
import json

ACTIONS = {"BUY", "ADD", "HOLD", "REDUCE", "SELL", "EXIT", "DO_NOTHING"}
NEEDS_WEIGHT = {"BUY", "ADD", "REDUCE"}
NEEDS_THESIS = {"BUY", "ADD"}
THESIS_STATUS = {"bullish", "bearish", "neutral", "invalidated", "closed"}


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _str_list(x):
    return isinstance(x, list) and len(x) > 0 and all(isinstance(s, str) and s.strip() for s in x)


def validate(raw, cfg, cycle_id):
    errors = []
    text = (raw or "").strip()
    fence_stripped = False
    if text.startswith("```"):
        fence_stripped = True
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0].strip()
    try:
        obj = json.loads(text)
    except Exception as e:
        return None, [f"invalid JSON: {e}"]
    if not isinstance(obj, dict):
        return None, ["top level is not an object"]

    uni = set(cfg["universe"])
    if obj.get("cycle_id") != cycle_id:
        errors.append(f"cycle_id mismatch: {obj.get('cycle_id')!r} != {cycle_id!r}")

    outlook = obj.get("outlook")
    if not isinstance(outlook, dict):
        errors.append("outlook missing or not an object")
    else:
        missing = uni - set(outlook)
        extra = set(outlook) - uni
        if missing:
            errors.append(f"outlook missing assets: {sorted(missing)}")
        if extra:
            errors.append(f"outlook has unknown assets: {sorted(extra)}")
        for a, v in outlook.items():
            if a in uni and not (isinstance(v, int) and not isinstance(v, bool) and -2 <= v <= 2):
                errors.append(f"outlook[{a}] must be an integer -2..2, got {v!r}")

    decisions = obj.get("decisions", [])
    if not isinstance(decisions, list):
        errors.append("decisions is not a list")
        decisions = []
    seen = set()
    for i, d in enumerate(decisions):
        tag = f"decisions[{i}]"
        if not isinstance(d, dict):
            errors.append(f"{tag} not an object"); continue
        a, act = d.get("asset"), d.get("action")
        if a not in uni:
            errors.append(f"{tag} unknown asset {a!r}")
        if a in seen:
            errors.append(f"{tag} duplicate decision for {a}")
        seen.add(a)
        if act not in ACTIONS:
            errors.append(f"{tag} invalid action {act!r}"); continue
        if act in NEEDS_WEIGHT:
            tw = d.get("target_weight")
            if not (_num(tw) and 0 <= tw <= 1):
                errors.append(f"{tag} {act} needs target_weight in [0,1], got {tw!r}")
        if act in NEEDS_THESIS:
            if not _str_list(d.get("reasons")):
                errors.append(f"{tag} {act} needs non-empty reasons")
            if not _str_list(d.get("invalidation")):
                errors.append(f"{tag} {act} needs non-empty invalidation")
        sl = d.get("stop_loss_pct")
        if sl is not None and not (_num(sl) and sl > 0):
            errors.append(f"{tag} stop_loss_pct must be a positive number")
        c = d.get("confidence")
        if c is not None and not (_num(c) and 0 <= c <= 100):
            errors.append(f"{tag} confidence must be 0..100")

    updates = obj.get("thesis_updates", [])
    if not isinstance(updates, list):
        errors.append("thesis_updates is not a list")
        updates = []
    for i, u in enumerate(updates):
        if not isinstance(u, dict) or u.get("asset") not in uni or u.get("status") not in THESIS_STATUS:
            errors.append(f"thesis_updates[{i}] needs a known asset and status in {sorted(THESIS_STATUS)}")

    if errors:
        return None, errors
    obj.setdefault("decisions", [])
    obj.setdefault("thesis_updates", [])
    obj.setdefault("portfolio_note", "")
    obj["_fence_stripped"] = fence_stripped
    return obj, []
