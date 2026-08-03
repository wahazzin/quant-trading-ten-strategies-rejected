"""
reconcile.py -- the fix for four separate incidents of stale, duplicate,
or erroneous broker orders in this project:
  1. A ghost -25 short position that didn't match any known trade.
  2. 2 stale SELL bracket-order legs from old F-symbol testing, resting
     at IBKR with no corresponding local record (caught by
     ops/verify_system.py's Check 5).
  3. 60 resting orders instead of 20 -- value_rebalance.py sized every
     order from ib.positions() only, so re-running it before a GTC
     order filled recomputed the same diff and submitted a duplicate
     on top of the one already resting. Three runs across
     markets-closed periods tripled every order.
  4. 5 erroneous SELL orders on Alpaca -- a resting order stays "open"
     while PARTIALLY filled, and the pending-quantity calculation used
     the order's original requested qty instead of qty - filled_qty.
     A 4-share order filled 2-of-4 showed position=2 AND pending=4, so
     the diff math saw effective=6 against target=4 and tried to SELL
     2 shares out of a position the target still wanted held. Caught
     only by Alpaca's own wash-trade protection (error 40310000), not
     by anything in this project.

Each of the first three was fixed in its own script, one at a time,
after the fact -- which is exactly how #4 slipped through even after
the "fix it at the root" pass that produced this file: the root fix
computed pending-order exposure in one place, but that one place
(bot/broker/alpaca_client.py) still used the wrong number, and nothing
here double-checked the *result* was sane before letting it through.

So there are now two independent layers:

  1. pending_quantity() -- the ONE place "how much exposure is still
     working in open orders" gets computed, from raw broker order
     objects, so it's covered by unit tests instead of trusted to live
     correctly inside whichever broker client happens to call it.
  2. check_trade_safety() -- a hard backstop that refuses to let a
     computed trade through if it looks like the bug shape above
     (selling into a position the target wants held) or is simply
     implausibly large, REGARDLESS of what produced the bad number.
     This is deliberately redundant with #1: the whole point is that
     the next bug of this shape, whatever causes it, still gets caught
     here even if it isn't the same root cause as incident #4.

The core idea, unchanged from before: the ONLY correct source of "what
does this account currently have exposure to" is POSITION + PENDING
ORDERS, not position alone. An order resting unfilled (or partially
filled) has zero or partial effect on positions but is just as real a
commitment as a filled one.
"""


def pending_quantity(open_orders_raw):
    """
    Net signed REMAINING (unfilled) quantity of open orders, per symbol,
    computed directly from raw broker order objects.

    Uses (qty - filled_qty), NOT qty -- an order stays "open" in a
    broker's API while PARTIALLY filled, and qty is always the
    original total requested. Using raw qty double-counts the portion
    that's already filled and visible in current_positions: a 4-share
    order filled 2-of-4 has remaining=2, not 4. This is incident #4
    above -- fixed here, once, instead of inside a broker client.

    open_orders_raw: list of raw order dicts, each with at least
        'symbol', 'side' ('buy'/'sell'), 'qty', and optionally
        'filled_qty' (missing or None is treated as 0) -- the shape
        Alpaca's GET /v2/orders?status=open returns directly.
    """
    pending = {}
    for o in open_orders_raw:
        sym = o["symbol"]
        remaining = float(o["qty"]) - float(o.get("filled_qty") or 0)
        signed = remaining if o["side"] == "buy" else -remaining
        pending[sym] = pending.get(sym, 0.0) + signed
    return pending


def check_trade_safety(symbol, target_qty, held, pending, diff):
    """
    Hard safety check applied to every computed trade before it is
    allowed into the plan. Returns a list of human-readable reasons the
    trade is unsafe (empty = safe). Any reason present means the
    caller must refuse to submit it and print a warning instead --
    this must never happen a fifth time.

    Two rules:

    1. Refuse a SELL if the target still wants this position held
       (target_qty > 0) and the ACTUAL held quantity does not exceed
       the target (held <= target_qty). A trade that would sell down
       a position the target says to hold -- rather than trim an
       actual overage -- is exactly incident #4's shape, whatever
       produced the bad number this time. A legitimate trim (held >
       target_qty) is unaffected; a full drop (target_qty == 0) is
       unaffected.

    2. Refuse any trade whose size exceeds 2x the intended reference
       size (the target quantity for a new/adjusted position, or the
       current held size for a full close). Catches gross sizing bugs
       independent of direction.
    """
    reasons = []
    if diff < 0 and target_qty > 0 and held <= target_qty:
        reasons.append(
            f"refusing to SELL {abs(diff):g} {symbol}: held ({held:g}) does not exceed "
            f"target ({target_qty:g}) -- target says to hold this position, not reduce it "
            f"(pending={pending:g})"
        )

    reference = target_qty if target_qty > 0 else abs(held)
    if reference > 0 and abs(diff) > 2 * reference:
        action = "BUY" if diff > 0 else "SELL"
        reasons.append(
            f"refusing to {action} {abs(diff):g} {symbol}: exceeds 2x the intended size "
            f"(target={target_qty:g}, reference={reference:g})"
        )

    return reasons


def reconcile(symbols, target, current_positions, open_orders_raw):
    """
    Compute the net quantity to trade per symbol so that
    (current position + pending orders + this trade) == target, then
    run every nonzero result through check_trade_safety() before
    handing it back.

    symbols: the set of tickers this call is allowed to touch -- e.g.
        target.keys() unioned with whatever this strategy previously
        tracked. This is deliberately NOT "every symbol with a position
        or order at the broker" -- an account can hold positions this
        strategy never touched (old test trades, manual holdings), and
        nothing about those should be inferred as "needs fixing" just
        because it's unrecognized. Scoping is the caller's
        responsibility precisely so a bug here can't silently expand
        to touch unrelated holdings.
    target: {symbol: target_shares} -- the desired end state. A symbol
        in `symbols` but absent from `target` is treated as target 0
        (i.e. "should be fully closed").
    current_positions: {symbol: qty} -- actual filled position size,
        from the broker, not from any local cache.
    open_orders_raw: list of raw broker order dicts (see
        pending_quantity()'s docstring for the required shape) for
        every currently open/working order across ALL symbols (not
        pre-filtered or pre-aggregated) -- this function owns turning
        that into per-symbol pending exposure, so that math lives in
        exactly one place and is covered by unit tests.

    Returns (diffs, warnings):
      diffs -- {symbol: signed_diff} for every SAFE nonzero trade
          needed. Positive = BUY, negative = SELL. A symbol already
          fully reconciled, or one whose trade failed
          check_trade_safety(), is absent from this dict.
      warnings -- list of human-readable strings, one per trade that
          was computed but refused by check_trade_safety(). Callers
          must print these and must NOT submit anything for the
          symbols they mention until a human looks.
    """
    pending_orders = pending_quantity(open_orders_raw)
    diffs = {}
    warnings = []
    for sym in symbols:
        target_qty = target.get(sym, 0)
        held = current_positions.get(sym, 0)
        pending = pending_orders.get(sym, 0)
        diff = target_qty - (held + pending)
        if diff == 0:
            continue
        reasons = check_trade_safety(sym, target_qty, held, pending, diff)
        if reasons:
            warnings.extend(reasons)
            continue
        diffs[sym] = diff
    return diffs, warnings


def format_plan(diffs):
    """Human-readable BUY/SELL lines for a reconciliation result, sorted
    for stable, diffable output."""
    if not diffs:
        return ["  (nothing to trade -- fully reconciled)"]
    lines = []
    for sym in sorted(diffs):
        qty = diffs[sym]
        action = "BUY" if qty > 0 else "SELL"
        lines.append(f"  {action} {abs(qty):g} {sym}")
    return lines


def execute_plan(diffs, place_order_fn, dry_run=False):
    """
    Print the reconciled plan, then either submit it (dry_run=False) or
    stop after printing (dry_run=True) -- the plan is IDENTICAL either
    way, only whether place_order_fn actually gets called differs. This
    means dry-run output can never drift from what a real run would do.

    place_order_fn(symbol, action, qty) -> whatever the caller wants
    back (an order confirmation, a Trade object, etc.) -- this function
    stays broker-agnostic; the caller supplies the broker-specific
    submission call.

    Returns a list of (symbol, action, qty, result_or_None) -- result
    is None for every line when dry_run=True.
    """
    for line in format_plan(diffs):
        print(line)

    if dry_run:
        if diffs:
            print(f"  [DRY RUN -- {len(diffs)} order(s) above were NOT submitted]")
        return [(sym, "BUY" if qty > 0 else "SELL", abs(qty), None) for sym, qty in sorted(diffs.items())]

    results = []
    for sym in sorted(diffs):
        qty = diffs[sym]
        action = "BUY" if qty > 0 else "SELL"
        result = place_order_fn(sym, action, abs(qty))
        results.append((sym, action, abs(qty), result))
    return results
