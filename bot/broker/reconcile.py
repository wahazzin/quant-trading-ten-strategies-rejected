"""
reconcile.py -- the fix for three separate incidents of stale or
duplicate broker orders in this project:
  1. A ghost -25 short position that didn't match any known trade.
  2. 2 stale SELL bracket-order legs from old F-symbol testing, resting
     at IBKR with no corresponding local record (caught by
     ops/verify_system.py's Check 5).
  3. 60 resting orders instead of 20 -- value_rebalance.py sized every
     order from ib.positions() only, so re-running it before a GTC
     order filled recomputed the same diff and submitted a duplicate
     on top of the one already resting. Three runs across
     markets-closed periods tripled every order.

Each of those was fixed in its own script, one at a time, after the
fact. This is the fix applied once, at the root: a single reconciliation
function every order-placing script calls before submitting anything.
No script should ever compute "what do I need to trade" on its own
again.

The core idea: the ONLY correct source of "what does this account
currently have exposure to" is POSITION + PENDING ORDERS, not position
alone. An order resting unfilled has zero effect on positions but is
just as real a commitment as a filled one.
"""


def reconcile(symbols, target, current_positions, pending_orders):
    """
    Compute the net quantity to trade per symbol so that
    (current position + pending orders + this trade) == target.

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
    pending_orders: {symbol: signed_qty} -- net signed quantity of
        OPEN/working orders per symbol (positive = net buy exposure
        working, negative = net sell exposure working), from the
        broker, not from any local cache. This is the piece that was
        missing before.

    Returns {symbol: signed_diff} for every symbol in `symbols` where
    a nonzero trade is needed. Positive = BUY, negative = SELL. A
    symbol already fully reconciled (by position, by a pending order,
    or by both together) is simply absent from the result -- there is
    nothing left to place for it.
    """
    diffs = {}
    for sym in symbols:
        target_qty = target.get(sym, 0)
        held = current_positions.get(sym, 0)
        pending = pending_orders.get(sym, 0)
        diff = target_qty - (held + pending)
        if diff != 0:
            diffs[sym] = diff
    return diffs


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
