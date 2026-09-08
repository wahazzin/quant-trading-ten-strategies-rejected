"""
alpaca_execution.py -- places bracket (entry + stop-loss + take-profit)
orders on Alpaca using their native order_class="bracket" support: one
API call creates all three legs atomically, unlike the old IBKR path
(execution.py, retired) which needed three separate manually-linked
orders and was the root cause of several of the original broker-
reconciliation bugs (stale legs, duplicate orders).

STOP/TARGET RULE IS AN EXPLICIT, UNTESTED PLACEHOLDER (see
DEFAULT_STOP_PCT / DEFAULT_TARGET_PCT below). No backtest has validated
any specific stop-loss or take-profit distance for this project -- the
19 rejected tests and the value_bm forward test were never about exit
timing. This module exists to test EXECUTION MECHANICS (does a bracket
order actually get placed, filled, and correctly tracked end-to-end)
with real (paper) broker behavior, separate from and prior to any claim
that this specific stop/target distance has edge. Do not treat a
profitable paper-trading run using this rule as evidence the RULE works
-- it would only be evidence the SIGNAL (value_bm) plus this arbitrary
exit rule worked together, which is not a claim this project is ready to
make.
"""
from bot.broker.alpaca_client import AlpacaClient
from bot.risk.position_sizing import calculate_position_size
from bot.monitor.discord_notify import notify_fill
import time
from datetime import datetime

FILL_WAIT_SECONDS = 8   # same pattern as ops/value_rebalance.py
FILL_POLL_INTERVAL = 2

DEFAULT_STOP_PCT = 0.05     # 5% below entry -- placeholder, not backtested
DEFAULT_TARGET_PCT = 0.10   # 10% above entry -- placeholder, not backtested


def compute_stop_target(entry_price, stop_pct=DEFAULT_STOP_PCT, target_pct=DEFAULT_TARGET_PCT):
    """Pure function, no network/broker calls -- easy to unit test on its own."""
    if entry_price <= 0:
        raise ValueError("entry_price must be positive")
    stop_price = round(entry_price * (1 - stop_pct), 2)
    target_price = round(entry_price * (1 + target_pct), 2)
    return stop_price, target_price


class AlpacaExecutor:
    def __init__(self, client: AlpacaClient, risk_manager, journal=None):
        self.client = client
        self.risk_manager = risk_manager
        self.journal = journal

    def place_bracket_order(
        self,
        symbol,
        equity,
        entry_price=None,
        stop_pct=DEFAULT_STOP_PCT,
        target_pct=DEFAULT_TARGET_PCT,
        risk_per_trade_pct=0.015,
        tif="gtc",
    ):
        """
        entry_price: if None, fetched live via client.get_last_price().
        Passing it explicitly is mainly for testing without hitting the
        market-data endpoint.
        """
        if not self.risk_manager.check_trading_allowed(equity):
            print(f"  [execution] BLOCKED: risk manager denies trading for {symbol}.")
            return None

        existing = self.client.get_position(symbol)
        if existing != 0:
            print(f"  [execution] BLOCKED: already holding {existing} shares of {symbol}.")
            return None

        if entry_price is None:
            entry_price = self.client.get_last_price(symbol)

        stop_price, target_price = compute_stop_target(entry_price, stop_pct, target_pct)

        qty = calculate_position_size(
            equity=equity,
            entry_price=entry_price,
            stop_price=stop_price,
            risk_per_trade_pct=risk_per_trade_pct,
        )
        if qty <= 0:
            print(f"  [execution] BLOCKED: calculated position size is 0 for {symbol}.")
            return None

        payload = {
            "symbol": symbol,
            "qty": str(int(qty)),
            "side": "buy",
            "type": "market",
            "time_in_force": tif,
            "order_class": "bracket",
            "take_profit": {"limit_price": str(target_price)},
            "stop_loss": {"stop_price": str(stop_price)},
        }

        import requests
        r = requests.post(
            f"{self.client.base_url}/v2/orders",
            headers=self.client._headers,
            json=payload,
            timeout=15,
        )
        if r.status_code >= 400:
            print(f"  [execution] ORDER REJECTED for {symbol}: {r.status_code} {r.text}")
            return {"error": True, "status_code": r.status_code, "message": r.text}

        order = r.json()
        order_id = order.get("id")
        print(
            f"  [execution] Bracket order submitted: {symbol} qty={qty} "
            f"entry~{entry_price:.2f} stop={stop_price} target={target_price} "
            f"(order id {order_id})"
        )

        # Poll for a real fill, same pattern as ops/value_rebalance.py --
        # notify_fill and journaling only happen on CONFIRMED fill, never
        # on mere order acceptance. If the wait window expires (e.g.
        # market closed), the order is still live -- it just isn't
        # confirmed filled yet, so nothing gets journaled or announced
        # prematurely. A separate catch-up pass (like value_rebalance.py's)
        # is what would confirm it later; this function does not loop
        # forever waiting.
        waited = 0
        status_resp = order
        while waited < FILL_WAIT_SECONDS:
            time.sleep(FILL_POLL_INTERVAL)
            waited += FILL_POLL_INTERVAL
            status_resp = self.client.get_order(order_id)
            if status_resp["status"] == "filled":
                filled_qty = float(status_resp["filled_qty"])
                filled_price = float(status_resp["filled_avg_price"])
                fill_time = datetime.fromisoformat(status_resp["filled_at"].replace("Z", "+00:00"))
                if self.journal is not None:
                    self.journal.record_entry_fill(
                        symbol=symbol, shares=filled_qty, price=filled_price, fill_time=fill_time,
                    )
                notify_fill(symbol, "buy", int(filled_qty), filled_price)
                print(f"  [execution] FILLED: {filled_qty:g} {symbol} @ {filled_price:.2f}")
                return {
                    "symbol": symbol, "quantity": filled_qty, "entry": filled_price,
                    "stop": stop_price, "target": target_price,
                    "order_id": order_id, "status": "filled", "raw_response": status_resp,
                }
            if status_resp["status"] in ("canceled", "expired", "rejected"):
                print(f"  [execution] {status_resp['status'].upper()}: {symbol}")
                return {"symbol": symbol, "order_id": order_id, "status": status_resp["status"]}

        print(f"  [execution] PENDING: {symbol} not yet filled within {FILL_WAIT_SECONDS}s "
              f"(status={status_resp.get('status')}) -- no notification sent, not yet journaled")
        return {
            "symbol": symbol, "quantity": qty, "entry": entry_price,
            "stop": stop_price, "target": target_price,
            "order_id": order_id, "status": "pending", "raw_response": status_resp,
        }


def catch_up_fills(client: AlpacaClient, journal, symbols=None):
    """
    Journals and Discord-notifies any fill that completed AFTER
    place_bracket_order's short polling window gave up -- the exact gap
    left by that function's docstring. Same pattern as
    ops/value_rebalance.py's catch-up pass, generalized for reuse.

    Call this at the START of any script/cycle that runs periodically
    (a scheduled orchestrator run, a manual status check, etc.) -- NOT
    from inside place_bracket_order itself, since a pending order might
    still fill hours or days later, potentially across many separate
    script runs (e.g. a GTC order placed while markets are closed).

    symbols: optional set/list to restrict which tickers to check.
    Defaults to every symbol the broker currently shows a nonzero
    position in.

    A fill already reflected in journal.get_open_position() (quantity
    matches within a small tolerance) is treated as already handled --
    never double-journaled, never double-notified.
    """
    current_positions = client.get_positions()
    check_symbols = symbols if symbols is not None else current_positions.keys()

    caught_up = []
    for sym in sorted(check_symbols):
        held = current_positions.get(sym, 0)
        if held == 0:
            continue

        existing = journal.get_open_position(sym)
        already_journaled_qty = existing["quantity"] if existing is not None else 0.0
        if existing is not None and abs(already_journaled_qty - held) < 0.5:
            continue  # already correctly journaled, nothing new here

        candidates = client.get_closed_orders(sym) + client.get_open_orders_raw(sym)
        filled = [o for o in candidates if o.get("filled_qty") and float(o["filled_qty"]) > 0]
        if not filled:
            print(f"  [catch_up] {sym}: broker shows {held} shares but no filled "
                  f"order found -- skipping (manual check needed)")
            continue

        latest = max(filled, key=lambda o: o["filled_at"])
        fill_time = datetime.fromisoformat(latest["filled_at"].replace("Z", "+00:00"))
        new_qty = float(latest["filled_qty"]) - already_journaled_qty
        if new_qty <= 0:
            continue

        filled_price = float(latest["filled_avg_price"])
        journal.record_entry_fill(symbol=sym, shares=new_qty, price=filled_price, fill_time=fill_time)
        notify_fill(sym, "buy", int(new_qty), filled_price)
        print(f"  [catch_up] {sym}: journaled + notified -- {new_qty:g} shares @ "
              f"{filled_price:.2f} (order {latest['id']}, previously unconfirmed)")
        caught_up.append(sym)

    if not caught_up:
        print("  [catch_up] nothing to catch up -- all broker positions already journaled")
    return caught_up
