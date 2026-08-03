"""
test_reconcile.py -- unit tests for bot/broker/reconcile.py, added after
the 4th reconciliation incident (partial fills broke the pending-quantity
math and produced 5 erroneous SELL orders, caught only by Alpaca's own
wash-trade protection). See reconcile.py's module docstring for the full
incident history.

Run with:  python -m unittest tests.test_reconcile
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.broker.reconcile import reconcile, pending_quantity, check_trade_safety


def order(symbol, side, qty, filled_qty=0):
    return {"symbol": symbol, "side": side, "qty": str(qty), "filled_qty": str(filled_qty)}


class TestPendingQuantity(unittest.TestCase):

    def test_no_orders(self):
        self.assertEqual(pending_quantity([]), {})

    def test_unfilled_buy(self):
        self.assertEqual(pending_quantity([order("AAA", "buy", 10)]), {"AAA": 10.0})

    def test_partial_fill_uses_remaining_not_original_qty(self):
        # This is incident #4: a 4-share order filled 2-of-4 must contribute
        # remaining=2 to pending, not the original qty=4.
        self.assertEqual(pending_quantity([order("AAA", "buy", 4, filled_qty=2)]), {"AAA": 2.0})

    def test_sell_side_is_negative(self):
        self.assertEqual(pending_quantity([order("AAA", "sell", 3)]), {"AAA": -3.0})

    def test_multiple_orders_same_symbol_net_together(self):
        orders = [order("AAA", "buy", 5), order("AAA", "sell", 2, filled_qty=1)]
        # buy remaining=5, sell remaining=1 (signed -1) -> net 4
        self.assertEqual(pending_quantity(orders), {"AAA": 4.0})


class TestReconcile(unittest.TestCase):
    """The four required cases: no position + no orders, partial fill,
    over-target, and exact-target."""

    def test_no_position_no_orders(self):
        diffs, warnings = reconcile({"AAA"}, {"AAA": 10}, {}, [])
        self.assertEqual(diffs, {"AAA": 10})
        self.assertEqual(warnings, [])

    def test_partial_fill_already_reconciled(self):
        # Filled 2 of a 4-share target; the remaining 2 shares are still
        # resting as an open order (qty=4, filled_qty=2 -> remaining=2).
        # Position(2) + pending(2) == target(4): nothing left to trade.
        diffs, warnings = reconcile(
            {"AAA"}, {"AAA": 4}, {"AAA": 2}, [order("AAA", "buy", 4, filled_qty=2)]
        )
        self.assertEqual(diffs, {})
        self.assertEqual(warnings, [])

    def test_over_target_trims_down(self):
        diffs, warnings = reconcile({"AAA"}, {"AAA": 4}, {"AAA": 6}, [])
        self.assertEqual(diffs, {"AAA": -2})
        self.assertEqual(warnings, [])

    def test_exact_target_no_trade(self):
        diffs, warnings = reconcile({"AAA"}, {"AAA": 4}, {"AAA": 4}, [])
        self.assertEqual(diffs, {})
        self.assertEqual(warnings, [])

    def test_drop_symbol_not_in_target_sells_all(self):
        diffs, warnings = reconcile({"AAA"}, {}, {"AAA": 10}, [])
        self.assertEqual(diffs, {"AAA": -10})
        self.assertEqual(warnings, [])

    def test_stray_resting_buy_refused_not_sold_down(self):
        # target=4, held=2 (below target), but a resting BUY order for 5 more
        # shares (unrelated/stray/duplicate, filled_qty=0) pushes effective
        # exposure to 7. The naive diff (4 - 7 = -3) would SELL 2 shares out
        # of a position the target still wants held -- this must be refused,
        # not submitted, even though the pending math itself is correct here.
        diffs, warnings = reconcile(
            {"AAA"}, {"AAA": 4}, {"AAA": 2}, [order("AAA", "buy", 5, filled_qty=0)]
        )
        self.assertEqual(diffs, {})
        self.assertEqual(len(warnings), 1)
        self.assertIn("AAA", warnings[0])


class TestCheckTradeSafety(unittest.TestCase):

    def test_incident_shape_refused(self):
        # target=4, held=2 (does not exceed target), diff=-2 SELL -- the
        # exact shape of incident #4.
        reasons = check_trade_safety("AAA", target_qty=4, held=2, pending=4, diff=-2)
        self.assertTrue(reasons)

    def test_legit_trim_allowed(self):
        # held(6) exceeds target(4) -- selling down to target is expected.
        reasons = check_trade_safety("AAA", target_qty=4, held=6, pending=0, diff=-2)
        self.assertEqual(reasons, [])

    def test_full_drop_allowed(self):
        # target=0 -- a full close is always allowed regardless of held size.
        reasons = check_trade_safety("AAA", target_qty=0, held=10, pending=0, diff=-10)
        self.assertEqual(reasons, [])

    def test_oversized_buy_refused(self):
        # target=4 but diff=10 is more than 2x -- refuse regardless of direction.
        reasons = check_trade_safety("AAA", target_qty=4, held=0, pending=0, diff=10)
        self.assertTrue(reasons)

    def test_oversized_sell_on_close_refused(self):
        # target=0, held=3, but diff=-10 (more than 2x abs(held)) -- refuse.
        reasons = check_trade_safety("AAA", target_qty=0, held=3, pending=0, diff=-10)
        self.assertTrue(reasons)

    def test_normal_new_entry_allowed(self):
        reasons = check_trade_safety("AAA", target_qty=10, held=0, pending=0, diff=10)
        self.assertEqual(reasons, [])


if __name__ == "__main__":
    unittest.main()
