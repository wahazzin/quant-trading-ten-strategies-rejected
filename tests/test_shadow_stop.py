import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ops"))
import capm2_shadow_stop as S  # noqa: E402


class TestShadow(unittest.TestCase):
    def test_mirror_copies_holdings_with_own_equity(self):
        g = {"cash": 0.0, "holdings": {}}
        pos = [{"symbol": "A", "qty": "60", "market_value": "600", "current_price": "10"},
               {"symbol": "B", "qty": "20", "market_value": "400", "current_price": "20"}]
        S.mirror(g, pos, "0", 1000.0, 2000.0, "d", {"A": {"d": 10.0}, "B": {"d": 20.0}})
        self.assertAlmostEqual(g["holdings"]["A"]["shares"], 120.0)
        self.assertAlmostEqual(g["holdings"]["B"]["shares"], 40.0)
        self.assertLess(abs(g["cash"]), 0.1)                   # fully invested, minus tiny fees

    def test_mirror_is_exact_copy_when_live_price_differs_from_close(self):
        # 2026-10-08 bug: start ran pre-market; live price 235.4 vs last close 237.4 -> copy held fewer shares.
        g = {"cash": 0.0, "holdings": {}}
        pos = [{"symbol": "N", "qty": "171", "market_value": str(171 * 235.4), "current_price": "235.4"}]
        eq = 171 * 235.4 + 500.0
        S.mirror(g, pos, "500", eq, eq, "d", {"N": {"d": 237.4}})
        self.assertAlmostEqual(g["holdings"]["N"]["shares"], 171.0)   # same share count as the real account
        self.assertAlmostEqual(g["cash"], 500.0 - 171 * 237.4 * S.FEE, delta=0.01)  # only the copy fee

    def test_never_writes_to_broker(self):
        src = open(S.__file__).read()
        for bad in ("requests.post", "requests.patch", "requests.delete", "/v2/orders"):
            self.assertNotIn(bad, src)


if __name__ == "__main__":
    unittest.main()


class TestStopFires(unittest.TestCase):
    def test_15pct_trailing_stop_sells_and_keeps_cash(self):
        import json, tempfile
        d = tempfile.mkdtemp(); cwd = os.getcwd(); os.chdir(d)
        os.environ.update(CAPM2_US_KEY_ID="k", CAPM2_US_SECRET_KEY="s")
        os.environ.pop("CAPM2_SE_KEY_ID", None)
        px = {"OTLY": {"2026-10-06": 10.0}}
        def fake_get(url, h, **p):
            if url.endswith("/v2/clock"):
                return {}
            if url.endswith("/v2/account/activities"):
                return []
            if url.endswith("/v2/account"):
                return {"equity": "1000", "cash": "0"}
            return [{"symbol": "OTLY", "qty": "100", "market_value": "1000", "current_price": "10"}]
        orig_get, orig_closes = S.get, S.closes
        S.get = fake_get
        S.closes = lambda h, syms, start: {s: dict(px.get(s, {})) for s in syms}
        try:
            from datetime import date
            S.run(date(2026, 10, 6))                                   # start: mirror real account
            px["OTLY"].update({"2026-10-07": 12.0, "2026-10-08": 10.1})   # peak 12, then -15.8%
            S.run(date(2026, 10, 8))
            st = json.load(open("status/shadow_stop.json"))
            g = st["groups"]["US"]
            self.assertEqual(g["holdings"], {})
            self.assertTrue(any(x["event"] == "STOP" and x["symbol"] == "OTLY" for x in st["log"]))
            self.assertAlmostEqual(g["cash"], 100 * 10.1 * (1 - S.FEE), delta=0.5)
        finally:
            S.get, S.closes = orig_get, orig_closes
            os.chdir(cwd)


class TestLateRebalanceAndPartialBar(unittest.TestCase):
    """2026-10-26 fixes: (1) a day whose session is still open is not processed; (2) the weekly re-copy waits for
    CAPM2's real rebalance (here: Tuesday, because GitHub started Monday's job after the close) and copies the
    holdings as of that close, even when several days are processed in one later run."""

    def test_session_open_day(self):
        self.assertEqual(S.session_open_day({"timestamp": "2026-10-19T09:37:00-04:00",
                                             "next_close": "2026-10-19T16:00:00-04:00"}), "2026-10-19")
        self.assertIsNone(S.session_open_day({"timestamp": "2026-10-19T20:05:00-04:00",
                                              "next_close": "2026-10-20T16:00:00-04:00"}))
        self.assertIsNone(S.session_open_day(None))

    def test_real_as_of_undoes_later_fills(self):
        pos = [{"symbol": "B", "qty": "10", "market_value": "200", "current_price": "20"}]
        fills = [{"transaction_time": "2026-10-20T19:01:37Z", "symbol": "A", "side": "sell", "qty": "5", "price": "11"},
                 {"transaction_time": "2026-10-20T19:01:43Z", "symbol": "B", "side": "buy", "qty": "10", "price": "19"}]
        prices = {"A": {"2026-10-19": 10.0, "2026-10-20": 11.0}, "B": {"2026-10-19": 18.0, "2026-10-20": 20.0}}
        p, c, e = S.real_as_of("2026-10-19", pos, "100", fills, prices)
        self.assertEqual({x["symbol"]: float(x["qty"]) for x in p}, {"A": 5.0})
        self.assertAlmostEqual(c, 100 - 55 + 190)
        self.assertAlmostEqual(e, 235 + 50)
        p, c, e = S.real_as_of("2026-10-20", pos, "100", fills, prices)
        self.assertEqual({x["symbol"]: float(x["qty"]) for x in p}, {"B": 10.0})
        self.assertAlmostEqual(e, 300.0)

    def test_tuesday_rebalance_is_copied_on_tuesday(self):
        import json, tempfile
        from datetime import date
        d = tempfile.mkdtemp(); cwd = os.getcwd(); os.chdir(d)
        os.environ.update(CAPM2_US_KEY_ID="k", CAPM2_US_SECRET_KEY="s")
        os.environ.pop("CAPM2_SE_KEY_ID", None)
        os.makedirs("status")
        # state after Friday 10-16: shadow holds 10 A (copy of last week's real account), no cash
        json.dump({"start": "2026-10-07", "log": [], "groups": {"US": {
            "cash": 0.0, "holdings": {"A": {"shares": 10.0, "peak": 10.0, "entry": 10.0}},
            "last_day": "2026-10-16", "last_sync": "2026-10-12", "history": []}}}, open(S.STATE, "w"))
        px = {"A": {"2026-10-19": 10.0, "2026-10-20": 10.0, "2026-10-21": 10.0},
              "B": {"2026-10-19": 20.0, "2026-10-20": 20.0, "2026-10-21": 22.0}}
        fills = [{"id": "1", "transaction_time": "2026-10-20T19:01:37Z", "symbol": "A", "side": "sell", "qty": "10", "price": "10"},
                 {"id": "2", "transaction_time": "2026-10-20T19:01:43Z", "symbol": "B", "side": "buy", "qty": "5", "price": "20"}]
        clock = {"timestamp": "2026-10-22T09:40:00-04:00", "next_close": "2026-10-22T16:00:00-04:00"}

        def fake_get(url, h, **p):
            if url.endswith("/v2/clock"):
                return clock
            if url.endswith("/v2/account/activities"):
                return fills
            if url.endswith("/v2/account"):
                return {"equity": "110", "cash": "0"}
            return [{"symbol": "B", "qty": "5", "market_value": "110", "current_price": "22"}]
        orig_get, orig_closes = S.get, S.closes
        S.get = fake_get
        S.closes = lambda h, syms, start: {s: {k: v for k, v in px.get(s, {}).items()} for s in syms}
        px["B"]["2026-10-22"] = 21.0                                  # partial bar of a still-open session
        try:
            S.run(date(2026, 10, 22))
            st = json.load(open(S.STATE))
            g = st["groups"]["US"]
            syncs = [x for x in st["log"] if x["event"] == "WEEKLY_SYNC"]
            self.assertEqual([x["day"] for x in syncs], ["2026-10-20"])   # not Monday 10-19
            self.assertEqual(list(g["holdings"]), ["B"])
            self.assertAlmostEqual(g["holdings"]["B"]["shares"], 5.0, places=6)
            self.assertEqual(g["last_day"], "2026-10-21")                 # 10-22 not booked (session open)
            self.assertAlmostEqual(g["history"][-1]["shadow"], 110.0, delta=0.05)
        finally:
            S.get, S.closes = orig_get, orig_closes
            os.chdir(cwd)
