import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ops"))
import capm2_shadow_stop as S  # noqa: E402


class TestShadow(unittest.TestCase):
    def test_mirror_copies_weights_with_own_equity(self):
        g = {"cash": 0.0, "holdings": {}}
        pos = [{"symbol": "A", "market_value": "600", "current_price": "10"}, {"symbol": "B", "market_value": "400", "current_price": "20"}]
        S.mirror(g, pos, 1000.0, 2000.0, "d", {"A": {"d": 10.0}, "B": {"d": 20.0}})
        self.assertAlmostEqual(g["holdings"]["A"]["shares"], 120.0)
        self.assertAlmostEqual(g["holdings"]["B"]["shares"], 40.0)
        self.assertLess(g["cash"], 0.01)                       # fully invested, minus tiny fees

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
            if url.endswith("/v2/account"):
                return {"equity": "1000"}
            return [{"symbol": "OTLY", "market_value": "1000", "current_price": "10"}]
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
