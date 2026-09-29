"""
test_crypto_ai_core.py -- unit tests for the parts where a silent bug would invalidate the
experiment: paper exchange, risk engine, breakers, portfolio accounting, spec lock.

Run with:  python -m unittest tests.test_crypto_ai_core -v
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_ai import lock as lockmod
from crypto_ai.execution.paper_exchange import execute, replay_stops
from crypto_ai.journal import iso
from crypto_ai.lock import load_config
from crypto_ai.portfolio.state import Portfolio, Position
from crypto_ai.risk.engine import Proposal, evaluate, update_breakers

CFG = load_config()
# Rule-isolation config: turnover cap effectively off, so position/exposure caps can be tested
# on their own. Turnover behaviour (R6) is tested separately against the REAL config.
CFG_LOOSE = json.loads(json.dumps(CFG))
CFG_LOOSE["risk"]["max_turnover_24h"] = 100.0
NOW = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
UNI = CFG["universe"]


def snap(prices=None, spread_bps=1.0, vol=1e9):
    prices = prices or {a: 100.0 for a in UNI}
    return {"assets": {a: {"mid": p, "bid": p * (1 - spread_bps / 2e4), "ask": p * (1 + spread_bps / 2e4),
                           "spread_bps": spread_bps, "volume_24h_usd": vol, "features": {}}
                       for a, p in prices.items()}}


def quote(s, a):
    v = s["assets"][a]
    return {"mid": v["mid"], "bid": v["bid"], "ask": v["ask"], "volume_24h_usd": v["volume_24h_usd"]}


def prop(asset, action, tw=None, stop=None):
    return Proposal("t", asset, action, tw, stop)


def approve_and_fill(pf, s, proposals, cfg=CFG_LOOSE):
    decs = evaluate(proposals, pf, s, cfg, NOW)
    for d in decs:
        if d["status"] in ("APPROVED", "CLIPPED"):
            execute(pf, d["asset"], d["side"], d["notional_usd"], quote(s, d["asset"]), CFG, NOW,
                    "test", stop_pct=d["stop_pct"], full_exit=d["full_exit"])
    return decs


class TestExchange(unittest.TestCase):

    def test_buy_pays_spread_fee_slippage(self):
        s, pf = snap(spread_bps=10), Portfolio("t", 10000)
        f = execute(pf, "BTC-USD", "buy", 1000, quote(s, "BTC-USD"), CFG, NOW, "test")
        self.assertGreater(f["price"], s["assets"]["BTC-USD"]["ask"])     # slippage on top of ask
        self.assertGreater(f["fee_usd"], 0)
        self.assertGreater(f["spread_cost_usd"], 0)
        self.assertLess(pf.equity({a: 100.0 for a in UNI}), 10000)         # costs are real, immediately

    def test_round_trip_always_loses_money_at_flat_prices(self):
        s, pf = snap(), Portfolio("t", 10000)
        execute(pf, "ETH-USD", "buy", 5000, quote(s, "ETH-USD"), CFG, NOW, "test")
        execute(pf, "ETH-USD", "sell", 0, quote(s, "ETH-USD"), CFG, NOW, "test", full_exit=True)
        self.assertLess(pf.cash, 10000)
        self.assertEqual(pf.positions, {})

    def test_buy_scaled_down_to_cash_never_overdrawn(self):
        s, pf = snap(), Portfolio("t", 1000)
        execute(pf, "BTC-USD", "buy", 5000, quote(s, "BTC-USD"), CFG, NOW, "test")
        self.assertGreaterEqual(pf.cash, -1e-9)
        self.assertLess(pf.cash, 5)

    def test_sell_more_than_held_is_capped(self):
        s, pf = snap(), Portfolio("t", 10000)
        execute(pf, "BTC-USD", "buy", 1000, quote(s, "BTC-USD"), CFG, NOW, "test")
        f = execute(pf, "BTC-USD", "sell", 999999, quote(s, "BTC-USD"), CFG, NOW, "test")
        self.assertEqual(pf.positions, {})
        self.assertAlmostEqual(f["qty"], 10.0, delta=0.2)

    def test_cannot_sell_what_you_dont_hold(self):
        s, pf = snap(), Portfolio("t", 10000)
        self.assertIsNone(execute(pf, "BTC-USD", "sell", 100, quote(s, "BTC-USD"), CFG, NOW, "test"))

    def test_stop_only_tightens(self):
        s, pf = snap(), Portfolio("t", 10000)
        execute(pf, "BTC-USD", "buy", 500, quote(s, "BTC-USD"), CFG, NOW, "test", stop_pct=10)
        execute(pf, "BTC-USD", "buy", 500, quote(s, "BTC-USD"), CFG, NOW, "test", stop_pct=18)
        self.assertEqual(pf.positions["BTC-USD"].stop_pct, 10)
        execute(pf, "BTC-USD", "buy", 500, quote(s, "BTC-USD"), CFG, NOW, "test", stop_pct=6)
        self.assertEqual(pf.positions["BTC-USD"].stop_pct, 6)

    def _held(self, opened):
        pf = Portfolio("t", 0.0)
        pf.positions["BTC-USD"] = Position(1.0, 100.0, 10.0, iso(opened))   # stop = 90
        return pf

    def _candle(self, start, o, l):
        return {"t": int(start.timestamp()), "open": o, "high": max(o, 100), "low": l, "close": o, "volume": 1}

    def test_stop_gap_fills_at_open_not_stop(self):
        opened = NOW - timedelta(hours=5)
        pf = self._held(opened)
        c = self._candle(NOW - timedelta(hours=2), o=80.0, l=78.0)          # gapped through 90
        fills = replay_stops(pf, {"BTC-USD": [c]}, snap()["assets"], CFG, NOW)
        self.assertEqual(len(fills), 1)
        self.assertLess(fills[0]["price"], 80.0)                            # worse than the open
        self.assertEqual(pf.positions, {})

    def test_stop_touch_fills_near_stop_price(self):
        pf = self._held(NOW - timedelta(hours=5))
        c = self._candle(NOW - timedelta(hours=2), o=95.0, l=89.0)
        fills = replay_stops(pf, {"BTC-USD": [c]}, snap()["assets"], CFG, NOW)
        self.assertAlmostEqual(fills[0]["price"], 90.0, delta=0.1)

    def test_stop_ignores_candles_before_position_existed(self):
        pf = self._held(NOW - timedelta(hours=1))
        c = self._candle(NOW - timedelta(hours=3), o=50.0, l=40.0)          # crash BEFORE we owned it
        self.assertEqual(replay_stops(pf, {"BTC-USD": [c]}, snap()["assets"], CFG, NOW), [])
        self.assertIn("BTC-USD", pf.positions)

    def test_stop_not_hit_no_fill(self):
        pf = self._held(NOW - timedelta(hours=5))
        c = self._candle(NOW - timedelta(hours=2), o=99.0, l=91.0)
        self.assertEqual(replay_stops(pf, {"BTC-USD": [c]}, snap()["assets"], CFG, NOW), [])


class TestRiskEngine(unittest.TestCase):

    def setUp(self):
        self.s, self.pf = snap(), Portfolio("t", 10000)

    def one(self, proposals, cfg=CFG_LOOSE):
        return evaluate(proposals, self.pf, self.s, cfg, NOW)

    def test_loss_at_stop_cap_binds_before_asset_cap(self):
        d = self.one([prop("BTC-USD", "BUY", 0.9)])[0]      # default stop 18% -> 5/18 = 27.8% < 35%
        self.assertEqual(d["status"], "CLIPPED")
        self.assertIn("R2_LOSS_AT_STOP_CAP", d["codes"])
        self.assertAlmostEqual(d["final_weight"], 5 / 18, places=6)

    def test_tighter_stop_allows_bigger_position_up_to_asset_cap(self):
        d = self.one([prop("BTC-USD", "BUY", 0.9, stop=10)])[0]
        self.assertIn("R1_ASSET_CAP", d["codes"])
        self.assertAlmostEqual(d["final_weight"], 0.35, places=6)

    def test_alt_asset_cap(self):
        d = self.one([prop("SOL-USD", "BUY", 0.5)])[0]
        self.assertAlmostEqual(d["final_weight"], 0.15, places=6)

    def test_stop_looser_than_max_is_clipped_and_flagged(self):
        d = self.one([prop("BTC-USD", "BUY", 0.1, stop=50)])[0]
        self.assertEqual(d["stop_pct"], CFG["risk"]["max_stop_pct"])
        self.assertIn("R5_STOP_ADJUSTED", d["codes"])

    def test_alt_cluster_cap(self):
        alts = ["SOL-USD", "XRP-USD", "ADA-USD", "LINK-USD"]
        decs = self.one([prop(a, "BUY", 0.15) for a in alts])
        finals = [d["final_weight"] for d in decs]
        self.assertAlmostEqual(sum(f for f in finals if f), 0.40, places=6)   # never above 40% total
        self.assertTrue(any("R4_ALT_CLUSTER" in d["codes"] for d in decs))

    def test_gross_exposure_cap_never_exceeded(self):
        decs = self.one([prop("BTC-USD", "BUY", 0.35, 10), prop("ETH-USD", "BUY", 0.35, 10),
                         prop("SOL-USD", "BUY", 0.15), prop("XRP-USD", "BUY", 0.15)])
        total = sum(d["final_weight"] for d in decs if d["final_weight"])
        self.assertLessEqual(total, 0.80 + 1e-9)

    def test_turnover_limit_clips_and_then_rejects(self):
        self.pf.record_trade(NOW - timedelta(hours=1), 2400.0, "test")       # 24% of equity already traded
        d = self.one([prop("BTC-USD", "BUY", 0.27)], cfg=CFG)[0]
        self.assertEqual(d["status"], "CLIPPED")
        self.assertIn("R6_TURNOVER", d["codes"])
        self.assertAlmostEqual(d["notional_usd"], 100.0, delta=0.01)
        self.pf.record_trade(NOW - timedelta(hours=1), 200.0, "test")
        d = self.one([prop("ETH-USD", "BUY", 0.2)], cfg=CFG)[0]
        self.assertEqual(d["status"], "REJECTED")

    def test_real_config_first_day_deployment_is_capped_at_25pct(self):
        # Documents a real, intended consequence: the AI can deploy at most 25% of equity per
        # 24h, so building a full 80% portfolio takes ~4 days. Benchmarks don't have this ramp.
        decs = self.one([prop("BTC-USD", "BUY", 0.35, 10), prop("ETH-USD", "BUY", 0.35, 10)], cfg=CFG)
        self.assertAlmostEqual(sum(d["notional_usd"] for d in decs), 2500.0, delta=0.01)

    def test_old_trades_dont_count_toward_turnover(self):
        self.pf.record_trade(NOW - timedelta(hours=30), 9000.0, "test")
        d = self.one([prop("BTC-USD", "BUY", 0.2)], cfg=CFG)[0]
        self.assertEqual(d["status"], "APPROVED")

    def test_min_order_rejected(self):
        d = self.one([prop("BTC-USD", "BUY", 0.001)])[0]                     # $10
        self.assertEqual(d["status"], "REJECTED")
        self.assertIn("R7_MIN_ORDER", d["codes"])

    def test_wide_spread_rejects_buy(self):
        self.s = snap(spread_bps=80)
        d = self.one([prop("BTC-USD", "BUY", 0.2)])[0]
        self.assertIn("R8_SPREAD_TOO_WIDE", d["codes"])

    def test_liquidity_cap(self):
        self.s = snap(vol=1_000_000)                                         # cap = 0.5% = $5,000...
        self.s = snap(vol=100_000)                                           # ...cap = $500
        d = self.one([prop("BTC-USD", "BUY", 0.25, stop=10)])[0]
        self.assertIn("R8_LIQUIDITY", d["codes"])
        self.assertAlmostEqual(d["notional_usd"], 500.0, delta=0.01)

    def test_action_direction_mismatches_rejected(self):
        d = self.one([prop("BTC-USD", "ADD", 0.2)])[0]
        self.assertIn("R12_ADD_NOT_HELD_USE_BUY", d["codes"])
        approve_and_fill(self.pf, self.s, [prop("BTC-USD", "BUY", 0.2)])
        d = self.one([prop("BTC-USD", "BUY", 0.25)])[0]
        self.assertIn("R12_BUY_ON_HELD_USE_ADD", d["codes"])
        d = self.one([prop("BTC-USD", "REDUCE", 0.5)])[0]
        self.assertIn("R12_REDUCE_NOT_LOWER", d["codes"])

    def test_cannot_sell_unheld(self):
        d = self.one([prop("BTC-USD", "EXIT")])[0]
        self.assertIn("R14_NOT_HELD", d["codes"])

    def test_duplicate_orders_same_asset_rejected(self):
        decs = self.one([prop("BTC-USD", "BUY", 0.1), prop("BTC-USD", "BUY", 0.2)])
        self.assertEqual(sorted(d["status"] for d in decs), ["APPROVED", "REJECTED"])

    def test_hold_and_do_nothing_are_noops(self):
        decs = self.one([prop("BTC-USD", "HOLD"), prop("ETH-USD", "DO_NOTHING")])
        self.assertTrue(all(d["status"] == "NOOP" for d in decs))

    def test_halted_blocks_buys_but_allows_sells(self):
        approve_and_fill(self.pf, self.s, [prop("BTC-USD", "BUY", 0.2)])
        self.pf.halted_until = iso(NOW + timedelta(days=3))
        buy = self.one([prop("ETH-USD", "BUY", 0.1)])[0]
        sell = self.one([prop("BTC-USD", "EXIT")])[0]
        self.assertIn("R13_HALTED", buy["codes"])
        self.assertEqual(sell["status"], "APPROVED")

    def test_sells_evaluated_before_buys_so_rotation_works(self):
        approve_and_fill(self.pf, self.s, [prop("BTC-USD", "BUY", 0.27), prop("ETH-USD", "BUY", 0.27)])
        s_before = self.pf.cash
        decs = self.one([prop("SOL-USD", "BUY", 0.15), prop("BTC-USD", "EXIT")])
        self.assertEqual(decs[0]["asset"], "BTC-USD")                        # sell processed first
        self.assertGreater(s_before, 0)

    def test_buys_cannot_exceed_cash(self):
        pf = Portfolio("t", 300.0)
        d = evaluate([prop("BTC-USD", "BUY", 0.9, stop=10)], pf, self.s, CFG, NOW)[0]
        self.assertLessEqual(d["notional_usd"], 300.0)


class TestBreakers(unittest.TestCase):

    def mids(self, px=100.0):
        return {a: px for a in UNI}

    def test_drawdown_halt_liquidates_and_locks_out(self):
        pf = Portfolio("t", 3000.0)
        pf.positions["BTC-USD"] = Position(70.0, 100.0, 18.0, iso(NOW))       # equity 3000 + 7000 at 100
        pf.peak_equity, pf.day_start_date, pf.day_start_equity = 10000.0, NOW.strftime("%Y-%m-%d"), 7000.0
        liquidate, events = update_breakers(pf, self.mids(40.0), NOW, CFG)     # equity 3000 + 2800 = 5800
        self.assertTrue(liquidate)
        self.assertIsNotNone(pf.halted_until)
        self.assertTrue(any(e["type"] == "R10_DRAWDOWN_HALT" for e in events))

    def test_daily_loss_blocks_buys_for_24h(self):
        pf = Portfolio("t", 0.0)
        pf.positions["BTC-USD"] = Position(100.0, 100.0, 18.0, iso(NOW))
        pf.peak_equity, pf.day_start_date, pf.day_start_equity = 10000.0, NOW.strftime("%Y-%m-%d"), 10000.0
        liquidate, events = update_breakers(pf, self.mids(90.0), NOW, CFG)     # -10% on the day
        self.assertFalse(liquidate)
        self.assertIsNotNone(pf.no_buys_until)

    def test_lockout_expiry_resets_peak_so_it_doesnt_instantly_retrip(self):
        pf = Portfolio("t", 6000.0)
        pf.peak_equity = 10000.0
        pf.halted_until = iso(NOW - timedelta(hours=1))
        liquidate, events = update_breakers(pf, {}, NOW, CFG)
        self.assertFalse(liquidate)
        self.assertIsNone(pf.halted_until)
        self.assertEqual(pf.peak_equity, 6000.0)

    def test_portfolio_serialization_roundtrip(self):
        pf = Portfolio("t", 1234.5)
        pf.positions["ETH-USD"] = Position(2.0, 50.0, 12.0, iso(NOW), "c1")
        pf.record_trade(NOW, 100.0, "x")
        pf2 = Portfolio.from_dict(json.loads(json.dumps(pf.to_dict())))
        self.assertEqual(pf2.cash, 1234.5)
        self.assertEqual(pf2.positions["ETH-USD"].stop_price, pf.positions["ETH-USD"].stop_price)
        self.assertEqual(len(pf2.trades), 1)


class TestLockAndIsolation(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pkg = os.path.join(self.tmp, "pkg")
        os.makedirs(os.path.join(self.pkg, "prompts"))
        src = lockmod.PKG_DIR
        for f in ["PREREGISTRATION.md", "experiment.json", os.path.join("prompts", "trader_system_v1.md")]:
            shutil.copy(os.path.join(src, f), os.path.join(self.pkg, f))
        self.state = os.path.join(self.tmp, "state")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_hashes_identical_across_line_endings(self):
        base = lockmod.compute_hashes(self.pkg)
        p = os.path.join(self.pkg, "PREREGISTRATION.md")
        data = open(p, "rb").read().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        open(p, "wb").write(data)
        self.assertEqual(lockmod.compute_hashes(self.pkg), base)

    def test_lock_detects_prompt_edit_and_amendment_accepts_it(self):
        lockmod.create_lock(self.state, self.pkg)
        self.assertTrue(lockmod.verify_lock(self.state, self.pkg)[0])
        with open(os.path.join(self.pkg, "prompts", "trader_system_v1.md"), "a") as f:
            f.write("\nBe more aggressive.\n")
        ok, problems = lockmod.verify_lock(self.state, self.pkg)
        self.assertFalse(ok)
        self.assertTrue(any("trader_system_v1" in x for x in problems))
        lockmod.amend(self.state, "test amendment: prompt wording change", self.pkg)
        self.assertTrue(lockmod.verify_lock(self.state, self.pkg)[0])
        self.assertTrue(os.path.exists(os.path.join(self.state, "AMENDMENTS.md")))

    def test_lock_edit_of_experiment_json_detected(self):
        lockmod.create_lock(self.state, self.pkg)
        p = os.path.join(self.pkg, "experiment.json")
        cfg = json.load(open(p))
        cfg["risk"]["max_gross_exposure"] = 1.0
        json.dump(cfg, open(p, "w"))
        self.assertFalse(lockmod.verify_lock(self.state, self.pkg)[0])

    def test_cannot_lock_twice_or_with_mock(self):
        lockmod.create_lock(self.state, self.pkg)
        with self.assertRaises(RuntimeError):
            lockmod.create_lock(self.state, self.pkg)
        p = os.path.join(self.pkg, "experiment.json")
        cfg = json.load(open(p)); cfg["llm"]["provider"] = "mock"; json.dump(cfg, open(p, "w"))
        with self.assertRaises(RuntimeError):
            lockmod.create_lock(os.path.join(self.tmp, "state2"), self.pkg)

    def test_amendment_needs_real_reason(self):
        lockmod.create_lock(self.state, self.pkg)
        with self.assertRaises(ValueError):
            lockmod.amend(self.state, "x", self.pkg)

    def test_package_is_paper_only_by_construction(self):
        forbidden = ("bot.broker", "ib_async", "alpaca_client", "place_market_order")
        for root, _, files in os.walk(lockmod.PKG_DIR):
            for fn in files:
                if fn.endswith(".py"):
                    src = open(os.path.join(root, fn), encoding="utf-8").read()
                    for bad in forbidden:
                        self.assertNotIn(bad, src, f"{fn} references {bad}")


if __name__ == "__main__":
    unittest.main()
