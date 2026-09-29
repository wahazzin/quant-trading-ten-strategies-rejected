"""
test_crypto_ai_pipeline.py -- schema validation, thesis handling, and full end-to-end cycles
on a FAKE offline market (no network, no API cost, fully deterministic).

Run with:  python -m unittest tests.test_crypto_ai_pipeline -v
"""
import json
import math
import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_ai.agents.llm import MockLLM
from crypto_ai.agents.schemas import validate
from crypto_ai.agents.trader import apply_thesis_updates
from crypto_ai.journal import Journal
from crypto_ai.lock import load_config
from crypto_ai.market_data.coinbase import DataFault, closed_only
from crypto_ai.runner import cycle_id_for, run_cycle

CFG = load_config()
UNI = CFG["universe"]
CID = "2026-10-01T12:00Z"


def good(**over):
    obj = {"cycle_id": CID, "outlook": {a: 0 for a in UNI},
           "decisions": [{"asset": "BTC-USD", "action": "BUY", "target_weight": 0.1,
                          "reasons": ["r"], "invalidation": ["i"], "confidence": 60}],
           "thesis_updates": [], "portfolio_note": ""}
    obj.update(over)
    return obj


class TestSchema(unittest.TestCase):

    def check(self, obj_or_text, ok):
        raw = obj_or_text if isinstance(obj_or_text, str) else json.dumps(obj_or_text)
        parsed, errors = validate(raw, CFG, CID)
        self.assertEqual(parsed is not None, ok, errors)
        return parsed, errors

    def test_valid(self):
        self.check(good(), True)

    def test_garbage_and_non_object(self):
        self.check("I think BTC goes up", False)
        self.check("[1,2]", False)

    def test_fence_is_stripped_but_recorded(self):
        p, _ = self.check("```json\n" + json.dumps(good()) + "\n```", True)
        self.assertTrue(p["_fence_stripped"])

    def test_outlook_must_cover_every_asset_with_int_in_range(self):
        o = {a: 0 for a in UNI}; del o["LTC-USD"]
        self.check(good(outlook=o), False)
        self.check(good(outlook={**{a: 0 for a in UNI}, "BTC-USD": 3}), False)
        self.check(good(outlook={**{a: 0 for a in UNI}, "BTC-USD": True}), False)
        self.check(good(outlook={**{a: 0 for a in UNI}, "BTC-USD": 1.5}), False)
        self.check(good(outlook={**{a: 0 for a in UNI}, "DOGE-USD": 1}), False)

    def test_buy_without_thesis_rejected(self):
        d = good()["decisions"][0]
        self.check(good(decisions=[{**d, "invalidation": []}]), False)
        self.check(good(decisions=[{**d, "reasons": None}]), False)

    def test_bad_fields_rejected(self):
        d = good()["decisions"][0]
        self.check(good(decisions=[{**d, "asset": "DOGE-USD"}]), False)
        self.check(good(decisions=[{**d, "action": "YOLO"}]), False)
        self.check(good(decisions=[{**d, "target_weight": 1.5}]), False)
        self.check(good(decisions=[{**d, "confidence": 150}]), False)
        self.check(good(decisions=[d, d]), False)
        self.check(good(cycle_id="2026-01-01T00:00Z"), False)

    def test_hold_needs_nothing_extra(self):
        self.check(good(decisions=[{"asset": "BTC-USD", "action": "HOLD"}]), True)


class TestTheses(unittest.TestCase):

    def test_open_update_close(self):
        p = good(thesis_updates=[{"asset": "BTC-USD", "status": "bullish", "summary": "s",
                                  "reasons": ["r"], "invalidation": ["i"]}])
        th, hist = apply_thesis_updates({}, p, CID)
        self.assertEqual(th["BTC-USD"]["confidence_history"][-1]["confidence"], 60)
        p2 = good(cycle_id="c2", decisions=[], thesis_updates=[{"asset": "BTC-USD", "status": "closed"}])
        th2, hist2 = apply_thesis_updates(th, p2, "c2")
        self.assertNotIn("BTC-USD", th2)
        self.assertIn("BTC-USD", th)                    # original not mutated
        self.assertEqual(hist2[0]["status"], "closed")  # history keeps it


class FakeClient:
    """Deterministic synthetic market. Includes one in-progress candle per series, which the
    system must ignore."""

    def __init__(self, now, fail=False):
        self.now, self.fail = now, fail
        self.base = {a: 100.0 * (i + 1) for i, a in enumerate(UNI)}

    def _px(self, a, t):
        return self.base[a] * (1 + 0.05 * math.sin(t / 86400.0 + len(a)))

    def ticker(self, p):
        if self.fail:
            raise DataFault("simulated outage")
        px = self._px(p, self.now.timestamp())
        return {"price": px, "bid": px * 0.9999, "ask": px * 1.0001, "volume_base": 1e8 / px, "time": self.now}

    def candles(self, p, g):
        end = int(self.now.timestamp()) // g * g            # start of the in-progress candle
        rows = []
        for k in range(350):
            t = end - k * g
            c = self._px(p, t + g)
            rows.append({"t": t, "open": self._px(p, t), "high": c * 1.01, "low": c * 0.99,
                         "close": c, "volume": 1e6})
        return sorted(rows, key=lambda r: r["t"])


class BadLLM:
    def complete(self, system, user, ctx=None):
        return {"text": "Sure! BTC looks great.", "model_id": "bad", "tokens_in": 1,
                "tokens_out": 1, "latency_s": 0}


class TestEndToEnd(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.t0 = datetime(2026, 10, 1, 12, 5, tzinfo=timezone.utc)
        self.j = Journal(self.dir)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def run_at(self, t, llm=None, client=None, lock=False):
        return run_cycle(self.dir, CFG, client or FakeClient(t), llm or MockLLM(), now=t,
                         dry_run=True, require_lock=lock)

    def test_closed_only_drops_in_progress_candle(self):
        rows = FakeClient(self.t0).candles("BTC-USD", 3600)
        kept = closed_only(rows, 3600, int(self.t0.timestamp()))
        self.assertEqual(len(kept), len(rows) - 1)
        self.assertTrue(all(r["t"] + 3600 <= self.t0.timestamp() for r in kept))

    def test_cycle_id_slots(self):
        self.assertEqual(cycle_id_for(datetime(2026, 1, 1, 17, 59, tzinfo=timezone.utc)), "2026-01-01T12:00Z")
        self.assertEqual(cycle_id_for(datetime(2026, 1, 1, 18, 0, tzinfo=timezone.utc)), "2026-01-01T18:00Z")

    def test_full_cycle_then_idempotent_then_next_cycle(self):
        r = self.run_at(self.t0)
        self.assertEqual(r["status"], "OK", r)
        arms = {e["arm"] for e in self.j.read("equity.jsonl")}
        self.assertEqual(arms, set(CFG["arms"]))
        self.assertEqual(len(self.j.read("snapshots.jsonl")), 1)
        self.assertEqual(self.run_at(self.t0 + timedelta(minutes=30))["status"], "ALREADY_DONE")
        r2 = self.run_at(self.t0 + timedelta(hours=6))
        self.assertEqual(r2["status"], "OK")
        ai_buys = [o for o in self.j.read("orders.jsonl") if o["arm"] == "ai_pv" and o["side"] == "buy"]
        self.assertEqual(len(ai_buys), 1)                  # mock HOLDs on cycle 2, no double buy
        bh = [o for o in self.j.read("orders.jsonl") if o["arm"] == "btc_hold"]
        self.assertEqual(len(bh), 1)                       # buy-and-hold really holds
        self.assertTrue(os.path.exists(os.path.join(self.dir, "theses.json")))
        fb = self.j.load_json("ai_feedback.json")
        self.assertEqual(fb["from_cycle"], "2026-10-01T18:00Z")

    def test_bad_llm_output_means_no_ai_trades_but_baselines_run(self):
        r = self.run_at(self.t0, llm=BadLLM())
        self.assertEqual(r["status"], "OK_AI_FAULT")
        self.assertFalse(any(o["arm"] == "ai_pv" for o in self.j.read("orders.jsonl")))
        self.assertTrue(any(o["arm"] == "btc_hold" for o in self.j.read("orders.jsonl")))
        p = self.j.read("prompts.jsonl")[0]
        self.assertEqual(len(p["attempts"]), 1 + CFG["llm"]["parse_retries"])
        self.assertTrue(any(e["type"] == "AI_FAULT" for e in self.j.read("events.jsonl")))

    def test_data_fault_skips_cycle_and_allows_retry(self):
        r = self.run_at(self.t0, client=FakeClient(self.t0, fail=True))
        self.assertEqual(r["status"], "DATA_FAULT")
        self.assertEqual(self.j.read("orders.jsonl"), [])
        self.assertEqual(self.run_at(self.t0 + timedelta(minutes=20))["status"], "OK")

    def test_refuses_to_run_without_lock_when_required(self):
        r = self.run_at(self.t0, lock=True)
        self.assertEqual(r["status"], "LOCK_MISMATCH")
        self.assertEqual(self.j.read("orders.jsonl"), [])

    def test_every_ai_decision_and_risk_outcome_is_logged(self):
        self.run_at(self.t0)
        rows = [r for r in self.j.read("decisions.jsonl") if r["arm"] == "ai_pv"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]["outlook"]), set(UNI))
        self.assertTrue(all("status" in d and "codes" in d for d in rows[0]["risk_decisions"]))


class TestFreeProvider(unittest.TestCase):
    """The free OpenAI-compatible provider: falls back between endpoints serving the SAME model,
    never to a different one, and fails loudly when nothing is configured."""

    class Resp:
        def __init__(self, code, body):
            self.status_code, self._b, self.text = code, body, json.dumps(body)

        def json(self):
            return self._b

    def setUp(self):
        from crypto_ai.agents import llm as L
        self.L = L
        self._post, self._sleep = L.requests.post, L.time.sleep
        L.time.sleep = lambda s: None
        self._env = {k: os.environ.pop(k, None) for k in ("GROQ_API_KEY", "OPENROUTER_API_KEY")}

    def tearDown(self):
        self.L.requests.post, self.L.time.sleep = self._post, self._sleep
        for k, v in self._env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def test_no_keys_fails_loudly(self):
        with self.assertRaises(self.L.LLMError):
            self.L.make_llm(CFG)

    def test_falls_back_to_same_weights_on_second_provider(self):
        os.environ["GROQ_API_KEY"], os.environ["OPENROUTER_API_KEY"] = "g", "o"
        calls = []

        def post(url, headers, json, timeout):
            calls.append(json["model"])
            if "groq" in url:
                return self.Resp(429, {"error": "rate limited"})
            return self.Resp(200, {"model": json["model"], "choices": [{"message": {"content": "{}"},
                                   "finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}})

        self.L.requests.post = post
        out = self.L.make_llm(CFG).complete("s", "u")
        self.assertEqual(out["provider"], "openrouter")
        self.assertEqual(len(out["endpoint_errors"]), 2)
        self.assertTrue(all("gpt-oss-120b" in m for m in calls))     # same weights every call

    def test_all_endpoints_down_raises(self):
        os.environ["GROQ_API_KEY"] = "g"
        self.L.requests.post = lambda url, headers, json, timeout: self.Resp(503, {})
        with self.assertRaises(self.L.LLMError):
            self.L.make_llm(CFG).complete("s", "u")


if __name__ == "__main__":
    unittest.main()
