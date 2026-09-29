"""
test_crypto_ai_metrics.py -- statistics and verdict logic. A bug here could turn a null result
into a false "PREDICTIVE", so thresholds are tested exactly as pre-registered.

Run with:  python -m unittest tests.test_crypto_ai_metrics -v
"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_ai.analytics import metrics as M


def ic(mean, t, h1=0.05, h2=0.05):
    return {"mean_ic": mean, "t_nw": t, "first_half_ic": h1, "second_half_ic": h2}


class TestMetrics(unittest.TestCase):

    def test_sharpe_and_sortino(self):
        r = np.array([0.01, -0.005, 0.02, 0.0, -0.01])
        self.assertAlmostEqual(M.sharpe(r), r.mean() / r.std(ddof=1) * math.sqrt(365))
        self.assertTrue(math.isnan(M.sharpe([0.01, 0.01])))
        self.assertTrue(math.isnan(M.sortino([0.01, 0.02])))      # no downside -> undefined

    def test_max_drawdown(self):
        self.assertAlmostEqual(M.max_drawdown([100, 120, 90, 130, 104]), -0.25)

    def test_alpha_beta_recovers_known_values(self):
        rng = np.random.default_rng(0)
        b = rng.normal(0, 0.03, 400)
        r = 0.001 + 0.5 * b + rng.normal(0, 0.001, 400)
        ab = M.alpha_beta(r, b)
        self.assertAlmostEqual(ab["beta"], 0.5, delta=0.01)
        self.assertAlmostEqual(ab["alpha_ann"], 0.365, delta=0.05)

    def test_newey_west_on_noise_is_not_significant(self):
        x = np.random.default_rng(1).normal(0, 1, 500)
        t, n = M.newey_west_t(x, 5)
        self.assertEqual(n, 500)
        self.assertLess(abs(t), 3)

    def test_newey_west_is_more_conservative_on_autocorrelated_data(self):
        rng = np.random.default_rng(2)
        e = rng.normal(0.1, 1, 600)
        x = np.convolve(e, np.ones(5) / 5, mode="valid")          # strongly autocorrelated
        naive = x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))
        t, _ = M.newey_west_t(x, 5)
        self.assertLess(abs(t), abs(naive))

    def test_spearman(self):
        self.assertAlmostEqual(M.spearman([1, 2, 3], [0.1, 0.2, 0.3]), 1.0)
        self.assertAlmostEqual(M.spearman([1, 2, 3], [0.3, 0.2, 0.1]), -1.0)
        self.assertTrue(math.isnan(M.spearman([0, 0, 0], [0.1, 0.2, 0.3])))

    def test_ic_series_alignment_and_skipped_cycles(self):
        snaps = [{"cycle_id": c, "assets": {"A": {"mid": a}, "B": {"mid": b}}}
                 for c, a, b in [("2026-10-01T00:00Z", 100, 100), ("2026-10-02T00:00Z", 110, 90)]]
        dec = [{"arm": "ai_pv", "cycle_id": "2026-10-01T00:00Z", "outlook": {"A": 2, "B": -2}},
               {"arm": "ai_pv", "cycle_id": "2026-10-01T06:00Z", "outlook": {"A": 2, "B": -2}},
               {"arm": "trend_quant", "cycle_id": "2026-10-01T00:00Z"}]
        rows = M.ic_series(dec, snaps, 24)
        self.assertEqual(len(rows), 1)                            # 06:00 has no t+24h snapshot
        self.assertAlmostEqual(rows[0]["ic"], 1.0)


class TestVerdict(unittest.TestCase):
    """Exactly the thresholds in PREREGISTRATION.md section 7."""

    def test_predictive_requires_everything(self):
        self.assertEqual(M.primary_verdict(ic(0.03, 2.0), ic(0.01, 1)), "PREDICTIVE")
        self.assertNotEqual(M.primary_verdict(ic(0.029, 2.5), ic(0.01, 1)), "PREDICTIVE")
        self.assertNotEqual(M.primary_verdict(ic(0.05, 1.99), ic(0.01, 1)), "PREDICTIVE")
        self.assertNotEqual(M.primary_verdict(ic(0.05, 3.0), ic(-0.01, 1)), "PREDICTIVE")
        self.assertNotEqual(M.primary_verdict(ic(0.05, 3.0, h1=-0.01), ic(0.01, 1)), "PREDICTIVE")

    def test_other_labels(self):
        self.assertEqual(M.primary_verdict(ic(0.025, 1.5), ic(0, 0)), "INCONCLUSIVE")
        self.assertEqual(M.primary_verdict(ic(-0.04, -2.5), ic(0, 0)), "HARMFUL")
        self.assertEqual(M.primary_verdict(ic(0.01, 0.8), ic(0, 0)), "NO EVIDENCE")
        self.assertEqual(M.primary_verdict(ic(float("nan"), float("nan")), ic(0, 0)), "NOT EVALUABLE")


if __name__ == "__main__":
    unittest.main()
