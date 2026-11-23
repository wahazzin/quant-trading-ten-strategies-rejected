import os
import sys
import unittest
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ops"))
from capm2_crash_prices import prior_close, ny_today  # noqa: E402

BARS = [
    {"t": "2026-11-18T05:00:00Z", "c": 100.0},
    {"t": "2026-11-19T05:00:00Z", "c": 97.0},
    {"t": "2026-11-20T05:00:00Z", "c": 88.0},   # today's in-progress bar (close = latest trade)
]


class TestCrashPrices(unittest.TestCase):
    def test_skips_todays_partial_bar(self):
        # Old code used bars[-1] -> 88/88-1 = 0%. A -9.3% day must be visible.
        prev = prior_close(BARS, "2026-11-20")
        self.assertEqual(prev, 97.0)
        self.assertAlmostEqual(88.0 / prev - 1, -0.0928, places=3)

    def test_no_partial_bar_yet(self):
        self.assertEqual(prior_close(BARS[:2], "2026-11-20"), 97.0)

    def test_no_prior_bar(self):
        self.assertIsNone(prior_close(BARS[2:], "2026-11-20"))
        self.assertIsNone(prior_close([], "2026-11-20"))

    def test_ny_date(self):
        # 03:00 UTC Fri = 22:00 Thu in New York
        self.assertEqual(ny_today(datetime(2026, 11, 20, 3, 0, tzinfo=timezone.utc)), "2026-11-19")
        self.assertEqual(ny_today(datetime(2026, 11, 20, 15, 0, tzinfo=timezone.utc)), "2026-11-20")


if __name__ == "__main__":
    unittest.main()
