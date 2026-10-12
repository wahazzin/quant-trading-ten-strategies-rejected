import os
import sys
import unittest
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ops"))
from capm2_should_run import decide, rebalance_attempted  # noqa: E402


class TestGate(unittest.TestCase):
    def test_rules(self):
        mon, tue = datetime(2026, 11, 2, 15), datetime(2026, 11, 3, 15)
        self.assertEqual(decide(mon, True, True), "RUN")
        self.assertTrue(decide(mon, False, False).startswith("SKIP"))          # holiday Monday: market closed
        self.assertTrue(decide(tue, True, False).startswith("RUN"))            # ... so Tuesday runs instead
        self.assertEqual(decide(tue, True, True, None), "SKIP")                # lookup failed: old rule, skip
        self.assertTrue(decide(tue, True, True, True).startswith("SKIP"))      # normal week: done Monday
        self.assertTrue(decide(tue, True, False, True).startswith("SKIP"))     # done already -> never twice

    def test_missed_monday(self):
        # GitHub started Monday's 15:00 job after the close (seen 2026-10-05, 7h20m late) -> gate skipped
        # Monday; Tuesday must catch up instead of silently skipping the whole week.
        tue = datetime(2026, 11, 3, 15)
        self.assertTrue(decide(tue, True, True, False).startswith("RUN"))
        mon = datetime(2026, 11, 2, 17, 30)
        self.assertTrue(decide(mon, True, True, True).startswith("SKIP"))      # 2nd Monday slot after 1st ran
        self.assertTrue(decide(mon, True, True, None).startswith("SKIP"))      # 2nd slot, history unreadable
        self.assertEqual(decide(datetime(2026, 11, 2, 15), True, True, None), "RUN")  # 1st slot as before

    def test_history(self):
        monday = datetime(2026, 11, 2, tzinfo=timezone.utc)
        ok = [{"name": "Run weekly reallocation", "conclusion": "success"}]
        skipped = [{"name": "Run weekly reallocation", "conclusion": "skipped"}]
        failed = [{"name": "Run weekly reallocation", "conclusion": "failure"}]
        self.assertTrue(rebalance_attempted([("1", "2026-11-02T15:03:00Z", ok)], monday, "9"))
        self.assertFalse(rebalance_attempted([("1", "2026-11-02T21:03:00Z", skipped)], monday, "9"))
        self.assertTrue(rebalance_attempted([("1", "2026-11-02T15:03:00Z", failed)], monday, "9"))  # no auto-retry
        self.assertFalse(rebalance_attempted([("1", "2026-10-26T15:03:00Z", ok)], monday, "9"))     # last week
        self.assertFalse(rebalance_attempted([("9", "2026-11-02T15:03:00Z", ok)], monday, "9"))     # itself


if __name__ == "__main__":
    unittest.main()
