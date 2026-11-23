import os
import sys
import unittest
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ops"))
from capm2_week_check import should_alert  # noqa: E402


class TestWeekCheck(unittest.TestCase):
    def test_rules(self):
        wed = datetime(2026, 11, 18, 1, 30, tzinfo=timezone.utc)   # after Tuesday's close, ISO week 47
        self.assertEqual(should_alert(wed, True, False, None), "2026-W47")       # skipped week -> alert
        self.assertIsNone(should_alert(wed, True, True, None))                   # rebalanced -> quiet
        self.assertIsNone(should_alert(wed, True, None, None))                   # history unreadable -> quiet
        self.assertIsNone(should_alert(wed, False, False, None))                 # Mon+Tue holidays -> quiet
        self.assertIsNone(should_alert(wed, True, False, "2026-W47"))            # alert only once a week
        tue = datetime(2026, 11, 17, 22, 0, tzinfo=timezone.utc)
        self.assertIsNone(should_alert(tue, True, False, None))                  # Tuesday: catch-up still possible


if __name__ == "__main__":
    unittest.main()
