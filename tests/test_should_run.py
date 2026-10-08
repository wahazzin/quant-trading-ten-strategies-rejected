import os
import sys
import unittest
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ops"))
from capm2_should_run import decide  # noqa: E402


class TestGate(unittest.TestCase):
    def test_rules(self):
        mon, tue = datetime(2026, 11, 2, 15), datetime(2026, 11, 3, 15)
        self.assertEqual(decide(mon, True, True), "RUN")
        self.assertTrue(decide(mon, False, False).startswith("SKIP"))          # holiday Monday: market closed
        self.assertTrue(decide(tue, True, False).startswith("RUN"))            # ... so Tuesday runs instead
        self.assertEqual(decide(tue, True, True), "SKIP")                      # normal week: Tuesday skips
