import os
import unittest


class TestReadOnly(unittest.TestCase):
    def test_status_script_never_writes_to_the_broker(self):
        src = open(os.path.join(os.path.dirname(__file__), "..", "ops", "capm2_status.py")).read()
        for bad in ("requests.post", "requests.patch", "requests.delete", "requests.put", "/v2/orders"):
            self.assertNotIn(bad, src)
        self.assertIn("paper-api.alpaca.markets", src)


if __name__ == "__main__":
    unittest.main()
