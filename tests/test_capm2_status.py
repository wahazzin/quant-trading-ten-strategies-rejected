import ast
import os
import unittest

SRC = open(os.path.join(os.path.dirname(__file__), "..", "ops", "capm2_status.py")).read()


class TestReadOnly(unittest.TestCase):
    def test_never_touches_orders_or_modifies_the_broker(self):
        for bad in ("/v2/orders", "requests.patch", "requests.delete", "requests.put", "api.alpaca.markets/v2"):
            self.assertNotIn(bad, SRC)
        self.assertIn("paper-api.alpaca.markets", SRC)

    def test_post_only_used_for_discord_inside_message(self):
        tree = ast.parse(SRC)
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            body = ast.get_source_segment(SRC, fn)
            if fn.name != "message":
                self.assertNotIn(".post(", body, f"POST found in {fn.name}")


if __name__ == "__main__":
    unittest.main()
