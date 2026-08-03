"""
alpaca_client.py -- Alpaca REST API client, mirroring IBKRClient's
interface (connect, disconnect, get_net_liquidation, get_account_currency,
get_position) so ops/ scripts built against that interface port over with
minimal changes. Paper trading by default (pass paper=False to switch to
live -- never done automatically, and nothing in this project does it).

Unlike ib_async's persistent socket connection to a locally-running
Gateway process, Alpaca's trading API is plain REST over HTTPS -- there
is no gateway to keep running for a months-long forward test. connect()
verifies the credentials/account are reachable (a real GET, not a socket
handshake) so calling code can check success the same way it did with
IBKR; disconnect() is a no-op kept for interface parity.

Credentials: APCA_API_KEY_ID / APCA_API_SECRET_KEY from .env -- the same
pair already used for the Alpaca news API elsewhere in this project.
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL = "https://api.alpaca.markets"
DATA_BASE_URL = "https://data.alpaca.markets"


class AlpacaClient:
    def __init__(self, paper=True):
        self.api_key = os.environ.get("APCA_API_KEY_ID")
        self.api_secret = os.environ.get("APCA_API_SECRET_KEY")
        if not self.api_key or not self.api_secret:
            raise RuntimeError("APCA_API_KEY_ID / APCA_API_SECRET_KEY not found in .env.")
        self.paper = paper
        self.base_url = PAPER_BASE_URL if paper else LIVE_BASE_URL
        self._headers = {"APCA-API-KEY-ID": self.api_key, "APCA-API-SECRET-KEY": self.api_secret}
        self._connected = False

    # ------------------------------------------------------------------
    # IBKRClient-compatible interface
    # ------------------------------------------------------------------

    def connect(self):
        try:
            resp = requests.get(f"{self.base_url}/v2/account", headers=self._headers, timeout=15)
            self._connected = resp.status_code == 200
        except requests.RequestException as e:
            print(f"Alpaca connection check failed: {e}")
            self._connected = False
        if self._connected:
            print(f"Alpaca {'PAPER' if self.paper else 'LIVE'} account reachable "
                  f"({self.base_url}).")
        return self._connected

    def disconnect(self):
        # No persistent session to close -- kept for interface parity with IBKRClient.
        self._connected = False

    def get_net_liquidation(self):
        resp = requests.get(f"{self.base_url}/v2/account", headers=self._headers, timeout=15)
        resp.raise_for_status()
        return float(resp.json()["equity"])

    def get_account_currency(self):
        return "USD"  # Alpaca accounts are USD-denominated; no FX conversion needed downstream

    def get_position(self, symbol):
        resp = requests.get(f"{self.base_url}/v2/positions/{symbol}", headers=self._headers, timeout=15)
        if resp.status_code == 404:
            return 0.0
        resp.raise_for_status()
        return float(resp.json()["qty"])

    # ------------------------------------------------------------------
    # Additional methods needed for the value-portfolio scripts
    # ------------------------------------------------------------------

    def get_positions(self):
        """All positions -- {symbol: qty}."""
        resp = requests.get(f"{self.base_url}/v2/positions", headers=self._headers, timeout=15)
        resp.raise_for_status()
        return {p["symbol"]: float(p["qty"]) for p in resp.json()}

    def get_open_orders(self):
        """Net signed REMAINING (unfilled) quantity of all open/pending orders per
        symbol -- positive = net buy exposure still working, negative = net sell
        exposure still working. This is the piece that was missing before: a
        resting order has zero effect on get_positions() until it fills, so any
        pre-flight check that only looks at positions will recompute the same
        diff and resubmit on every re-run.

        Uses (qty - filled_qty), NOT qty -- an order stays "open" in Alpaca's API
        while PARTIALLY filled (status='partially_filled'), and qty is always the
        original total requested. Using raw qty here double-counts the portion
        that's already filled and visible in get_positions(): a 4-share order
        filled 2-of-4 shows position=2 AND (with the bug) pending=4, so
        reconcile() sees effective=6 against a target of 4 and tries to SELL the
        difference -- exactly what happened the first time this ran (caught only
        because Alpaca's own wash-trade protection rejected the erroneous SELLs)."""
        resp = requests.get(f"{self.base_url}/v2/orders", headers=self._headers,
                             params={"status": "open"}, timeout=15)
        resp.raise_for_status()
        pending = {}
        for o in resp.json():
            sym = o["symbol"]
            remaining = float(o["qty"]) - float(o.get("filled_qty") or 0)
            signed = remaining if o["side"] == "buy" else -remaining
            pending[sym] = pending.get(sym, 0.0) + signed
        return pending

    def get_closed_orders(self, symbol=None, limit=50):
        """Recently closed (filled/cancelled/expired/rejected) orders --
        used to find and journal fills that happened between script runs."""
        params = {"status": "closed", "limit": limit, "direction": "desc"}
        if symbol:
            params["symbols"] = symbol
        resp = requests.get(f"{self.base_url}/v2/orders", headers=self._headers, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_open_orders_raw(self, symbol=None):
        """Full open-order objects, including PARTIALLY FILLED ones still
        working (for cancellation / detailed reporting / fill catch-up --
        a partial fill's filled_qty/filled_avg_price are visible here even
        though the order as a whole is still 'open')."""
        params = {"status": "open"}
        if symbol:
            params["symbols"] = symbol
        resp = requests.get(f"{self.base_url}/v2/orders", headers=self._headers, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def place_market_order(self, symbol, action, qty, tif="gtc"):
        payload = {
            "symbol": symbol,
            "qty": str(int(qty)),
            "side": action.lower(),
            "type": "market",
            "time_in_force": tif,
        }
        resp = requests.post(f"{self.base_url}/v2/orders", headers=self._headers,
                              json=payload, timeout=15)
        if resp.status_code >= 400:
            return {"error": True, "status_code": resp.status_code, "message": resp.text}
        return resp.json()

    def get_order(self, order_id):
        resp = requests.get(f"{self.base_url}/v2/orders/{order_id}", headers=self._headers, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def cancel_order(self, order_id):
        resp = requests.delete(f"{self.base_url}/v2/orders/{order_id}", headers=self._headers, timeout=15)
        return resp.status_code in (200, 204)

    def cancel_all_orders(self):
        resp = requests.delete(f"{self.base_url}/v2/orders", headers=self._headers, timeout=15)
        return resp.status_code in (200, 207)

    def get_last_price(self, symbol):
        """Latest trade price via the market-data API (same APCA key pair)."""
        resp = requests.get(f"{DATA_BASE_URL}/v2/stocks/{symbol}/trades/latest",
                             headers=self._headers, timeout=15)
        resp.raise_for_status()
        return float(resp.json()["trade"]["p"])

    def is_market_open(self):
        resp = requests.get(f"{self.base_url}/v2/clock", headers=self._headers, timeout=15)
        resp.raise_for_status()
        return resp.json()
