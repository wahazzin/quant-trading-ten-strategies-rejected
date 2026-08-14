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

Credentials: APCA_API_KEY_ID / APCA_API_SECRET_KEY from .env by default --
the same pair already used for the Alpaca news API elsewhere in this
project, and for the value-portfolio forward test. ops/capm_monitor.py
tracks two OTHER Alpaca paper accounts (different key pairs, read from
.env as CAPM_US_KEY_ID/CAPM_US_SECRET_KEY and CAPM_SE_KEY_ID/
CAPM_SE_SECRET_KEY) -- pass api_key/api_secret explicitly to connect to
one of those instead of the default account.
"""
import os
import requests
from dotenv import load_dotenv

from bot.broker.reconcile import pending_quantity

load_dotenv()

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL = "https://api.alpaca.markets"
DATA_BASE_URL = "https://data.alpaca.markets"


class AlpacaClient:
    def __init__(self, paper=True, api_key=None, api_secret=None):
        self.api_key = api_key or os.environ.get("APCA_API_KEY_ID")
        self.api_secret = api_secret or os.environ.get("APCA_API_SECRET_KEY")
        if not self.api_key or not self.api_secret:
            raise RuntimeError("Alpaca API key/secret not found -- pass api_key/api_secret "
                                "explicitly, or set APCA_API_KEY_ID / APCA_API_SECRET_KEY in .env.")
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

    def get_positions_raw(self):
        """Full position objects as Alpaca returns them (qty, avg_entry_price,
        market_value, cost_basis, unrealized_pl, unrealized_plpc, ...) -- used
        by ops/capm_monitor.py, which has no local trade journal for these
        accounts and reads P&L directly from the broker's own numbers instead
        of re-deriving them."""
        resp = requests.get(f"{self.base_url}/v2/positions", headers=self._headers, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_open_orders(self):
        """Net signed REMAINING (unfilled) quantity of all open/pending orders per
        symbol -- positive = net buy exposure still working, negative = net sell
        exposure still working. This is display/reporting sugar only now; the
        actual reconciliation math lives in bot/broker/reconcile.py's
        pending_quantity(), which takes the raw order objects (see
        get_open_orders_raw()) and does this computation itself. It used to be
        duplicated here too -- which is exactly how a partial-fill bug survived
        the original "fix the reconciliation bug at the root" pass (a resting
        order stays "open" while PARTIALLY filled, and using raw qty instead of
        qty - filled_qty double-counts the already-filled portion). See
        reconcile.py's module docstring, incident #4, for the full story."""
        return pending_quantity(self.get_open_orders_raw())

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

    def get_daily_bars(self, symbol, start, end):
        """Daily OHLCV bars for `symbol` between start and end (date strings or
        anything str()-able as YYYY-MM-DD), oldest first -- used by
        ops/event_monitor.py to compute forward returns from live data once
        those dates arrive (data/yf_universe.parquet is a static historical
        snapshot, not updated going forward).

        feed=iex: this account's data plan returns 403 on the default (SIP)
        feed for historical bars; IEX is the free tier and is what
        get_last_price already implicitly relies on being available."""
        resp = requests.get(f"{DATA_BASE_URL}/v2/stocks/{symbol}/bars",
                             headers=self._headers,
                             params={"timeframe": "1Day", "start": str(start), "end": str(end),
                                     "adjustment": "split", "limit": 10000, "feed": "iex"},
                             timeout=15)
        resp.raise_for_status()
        return resp.json().get("bars") or []

    def is_market_open(self):
        resp = requests.get(f"{self.base_url}/v2/clock", headers=self._headers, timeout=15)
        resp.raise_for_status()
        return resp.json()
