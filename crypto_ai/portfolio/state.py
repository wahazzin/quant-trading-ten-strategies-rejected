"""
state.py -- portfolio accounting for ONE arm (AI, or one baseline).

CONCEPTS:
  * equity = cash + sum(quantity x current mid price). This is the number every metric uses.
  * weight = position value / equity. The AI thinks in weights; the system converts to orders.
  * peak equity + drawdown = how far we are below the best equity ever reached. The circuit
    breaker watches it.
  * trades[] is a rolling log of traded dollars, used for the 24h turnover limit.

Stops: every position carries a stop price = average cost x (1 - stop_pct/100). Stops can only
be TIGHTENED after a position is opened, never loosened.
"""
from datetime import timedelta

from crypto_ai.journal import iso, parse_iso


class Position:
    def __init__(self, qty, avg_cost, stop_pct, opened_ts, opened_cycle=""):
        self.qty = qty
        self.avg_cost = avg_cost
        self.stop_pct = stop_pct
        self.opened_ts = opened_ts          # ISO string; stop replay only looks at later candles
        self.opened_cycle = opened_cycle

    @property
    def stop_price(self):
        return self.avg_cost * (1 - self.stop_pct / 100.0)

    def to_dict(self):
        return dict(qty=self.qty, avg_cost=self.avg_cost, stop_pct=self.stop_pct,
                    opened_ts=self.opened_ts, opened_cycle=self.opened_cycle)


class Portfolio:
    def __init__(self, name, cash):
        self.name = name
        self.cash = float(cash)
        self.positions = {}                 # asset -> Position
        self.peak_equity = float(cash)
        self.day_start_date = None          # 'YYYY-MM-DD' of the current UTC day
        self.day_start_equity = float(cash)
        self.no_buys_until = None           # ISO or None  (rule R9)
        self.halted_until = None            # ISO or None  (rule R10)
        self.trades = []                    # [{ts, notional, cause}]
        self.fees_usd = 0.0
        self.spread_usd = 0.0
        self.slippage_usd = 0.0
        self.traded_notional_usd = 0.0
        self.last_ts = None                 # ISO of the last snapshot this arm was marked at
        self.extra = {}                     # arm-specific memory (e.g. last rebalance month)

    def equity(self, mids):
        return self.cash + sum(p.qty * mids[a] for a, p in self.positions.items())

    def weights(self, mids):
        eq = self.equity(mids)
        return {a: p.qty * mids[a] / eq for a, p in self.positions.items()} if eq > 0 else {}

    def drawdown(self, mids):
        return self.equity(mids) / self.peak_equity - 1 if self.peak_equity > 0 else 0.0

    def traded_last_24h(self, now):
        cutoff = now - timedelta(hours=24)
        return sum(t["notional"] for t in self.trades if parse_iso(t["ts"]) > cutoff)

    def record_trade(self, now, notional, cause):
        self.trades.append({"ts": iso(now), "notional": notional, "cause": cause})
        self.traded_notional_usd += notional
        cutoff = now - timedelta(hours=48)   # keep 48h; only the last 24h is ever used
        self.trades = [t for t in self.trades if parse_iso(t["ts"]) > cutoff]

    def to_dict(self):
        d = {k: v for k, v in self.__dict__.items() if k != "positions"}
        d["positions"] = {a: p.to_dict() for a, p in self.positions.items()}
        return d

    @classmethod
    def from_dict(cls, d):
        pf = cls(d["name"], d["cash"])
        for k, v in d.items():
            if k != "positions":
                setattr(pf, k, v)
        pf.positions = {a: Position(**p) for a, p in d["positions"].items()}
        return pf
