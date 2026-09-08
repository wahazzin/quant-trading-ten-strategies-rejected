"""
Signal interface. Every signal is a plugin implementing one method.

The engine never imports a specific signal -- it iterates the registry.
That is what lets a signal be swapped, added, or retired without touching
the engine, and why the machine outlives any single hypothesis.

CONTRACT:
    score(data) -> pd.Series indexed by ticker, higher = more attractive

A signal MUST NOT look ahead. It may only use rows dated strictly before
the decision date. The engine slices the frame before handing it over;
signals must never re-read raw files themselves.
"""
from abc import ABC, abstractmethod
import pandas as pd


class Signal(ABC):
    name: str = "unnamed"
    # Evidence status, carried explicitly so nobody forgets what is proven:
    #   "unproven"  -- no out-of-sample support yet
    #   "rejected"  -- tested and failed (kept for logging/comparison only)
    #   "validated" -- cleared a sealed holdout
    evidence: str = "unproven"
    required_columns: tuple = ("ticker", "date", "close")

    @abstractmethod
    def score(self, data: pd.DataFrame) -> pd.Series:
        """Score per ticker. Higher = more attractive. NaN = skip."""
        raise NotImplementedError

    def validate_input(self, data: pd.DataFrame) -> None:
        missing = [c for c in self.required_columns if c not in data.columns]
        if missing:
            raise ValueError(f"{self.name}: missing columns {missing}")

    def __repr__(self) -> str:
        return f"<Signal {self.name} evidence={self.evidence}>"


class SignalRegistry:
    """Holds active signals. The engine iterates this and nothing else."""

    def __init__(self):
        self._signals: dict[str, Signal] = {}

    def register(self, signal: Signal) -> None:
        if signal.name in self._signals:
            raise ValueError(f"duplicate signal name: {signal.name}")
        self._signals[signal.name] = signal

    def unregister(self, name: str) -> None:
        self._signals.pop(name, None)

    def active(self) -> list:
        return list(self._signals.values())

    def score_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run every registered signal -> DataFrame indexed by ticker, one
        column per signal. A signal that raises is logged and dropped
        rather than killing the run: a broken signal must never take down
        live execution.
        """
        out = {}
        for sig in self._signals.values():
            try:
                sig.validate_input(data)
                s = sig.score(data)
                if not isinstance(s, pd.Series):
                    raise TypeError("score() must return a pd.Series")
                out[sig.name] = s
            except Exception as e:      # noqa: BLE001 - deliberate
                print(f"  [signal error] {sig.name}: {e}")
        return pd.DataFrame(out)

    def __len__(self) -> int:
        return len(self._signals)
