"""
Capacity gate -- enforces Test 19's lesson at the engine level.

Test 19 found that only 22% of a strategy's ranked signals could actually
be taken once a realistic 20-position capacity limit was simulated,
turning a real per-trade edge negative once the portfolio couldn't hold
every signal at once. A per-trade average is not a portfolio return.

RULE: capacity is checked BEFORE execution, every time -- never assumed,
never simulated after the fact. This module owns capacity only; it knows
nothing about signal quality (Combiner's job) or daily loss/drawdown
limits (RiskManager's job in circuit_breaker.py).
"""
import pandas as pd


class CapacityGate:
    def __init__(self, max_positions: int):
        if max_positions <= 0:
            raise ValueError("max_positions must be positive")
        self.max_positions = max_positions

    def available_slots(self, current_positions: dict) -> int:
        """current_positions: {ticker: quantity}. A ticker with quantity
        0 doesn't count as held."""
        open_count = sum(1 for qty in current_positions.values() if qty != 0)
        return max(0, self.max_positions - open_count)

    def filter(
        self, ranked_candidates: pd.Series, current_positions: dict
    ) -> pd.Series:
        """
        ranked_candidates: Series indexed by ticker, sorted descending by
        attractiveness (Combiner's output). Tickers already held are
        excluded from "new" consideration here -- whether to add to or
        exit an existing position is the execution layer's decision, not
        this gate's; this gate only answers "is there room for a NEW
        name."

        Returns only the top-ranked new candidates that fit in open
        slots. Never partially fills a slot -- a name either gets in or
        it doesn't, same as a real fixed position-count portfolio.
        """
        held = {t for t, q in current_positions.items() if q != 0}
        slots = self.available_slots(current_positions)

        new_candidates = ranked_candidates[~ranked_candidates.index.isin(held)]

        if slots <= 0:
            print(
                f"  [capacity] 0 slots open ({len(held)}/{self.max_positions} "
                f"held) -- no new entries this cycle"
            )
            return pd.Series(dtype=float)

        taken = new_candidates.head(slots)
        dropped = len(new_candidates) - len(taken)
        if dropped > 0:
            print(
                f"  [capacity] {dropped} ranked candidate(s) dropped -- "
                f"only {slots} slot(s) open ({len(held)}/{self.max_positions} held)"
            )

        return taken

    def report(self, ranked_candidates: pd.Series, current_positions: dict) -> dict:
        """
        Diagnostic: what fraction of ranked candidates could actually be
        taken this cycle. This is the exact number Test 19 needed and
        didn't have until the strategy was already dead -- call this on
        every cycle, not just when something looks wrong, so the gap
        between "signal says buy" and "portfolio can hold it" is always
        visible, not discovered after the fact.
        """
        held = {t for t, q in current_positions.items() if q != 0}
        new_candidates = ranked_candidates[~ranked_candidates.index.isin(held)]
        taken = self.filter(ranked_candidates, current_positions)
        total = len(new_candidates)

        return {
            "total_ranked": total,
            "takeable": len(taken),
            "pct_takeable": (len(taken) / total * 100) if total else 0.0,
            "slots_open": self.available_slots(current_positions),
            "positions_held": len(held),
            "max_positions": self.max_positions,
        }
