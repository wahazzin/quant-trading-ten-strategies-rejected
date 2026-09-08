"""
Engine orchestrator -- wires the full decision pipeline together.

    Universe Screener -> Signal Layer -> Combiner -> Sentiment Veto (log only)
    -> Risk Manager -> Capacity Gate -> Execution -> Journal/Attribution

DRY-RUN BY DEFAULT. dry_run=True (the default) runs the full pipeline and
returns exactly what WOULD have been ordered, without calling a broker.
Every stage's output is captured in the returned CycleResult so a human
can audit the reasoning before anything executes for real.

SENTIMENT VETO IS CURRENTLY A STUB. Per ENGINE_ARCHITECTURE.md and
RESEARCH_LOG Tests 13/14, sentiment is logged, not acted on, until it has
independently proven predictive value in trade_attribution.analyze(). The
hook exists here so wiring it in later never requires touching this
file's control flow -- only the stub function gets replaced.

EXECUTION IS WIRED IN via bot.broker.alpaca_execution.AlpacaExecutor, passed
in optionally as `executor`. dry_run=True (the default) never calls it,
regardless of whether one was passed. dry_run=False with no executor passed
raises rather than silently doing nothing -- an explicit choice made silently
wrong is worse than an explicit error.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from bot.screener.universe import UniverseScreener
from bot.signals.base import SignalRegistry
from bot.combiner.combiner import Combiner
from bot.risk.circuit_breaker import RiskManager
from bot.risk.capacity_gate import CapacityGate
from bot.monitor.discord_notify import notify_circuit_breaker


def sentiment_veto_stub(ticker: str, ticker_data: pd.DataFrame) -> Optional[float]:
    """
    Placeholder. Always returns None (no opinion). This function's ONLY
    job right now is to exist as a place to eventually compute and log a
    sentiment score per ticker for later attribution analysis -- it must
    never block or alter a decision until trade_attribution.analyze()'s
    sentiment-veto section shows it should. Replacing this function is
    the entire integration point for a real sentiment model later.
    """
    return None


@dataclass
class CycleResult:
    as_of: pd.Timestamp
    universe_size: int
    ranked_candidates: pd.Series
    sentiment_scores: dict
    trading_allowed: bool
    approved: pd.Series
    dry_run: bool
    notes: list = field(default_factory=list)


class Orchestrator:
    def __init__(
        self,
        screener: UniverseScreener,
        registry: SignalRegistry,
        combiner: Combiner,
        risk_manager: RiskManager,
        capacity_gate: CapacityGate,
        sentiment_fn: Callable[[str, pd.DataFrame], Optional[float]] = sentiment_veto_stub,
        executor=None,
    ):
        self.screener = screener
        self.registry = registry
        self.combiner = combiner
        self.risk_manager = risk_manager
        self.capacity_gate = capacity_gate
        self.sentiment_fn = sentiment_fn
        self.executor = executor

    def run_cycle(
        self,
        data: pd.DataFrame,
        as_of,
        current_positions: dict,
        equity: float,
        dry_run: bool = True,
    ) -> CycleResult:
        as_of = pd.Timestamp(as_of)
        notes = []

        # 1. Universe screener -- point-in-time, structural filters only
        eligible = self.screener.screen(data, as_of)
        notes.append(f"universe: {len(eligible)} eligible tickers as of {as_of.date()}")

        if eligible.empty:
            return CycleResult(
                as_of, 0, pd.Series(dtype=float), {}, False,
                pd.Series(dtype=float), dry_run,
                notes + ["ABORTED: empty universe, nothing downstream can run"],
            )

        universe_data = data[data["ticker"].isin(eligible.index)]

        # 2. Signal layer + 3. Combiner
        ranked = self.combiner.combine_from_registry(self.registry, universe_data)
        notes.append(f"combiner: {len(ranked)} tickers scored")

        # 4. Sentiment veto -- LOG ONLY. Never filters `ranked`. See
        # sentiment_veto_stub's docstring for why this must stay inert.
        sentiment_scores = {
            ticker: self.sentiment_fn(
                ticker, universe_data[universe_data["ticker"] == ticker]
            )
            for ticker in ranked.index
        }

        # 5. Risk manager -- portfolio-level circuit breaker, checked
        # before capacity so a tripped breaker halts everything downstream
        trading_allowed = self.risk_manager.check_trading_allowed(equity)
        if not trading_allowed:
            notes.append("BLOCKED: risk manager circuit breaker tripped -- no entries this cycle")
            notify_circuit_breaker(f"as of {as_of.date()}, equity={equity:.2f}")
            return CycleResult(
                as_of, len(eligible), ranked, sentiment_scores,
                False, pd.Series(dtype=float), dry_run, notes,
            )

        # 6. Capacity gate -- Test 19's lesson, enforced structurally
        approved = self.capacity_gate.filter(ranked, current_positions)
        cap_report = self.capacity_gate.report(ranked, current_positions)
        notes.append(
            f"capacity: {cap_report['takeable']}/{cap_report['total_ranked']} "
            f"takeable ({cap_report['pct_takeable']:.0f}%)"
        )

        # 7. Execution -- only fires when dry_run=False AND an executor was
        # actually passed in. See module docstring: an explicit error beats
        # a silent no-op when dry_run=False is asked for without an executor.
        placed_orders = []
        if dry_run:
            notes.append(f"DRY RUN: would enter {list(approved.index)} -- no orders sent")
        else:
            if self.executor is None:
                raise RuntimeError(
                    "dry_run=False was requested but no executor was passed to "
                    "this Orchestrator -- refusing to silently do nothing. "
                    "Pass executor=AlpacaExecutor(...) if you mean to trade for real."
                )
            for ticker in approved.index:
                result = self.executor.place_bracket_order(symbol=ticker, equity=equity)
                if result is not None and not result.get("error"):
                    placed_orders.append(result)
                    notes.append(f"LIVE: order placed for {ticker} -- {result}")
                else:
                    notes.append(f"LIVE: order NOT placed for {ticker} -- {result}")

        return CycleResult(
            as_of=as_of,
            universe_size=len(eligible),
            ranked_candidates=ranked,
            sentiment_scores=sentiment_scores,
            trading_allowed=trading_allowed,
            approved=approved,
            dry_run=dry_run,
            notes=notes,
        )

    def print_report(self, result: CycleResult) -> None:
        print("=" * 60)
        print(
            f"ORCHESTRATOR CYCLE -- {result.as_of.date()} "
            f"{'[DRY RUN]' if result.dry_run else '[LIVE]'}"
        )
        print("=" * 60)
        for note in result.notes:
            print(f"  - {note}")
        if not result.approved.empty:
            print("\n  Approved candidates:")
            for ticker, score in result.approved.items():
                print(f"    {ticker:<8} score={score:+.3f}")
        print("=" * 60)
