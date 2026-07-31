from datetime import date


class RiskManager:
    """
    Circuit breaker v2 -- now with persistent memory.

    Old version's fatal flaw: peak_equity and start_of_day_equity lived
    only in RAM. Every restart reset them to your current equity, which
    meant the 20% max-drawdown rule effectively never accumulated.
    You could lose 19% today, restart tomorrow, lose 19% again, forever.

    Now state is loaded from / saved to the trading.db via TradeJournal,
    so drawdown protection is measured from your true all-time peak.

    Locked parameters (your rule 12: these are hard limits):
      - max daily loss: 7.5% from start-of-day equity
      - max drawdown:  20% from all-time peak equity
    """

    def __init__(self, journal, max_daily_loss_pct=0.075, max_drawdown_pct=0.20):
        self.journal = journal
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct

        state = self.journal.load_risk_state()
        if state:
            self.current_day = state["day"]
            self.start_of_day_equity = state["start_of_day_equity"]
            self.peak_equity = state["peak_equity"]
        else:
            self.current_day = date.today()
            self.start_of_day_equity = None
            self.peak_equity = None

    def _persist(self):
        self.journal.save_risk_state(
            day=self.current_day,
            start_of_day_equity=self.start_of_day_equity,
            peak_equity=self.peak_equity,
        )

    def _reset_new_day(self, current_equity):
        today = date.today()
        if today != self.current_day:
            self.current_day = today
            self.start_of_day_equity = current_equity
        if self.start_of_day_equity is None:
            self.start_of_day_equity = current_equity

    def _update_peak(self, current_equity):
        if self.peak_equity is None or current_equity > self.peak_equity:
            self.peak_equity = current_equity

    def _refresh(self, current_equity):
        """
        Shared refresh step. Both check_trading_allowed() and status()
        call this FIRST, so status() can never show stale numbers again
        (that was the bug: status() used to skip this and could print
        yesterday's daily-loss figure before a trade ever ran today).
        """
        self._reset_new_day(current_equity)
        self._update_peak(current_equity)
        self._persist()

    def check_trading_allowed(self, current_equity):
        if current_equity is None or current_equity <= 0:
            print("BLOCKED: invalid equity value -- refusing to trade blind.")
            return False

        self._refresh(current_equity)

        daily_loss_pct = (
            (self.start_of_day_equity - current_equity) / self.start_of_day_equity
        )
        drawdown_pct = (self.peak_equity - current_equity) / self.peak_equity

        if daily_loss_pct >= self.max_daily_loss_pct:
            print(
                f"🛑 CIRCUIT BREAKER: daily loss {daily_loss_pct:.1%} "
                f">= limit {self.max_daily_loss_pct:.1%}. Trading halted for today."
            )
            return False

        if drawdown_pct >= self.max_drawdown_pct:
            print(
                f"🛑 CIRCUIT BREAKER: drawdown {drawdown_pct:.1%} from peak "
                f"{self.peak_equity:.2f} >= limit {self.max_drawdown_pct:.1%}. "
                f"Trading halted. Manual review required."
            )
            return False

        return True

    def status(self, current_equity):
        """Human-readable snapshot for logging / weekly reports."""
        self._refresh(current_equity)
        daily = (self.start_of_day_equity - current_equity) / self.start_of_day_equity
        dd = (self.peak_equity - current_equity) / self.peak_equity
        return (
            f"Day P&L: {-daily:+.2%} (limit -{self.max_daily_loss_pct:.1%}) | "
            f"Drawdown from peak: {dd:.2%} (limit {self.max_drawdown_pct:.1%}) | "
            f"Peak: {self.peak_equity:.2f}"
        )