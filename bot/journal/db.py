from sqlalchemy import create_engine, Column, Integer, Float, String, Date, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, date

Base = declarative_base()


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    symbol = Column(String)
    entry_time = Column(DateTime)
    exit_time = Column(DateTime)
    side = Column(String)
    quantity = Column(Integer)
    entry_price = Column(Float)
    exit_price = Column(Float)
    pnl = Column(Float)
    fees = Column(Float)
    strategy = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class AccountSnapshot(Base):
    __tablename__ = "account_snapshots"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    net_liquidation = Column(Float)
    cash = Column(Float)


class OpenPosition(Base):
    """
    The fix for the fake-entry-price bug AND the restart problem.
    Every buy fill is recorded here immediately, so when the position
    closes -- even days later, even in a different script run -- we
    know the REAL entry price and time.
    """
    __tablename__ = "open_positions"

    id = Column(Integer, primary_key=True)
    symbol = Column(String, unique=True)
    quantity = Column(Float)
    avg_entry_price = Column(Float)
    entry_time = Column(DateTime)


class RiskState(Base):
    """
    Persistent memory for the RiskManager. Without this, every restart
    wiped peak_equity and start_of_day_equity, which silently disabled
    your 20% max-drawdown protection.
    """
    __tablename__ = "risk_state"

    id = Column(Integer, primary_key=True)
    day = Column(Date)
    start_of_day_equity = Column(Float)
    peak_equity = Column(Float)


class TradeJournal:
    def __init__(self, db_path="sqlite:///trading.db"):
        self.engine = create_engine(db_path, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    # ------------------------------------------------------------------
    # ENTRY / EXIT FILL TRACKING (used by TradeMonitor)
    # ------------------------------------------------------------------

    def record_entry_fill(self, symbol, shares, price, fill_time):
        """
        Called on every BUY fill. Maintains a weighted-average entry price
        so partial fills and add-ons are handled correctly.
        """
        session = self.Session()
        try:
            pos = session.query(OpenPosition).filter_by(symbol=symbol).first()

            if pos is None:
                pos = OpenPosition(
                    symbol=symbol,
                    quantity=shares,
                    avg_entry_price=price,
                    entry_time=fill_time,
                )
                session.add(pos)
            else:
                total_cost = (pos.avg_entry_price * pos.quantity) + (price * shares)
                pos.quantity += shares
                pos.avg_entry_price = total_cost / pos.quantity

            session.commit()
        finally:
            session.close()

    def record_exit_fill(self, symbol, shares, price, fill_time, fees=0.0,
                         strategy="swing_v1"):
        """
        Called on every SELL fill. Logs a completed trade using the REAL
        stored entry price. Handles partial exits: sells 30 of 69 shares ->
        logs a 30-share trade, keeps 39 open.
        Returns True if a trade was logged.
        """
        session = self.Session()
        try:
            pos = session.query(OpenPosition).filter_by(symbol=symbol).first()

            if pos is None or pos.quantity <= 0:
                # Sold something we never recorded buying (e.g. position
                # predates this system). Nothing reliable to log.
                print(f"⚠️ SELL fill for {symbol} but no recorded entry. "
                      f"Skipping journal entry.")
                return False

            closed_qty = min(shares, pos.quantity)
            pnl = (price - pos.avg_entry_price) * closed_qty

            trade = Trade(
                symbol=symbol,
                side="LONG",
                quantity=int(closed_qty),
                entry_price=pos.avg_entry_price,
                exit_price=price,
                pnl=pnl,
                fees=fees,
                entry_time=pos.entry_time,
                exit_time=fill_time,
                strategy=strategy,
            )
            session.add(trade)

            pos.quantity -= closed_qty
            if pos.quantity <= 0:
                session.delete(pos)

            session.commit()
            print(f"✅ Trade logged: {symbol} {int(closed_qty)} shares | "
                  f"entry {pos.avg_entry_price:.2f} -> exit {price:.2f} | "
                  f"PnL {pnl:+.2f}")
            return True
        finally:
            session.close()

    def get_open_position(self, symbol):
        session = self.Session()
        try:
            pos = session.query(OpenPosition).filter_by(symbol=symbol).first()
            if pos is None:
                return None
            return {
                "symbol": pos.symbol,
                "quantity": pos.quantity,
                "avg_entry_price": pos.avg_entry_price,
                "entry_time": pos.entry_time,
            }
        finally:
            session.close()

    # ------------------------------------------------------------------
    # RISK STATE PERSISTENCE (used by RiskManager)
    # ------------------------------------------------------------------

    def load_risk_state(self):
        session = self.Session()
        try:
            state = session.query(RiskState).first()
            if state is None:
                return None
            return {
                "day": state.day,
                "start_of_day_equity": state.start_of_day_equity,
                "peak_equity": state.peak_equity,
            }
        finally:
            session.close()

    def save_risk_state(self, day, start_of_day_equity, peak_equity):
        session = self.Session()
        try:
            state = session.query(RiskState).first()
            if state is None:
                state = RiskState(
                    day=day,
                    start_of_day_equity=start_of_day_equity,
                    peak_equity=peak_equity,
                )
                session.add(state)
            else:
                state.day = day
                state.start_of_day_equity = start_of_day_equity
                state.peak_equity = peak_equity
            session.commit()
        finally:
            session.close()

    # ------------------------------------------------------------------
    # LEGACY / DIRECT LOGGING
    # ------------------------------------------------------------------

    def log_trade(self, symbol, side, quantity, entry_price, exit_price,
                  entry_time, exit_time, fees=0.0, strategy="swing_v1"):
        session = self.Session()
        try:
            pnl = (exit_price - entry_price) * quantity
            if side == "SHORT":
                pnl = -pnl

            trade = Trade(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=pnl,
                fees=fees,
                entry_time=entry_time,
                exit_time=exit_time,
                strategy=strategy,
            )
            session.add(trade)
            session.commit()
        finally:
            session.close()

    def log_account_snapshot(self, net_liquidation, cash):
        session = self.Session()
        try:
            snapshot = AccountSnapshot(
                net_liquidation=net_liquidation,
                cash=cash,
            )
            session.add(snapshot)
            session.commit()
        finally:
            session.close()