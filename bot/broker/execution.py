from ib_async import Stock, MarketOrder, StopOrder, LimitOrder
from bot.broker.ibkr_client import IBKRClient
from bot.broker.fx import FXConverter
from bot.risk.position_sizing import calculate_position_size


class OrderExecutor:
    def __init__(self, ib_client: IBKRClient, risk_manager):
        self.ib_client = ib_client
        self.risk_manager = risk_manager
        self.ib = ib_client.ib
        self.fx = FXConverter(self.ib)

    def _next_order_id(self):
        return self.ib.client.getReqId()

    def place_bracket_order(
        self,
        symbol,
        entry_price,
        stop_price,
        take_profit_price,
        equity,
        instrument_currency="USD",
        risk_per_trade_pct=0.015,
    ):
        if not self.risk_manager.check_trading_allowed(equity):
            print("BLOCKED: Risk manager denies trading.")
            return None

        existing_position = self.ib_client.get_position(symbol)
        if existing_position != 0:
            print(f"BLOCKED: Already holding position in {symbol} ({existing_position} shares).")
            return None

        account_currency = self.ib_client.get_account_currency()

        equity_converted = self.fx.convert(
            amount=equity,
            from_currency=account_currency,
            to_currency=instrument_currency,
        )

        print(f"Equity: {equity:.2f} {account_currency} -> {equity_converted:.2f} {instrument_currency}")

        position_size = calculate_position_size(
            equity=equity_converted,
            entry_price=entry_price,
            stop_price=stop_price,
            risk_per_trade_pct=risk_per_trade_pct,
        )

        if position_size <= 0:
            print("BLOCKED: Calculated position size is 0.")
            return None

        contract = Stock(symbol, "SMART", instrument_currency)
        self.ib.qualifyContracts(contract)

        parent_id = self._next_order_id()

        parent = MarketOrder("BUY", position_size)
        parent.orderId = parent_id
        parent.transmit = False
        parent.tif = "GTC"
        parent.outsideRth = False

        stop = StopOrder("SELL", position_size, round(stop_price, 2))
        stop.parentId = parent_id
        stop.orderId = parent_id + 1
        stop.transmit = False
        stop.tif = "GTC"
        stop.outsideRth = False

        target = LimitOrder("SELL", position_size, round(take_profit_price, 2))
        target.parentId = parent_id
        target.orderId = parent_id + 2
        target.transmit = True
        target.tif = "GTC"
        target.outsideRth = False

        self.ib.placeOrder(contract, parent)
        self.ib.placeOrder(contract, stop)
        self.ib.placeOrder(contract, target)

        self.ib.sleep(2)

        return {
            "symbol": symbol,
            "quantity": position_size,
            "entry": entry_price,
            "stop": stop_price,
            "target": take_profit_price,
            "parent_id": parent_id,
        }