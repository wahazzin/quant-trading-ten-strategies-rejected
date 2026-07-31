def calculate_position_size(
    equity,
    entry_price,
    stop_price,
    risk_per_trade_pct=0.015,
    max_position_pct=0.20,
):
    if entry_price <= 0 or stop_price <= 0 or equity <= 0:
        return 0

    risk_amount = equity * risk_per_trade_pct
    risk_per_share = abs(entry_price - stop_price)

    if risk_per_share == 0:
        return 0

    risk_based_size = risk_amount / risk_per_share

    max_position_value = equity * max_position_pct
    value_based_size = max_position_value / entry_price

    affordability_size = equity / entry_price

    final_size = min(risk_based_size, value_based_size, affordability_size)

    return int(final_size)