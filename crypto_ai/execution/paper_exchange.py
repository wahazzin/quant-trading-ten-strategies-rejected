"""
paper_exchange.py -- a simulated exchange that refuses to give perfect fills.

CONCEPTS (each one is a way a naive backtest lies to you):
  * SPREAD: you buy at the ASK and sell at the BID, never at the mid. Crossing the spread is a
    real cost on every trade.
  * FEE: the exchange charges a percentage of every fill (40 bps = 0.40% per side by default).
  * SLIPPAGE: your order moves the price a little against you; bigger orders relative to daily
    volume move it more. Modelled as a small base (2 bps majors, 6 bps alts) plus a size term.
  * LATENCY: fills are priced from a quote fetched AFTER the LLM answered, not from the price
    the AI was shown.
  * STOPS: a stop-loss is only as good as the market lets it be. If price gaps through the stop,
    you get the (worse) open, not your stop price. stop replay walks hourly candles that closed
    since the last cycle and fills accordingly.

Every fill records its cost split (fee / spread / slippage) so you can later ask "how much did
costs eat?" -- the question that killed most of your equities strategies.
"""
from datetime import datetime, timezone

from crypto_ai.journal import iso, parse_iso
from crypto_ai.portfolio.state import Position


def slippage_bps(cfg, asset, notional, volume_24h_usd):
    c = cfg["costs"]
    base = c["slippage_bps_major"] if asset in cfg["majors"] else c["slippage_bps_alt"]
    impact = c["impact_coeff"] * (notional / volume_24h_usd) * 1e4 if volume_24h_usd > 0 else 0.0
    return base + impact


def _fill_row(pf, cycle_id, ts, asset, side, qty, price, mid, fee, spread_cost, slip_cost, cause):
    return {"ts": iso(ts), "cycle_id": cycle_id, "arm": pf.name, "asset": asset, "side": side,
            "qty": qty, "price": price, "mid": mid, "notional": qty * price,
            "fee_usd": fee, "spread_cost_usd": spread_cost, "slippage_cost_usd": slip_cost,
            "cause": cause}


def _finish(pf, ts, gross, fee, spread_cost, slip_cost, cause):
    pf.fees_usd += fee
    pf.spread_usd += spread_cost
    pf.slippage_usd += slip_cost
    pf.record_trade(ts, gross, cause)


def execute(pf, asset, side, notional_usd, quote, cfg, now, cause, cycle_id="",
            stop_pct=None, full_exit=False, fee_bps=None):
    """Market order. `notional_usd` is the desired value at mid. Returns a fill row or None.
    Buys are scaled DOWN to available cash (never overdrawn). Sells never exceed holdings."""
    fee_rate = (cfg["costs"]["fee_bps"] if fee_bps is None else fee_bps) / 1e4
    mid, bid, ask = quote["mid"], quote["bid"], quote["ask"]
    slip = slippage_bps(cfg, asset, notional_usd, quote.get("volume_24h_usd", 0.0)) / 1e4

    if side == "buy":
        fill_px = ask * (1 + slip)
        qty = min(notional_usd / mid, pf.cash / (fill_px * (1 + fee_rate)))
        if qty * mid < 1.0:
            return None
        gross = qty * fill_px
        fee = gross * fee_rate
        pf.cash -= gross + fee
        sp = cfg["risk"]["max_stop_pct"] if stop_pct is None else stop_pct
        pos = pf.positions.get(asset)
        if pos is None:
            pf.positions[asset] = Position(qty, fill_px, sp, iso(now), cycle_id)
        else:
            total = pos.qty + qty
            pos.avg_cost = (pos.avg_cost * pos.qty + fill_px * qty) / total
            pos.qty = total
            pos.stop_pct = min(pos.stop_pct, sp)          # tighten only, never loosen
        spread_cost, slip_cost = qty * (ask - mid), qty * ask * slip
    elif side == "sell":
        pos = pf.positions.get(asset)
        if pos is None or pos.qty <= 0:
            return None
        qty = pos.qty if full_exit else min(pos.qty, notional_usd / mid)
        if qty >= pos.qty * (1 - 1e-9):
            qty = pos.qty
        fill_px = bid * (1 - slip)
        gross = qty * fill_px
        fee = gross * fee_rate
        pf.cash += gross - fee
        pos.qty -= qty
        if pos.qty <= 1e-12:
            del pf.positions[asset]
        spread_cost, slip_cost = qty * (mid - bid), qty * bid * slip
    else:
        raise ValueError(side)

    _finish(pf, now, gross, fee, spread_cost, slip_cost, cause)
    return _fill_row(pf, cycle_id, now, asset, side, qty, fill_px, mid, fee, spread_cost, slip_cost, cause)


def replay_stops(pf, hourly_by_asset, snapshot_assets, cfg, now, cycle_id="", since_ts=None):
    """Walk hourly candles (already closed) and trigger stops. Returns a list of fill rows.
    since_ts (unix seconds): only candles that CLOSED after this time are checked, i.e. candles
    not already examined by the previous cycle. Without it, tightening a stop could be triggered
    retroactively by an old candle -- a subtle look-ahead."""
    fee_rate = cfg["costs"]["fee_bps"] / 1e4
    fills = []
    for asset in list(pf.positions):
        pos = pf.positions[asset]
        opened = parse_iso(pos.opened_ts).timestamp()
        stop = pos.stop_price
        info = snapshot_assets[asset]
        for c in hourly_by_asset.get(asset, []):
            if c["t"] < opened:
                continue                                   # candle began before we owned it
            if since_ts is not None and c["t"] + 3600 <= since_ts:
                continue                                   # already checked last cycle
            if c["open"] <= stop:
                ref = c["open"]                            # gapped through the stop
            elif c["low"] <= stop:
                ref = stop
            else:
                continue
            half_spread = info["spread_bps"] / 2 / 1e4
            slip = slippage_bps(cfg, asset, pos.qty * ref, info["volume_24h_usd"]) / 1e4
            fill_px = ref * (1 - half_spread - slip)
            qty = pos.qty
            gross = qty * fill_px
            fee = gross * fee_rate
            pf.cash += gross - fee
            del pf.positions[asset]
            ts = min(now, datetime.fromtimestamp(c["t"] + 3600, tz=timezone.utc))
            _finish(pf, ts, gross, fee, qty * ref * half_spread, qty * ref * slip, "stop")
            row = _fill_row(pf, cycle_id, ts, asset, "sell", qty, fill_px, ref, fee,
                            qty * ref * half_spread, qty * ref * slip, "stop")
            row["stop_price"] = stop
            row["stop_candle_start"] = c["t"]
            fills.append(row)
            break
    return fills
