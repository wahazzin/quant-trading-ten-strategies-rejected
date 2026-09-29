"""
features.py -- turns raw candles + live quotes into the compact feature set the AI (and the
baselines) see. Also builds the per-cycle SNAPSHOT that every arm shares.

CONCEPTS:
  * mid price = (bid + ask) / 2, the fair "middle" between what buyers offer and sellers ask.
  * spread_bps = (ask - bid) / mid in basis points (1 bp = 0.01%). A round trip pays it.
  * annualised volatility = std of log returns scaled to a year (crypto trades 24/7, so
    hourly x sqrt(24*365), daily x sqrt(365)).
  * volume_ratio > 1 means the last 24h traded more than the recent daily norm.
  * RSI(14) is included as ONE input among many. It is not the strategy.

Strictness on purpose: ANY missing/stale/odd data raises DataFault and the whole cycle is
skipped. With 8 assets and retries this is rare, and "skip and log" is always safer than
"fill in a plausible number".
"""
import math

import numpy as np

from crypto_ai.market_data.coinbase import DataFault, closed_only


def _rsi(closes, n=14):
    d = np.diff(closes)
    gain, loss = np.clip(d, 0, None), np.clip(-d, 0, None)
    ag, al = gain[:n].mean(), loss[:n].mean()
    for g, l in zip(gain[n:], loss[n:]):        # Wilder smoothing
        ag = (ag * (n - 1) + g) / n
        al = (al * (n - 1) + l) / n
    return 100.0 if al == 0 else 100 - 100 / (1 + ag / al)


def compute_features(tk, hourly, daily, btc_daily_logret):
    if len(hourly) < 170 or len(daily) < 60:
        raise DataFault(f"not enough closed candles (hourly={len(hourly)}, daily={len(daily)})")
    mid = (tk["bid"] + tk["ask"]) / 2
    hc = np.array([c["close"] for c in hourly])
    hv = np.array([c["volume"] for c in hourly])
    dc = np.array([c["close"] for c in daily])
    dv = np.array([c["volume"] for c in daily])
    dh = np.array([c["high"] for c in daily])
    hlog = np.diff(np.log(hc[-169:]))
    dlog = np.diff(np.log(dc[-31:]))
    prior7 = dv[-8:-1].mean()
    corr = None
    if btc_daily_logret is not None and len(btc_daily_logret) == len(dlog):
        corr = float(np.corrcoef(dlog, btc_daily_logret)[0, 1])
    f = {
        "ret_1h": hc[-1] / hc[-2] - 1, "ret_6h": hc[-1] / hc[-7] - 1,
        "ret_24h": hc[-1] / hc[-25] - 1, "ret_7d": hc[-1] / hc[-169] - 1,
        "ret_30d": dc[-1] / dc[-31] - 1,
        "vol_7d_ann": float(np.std(hlog, ddof=1) * math.sqrt(24 * 365)),
        "vol_30d_ann": float(np.std(dlog, ddof=1) * math.sqrt(365)),
        "volume_ratio": float(hv[-24:].sum() / prior7) if prior7 > 0 else None,
        "dist_sma20": mid / dc[-20:].mean() - 1, "dist_sma50": mid / dc[-50:].mean() - 1,
        "drawdown_from_30d_high": mid / dh[-30:].max() - 1,
        "rsi14_daily": float(_rsi(dc[-120:])),
        "corr_btc_30d": corr,
        "sma50": float(dc[-50:].mean()), "daily_close": float(dc[-1]),
    }
    return {k: (None if v is None else round(float(v), 6)) for k, v in f.items()}


def build_snapshot(cfg, client, now):
    """Returns (snapshot, hourly_by_asset). hourly candles are kept in memory for stop replay
    and are not persisted (they are public and refetchable)."""
    now_ts = int(now.timestamp())
    max_age = cfg["data"]["max_quote_age_seconds"]
    uni = cfg["universe"]
    tick, hourly, daily = {}, {}, {}
    for p in uni:
        tk = client.ticker(p)
        age = (now - tk["time"]).total_seconds()
        if abs(age) > max_age:
            raise DataFault(f"{p}: quote age {age:.0f}s exceeds {max_age}s")
        if not (0 < tk["bid"] <= tk["ask"]):
            raise DataFault(f"{p}: crossed/invalid book bid={tk['bid']} ask={tk['ask']}")
        tick[p] = tk
        hourly[p] = closed_only(client.candles(p, 3600), 3600, now_ts)[-cfg["data"]["hourly_candles"]:]
        daily[p] = closed_only(client.candles(p, 86400), 86400, now_ts)[-cfg["data"]["daily_candles"]:]

    btc = "BTC-USD"
    btc_dc = np.array([c["close"] for c in daily[btc]])
    btc_lr = np.diff(np.log(btc_dc[-31:]))
    assets = {}
    for p in uni:
        tk = tick[p]
        mid = (tk["bid"] + tk["ask"]) / 2
        assets[p] = {
            "mid": mid, "bid": tk["bid"], "ask": tk["ask"],
            "spread_bps": round((tk["ask"] - tk["bid"]) / mid * 1e4, 4),
            "volume_24h_usd": round(tk["volume_base"] * mid, 2),
            "quote_age_s": round((now - tk["time"]).total_seconds(), 1),
            "features": compute_features(tk, hourly[p], daily[p], btc_lr),
        }
    return {"assets": assets}, hourly


def fresh_quotes(cfg, client, assets, now):
    """Post-decision quotes used to price fills. Returns {asset: quote or None}."""
    out = {}
    for p in assets:
        try:
            tk = client.ticker(p)
            if abs((now - tk["time"]).total_seconds()) > cfg["data"]["max_quote_age_seconds"] \
                    or not (0 < tk["bid"] <= tk["ask"]):
                out[p] = None
            else:
                mid = (tk["bid"] + tk["ask"]) / 2
                out[p] = {"mid": mid, "bid": tk["bid"], "ask": tk["ask"],
                          "volume_24h_usd": tk["volume_base"] * mid}
        except DataFault:
            out[p] = None
    return out
