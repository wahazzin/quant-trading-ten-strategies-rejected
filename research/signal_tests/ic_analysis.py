"""
ic_analysis.py -- Information Coefficient (IC) diagnostic scan.

Both prior signal tests (SMA crossover, RSI mean-reversion) foundered on
the same problem: only ~71 discrete trades is far too few to say
anything statistically. This script switches to a diagnostic question
-- "is there ANY predictive information in these signals at all?" --
answered using EVERY daily observation (~thousands per signal x horizon)
instead of only trade-entry days. It builds no strategy, selects no
best signal, and tunes nothing.

Methodology (self-contained, no dependency on the strategy scripts):
  - 8 candidate signals x 4 forward-return horizons = 32 combinations.
  - For each combination, the Spearman rank correlation between the
    signal value and the forward return is computed SEPARATELY per
    stock (using that stock's own ~500+ daily observations). This
    gives 13 per-stock IC values.
  - Those 13 values are then summarized: mean IC, std IC, and a
    one-sample t-test of whether the average IC differs from zero
    across stocks (t = mean / (std / sqrt(13))) -- this is the
    standard way IC significance is tested in factor research (e.g.
    Grinold & Kahn's IC framework): treat each independent slice (here,
    each stock) as one sample and test the cross-section of ICs.
  - The total pooled observation count (~thousands of stock-days) is
    reported alongside for context, even though the t-test itself uses
    the 13-stock cross-section as its sample size.
  - The per-stock IC breakdown is also printed in full, so a result
    driven by one or two stocks is visible rather than hidden inside
    an average.

Only the 70% practice window is ever read; the 30% holdout is never
touched.
"""
import os
import numpy as np
import pandas as pd

UNIVERSE = ["SBRA", "VLY", "FLO", "AROC", "HUN", "WEN",
            "CLF", "MGNI", "KSS", "TROX", "VSH", "UAA", "HL"]
TUNE = 0.70
HORIZONS = [1, 5, 10, 20]
MIN_STOCK_OBS = 20   # defensive floor; with ~500 practice days this never binds


# ------------------------------------------------------------------
# Wilder smoothing (shared recursive-average machinery for RSI & ATR)
# ------------------------------------------------------------------
def _wilder_smooth(values, period):
    """values[0] must be NaN/unused (e.g. a first difference with no
    prior bar); returns the Wilder-smoothed series, NaN until warmup."""
    n = len(values)
    out = np.full(n, np.nan)
    if n <= period:
        return out
    seed = np.nanmean(values[1:period + 1])
    out[period] = seed
    prev = seed
    for i in range(period + 1, n):
        prev = (prev * (period - 1) + values[i]) / period
        out[i] = prev
    return out


def compute_rsi(close, period=14):
    n = len(close)
    delta = np.full(n, np.nan)
    delta[1:] = np.diff(close)
    gain = np.where(np.nan_to_num(delta, nan=0.0) > 0, np.nan_to_num(delta, nan=0.0), 0.0)
    loss = np.where(np.nan_to_num(delta, nan=0.0) < 0, -np.nan_to_num(delta, nan=0.0), 0.0)
    gain[0], loss[0] = np.nan, np.nan
    avg_gain = _wilder_smooth(gain, period)
    avg_loss = _wilder_smooth(loss, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        rsi = 100 - 100 / (1 + rs)
    rsi = np.where(avg_loss == 0, np.where(avg_gain > 0, 100.0, 50.0), rsi)
    rsi = np.where(np.isnan(avg_gain) | np.isnan(avg_loss), np.nan, rsi)
    return rsi


def compute_atr_pct(high, low, close, period=14):
    n = len(close)
    prev_close = np.full(n, np.nan)
    prev_close[1:] = close[:-1]
    tr = np.full(n, np.nan)
    tr[1:] = np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - prev_close[1:]),
        np.abs(low[1:] - prev_close[1:]),
    ])
    atr = _wilder_smooth(tr, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        atr_pct = atr / close
    return atr_pct


def compute_signals(df):
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    closeS = df["close"]
    volumeS = df["volume"]

    ma20 = closeS.rolling(20).mean().to_numpy()
    std20 = closeS.rolling(20).std().to_numpy()
    ma50 = closeS.rolling(50).mean().to_numpy()
    std50 = closeS.rolling(50).std().to_numpy()
    vol_avg20 = volumeS.rolling(20).mean().to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        z20 = np.where(std20 > 0, (close - ma20) / std20, np.nan)
        z50 = np.where(std50 > 0, (close - ma50) / std50, np.nan)
        vol_ratio = np.where(vol_avg20 > 0, volume / vol_avg20, np.nan)

        mom20 = np.full(len(close), np.nan)
        mom20[20:] = close[20:] / close[:-20] - 1
        mom60 = np.full(len(close), np.nan)
        mom60[60:] = close[60:] / close[:-60] - 1
        ret5 = np.full(len(close), np.nan)
        ret5[5:] = close[5:] / close[:-5] - 1

    return {
        "RSI14": compute_rsi(close, 14),
        "Zscore_MA20": z20,
        "Zscore_MA50": z50,
        "Mom20": mom20,
        "Mom60": mom60,
        "VolRatio20": vol_ratio,
        "Ret5": ret5,
        "ATR14_pct": compute_atr_pct(high, low, close, 14),
    }


def forward_return(opens, horizon):
    """fwd[N] = open[N+1+horizon] / open[N+1] - 1: signal known at day N's
    close predicts the return from day N+1's open to day N+1+horizon's open."""
    n = len(opens)
    fwd = np.full(n, np.nan)
    last_n = n - 1 - horizon   # last valid N is n - 2 - horizon
    if last_n <= 0:
        return fwd
    idx = np.arange(0, last_n)
    fwd[idx] = opens[idx + 1 + horizon] / opens[idx + 1] - 1
    return fwd


def spearman_corr(x, y):
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def load_tune_data():
    data = {}
    for ticker in UNIVERSE:
        path = os.path.join("data", f"{ticker}_daily.csv")
        if not os.path.exists(path):
            print(f"{ticker}: SKIPPED (missing CSV at {path})")
            continue
        df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        n_tune = int(len(df) * TUNE)
        data[ticker] = df.iloc[:n_tune].reset_index(drop=True)
    return data


DATA = load_tune_data()
SIGNAL_NAMES = ["RSI14", "Zscore_MA20", "Zscore_MA50", "Mom20", "Mom60",
                "VolRatio20", "Ret5", "ATR14_pct"]

stock_signals = {}
stock_fwd = {}
for ticker, tune_df in DATA.items():
    stock_signals[ticker] = compute_signals(tune_df)
    opens = tune_df["open"].to_numpy(dtype=float)
    stock_fwd[ticker] = {h: forward_return(opens, h) for h in HORIZONS}

results = []
per_stock_detail = {}

for sig_name in SIGNAL_NAMES:
    for h in HORIZONS:
        per_stock_ic = {}
        total_obs = 0
        for ticker in DATA:
            sig_arr = stock_signals[ticker][sig_name]
            fwd_arr = stock_fwd[ticker][h]
            mask = ~np.isnan(sig_arr) & ~np.isnan(fwd_arr)
            n_valid = int(mask.sum())
            if n_valid < MIN_STOCK_OBS:
                continue
            rho = spearman_corr(sig_arr[mask], fwd_arr[mask])
            if np.isnan(rho):
                continue
            per_stock_ic[ticker] = rho
            total_obs += n_valid

        ics = np.array(list(per_stock_ic.values()))
        k = len(ics)
        mean_ic = float(ics.mean()) if k > 0 else float("nan")
        std_ic = float(ics.std(ddof=1)) if k > 1 else float("nan")
        if k > 1 and std_ic > 0:
            t_stat = mean_ic / (std_ic / np.sqrt(k))
        else:
            t_stat = float("nan")

        results.append({
            "signal": sig_name, "horizon": h,
            "mean_ic": mean_ic, "std_ic": std_ic, "t_stat": t_stat,
            "n_obs": total_obs, "n_stocks": k,
        })
        per_stock_detail[(sig_name, h)] = per_stock_ic

# sort by |t-stat| descending; NaN t-stats (shouldn't occur with 13 stocks
# and ~500 obs each, but handled defensively) sort to the bottom
results.sort(key=lambda r: abs(r["t_stat"]) if not np.isnan(r["t_stat"]) else -1, reverse=True)

print()
print("=" * 88)
print("IC DIAGNOSTIC SCAN -- 8 signals x 4 horizons, practice window only, sorted by |t-stat|")
print("=" * 88)
header = f"{'Signal':<14}{'Horizon':>8}{'MeanIC':>10}{'StdIC':>9}{'t-stat':>9}{'nObs':>8}{'nStk':>6}"
print(header)
print("-" * len(header))
for r in results:
    print(
        f"{r['signal']:<14}{r['horizon']:>7}d"
        f"{r['mean_ic']:>10.4f}"
        f"{r['std_ic']:>9.4f}"
        f"{r['t_stat']:>9.2f}"
        f"{r['n_obs']:>8}"
        f"{r['n_stocks']:>6}"
    )

print()
print("=" * 88)
print("Per-stock IC breakdown (same order as above)")
print("=" * 88)
tickers = list(DATA.keys())
detail_header = f"{'Signal':<14}{'Horizon':>8}  " + "".join(f"{t:>8}" for t in tickers)
print(detail_header)
print("-" * len(detail_header))
for r in results:
    key = (r["signal"], r["horizon"])
    detail = per_stock_detail[key]
    row = f"{r['signal']:<14}{r['horizon']:>7}d  "
    for t in tickers:
        val = detail.get(t)
        row += f"{val:>8.3f}" if val is not None else f"{'--':>8}"
    print(row)

n_combos = len(results)
n_sig_3 = sum(1 for r in results if not np.isnan(r["t_stat"]) and abs(r["t_stat"]) > 3)
n_sig_2 = sum(1 for r in results if not np.isnan(r["t_stat"]) and abs(r["t_stat"]) > 2)

print()
print("=" * 88)
print("Interpretation")
print("=" * 88)
print(f"{n_combos} signal x horizon combinations tested.")
print(f"{n_sig_3} combo(s) exceed |t-stat| > 3 (a conservative bar given {n_combos} comparisons).")
print(f"{n_sig_2} combo(s) exceed |t-stat| > 2 -- note that with {n_combos} independent combinations, "
      f"roughly 1-2 would be expected to exceed |t| > 2 by chance alone even if no real signal exists.")
print("This is a diagnostic scan only: no strategy was built, no signal was selected or")
print("recommended, and no parameters were tuned.")
