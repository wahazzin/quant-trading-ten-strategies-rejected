"""
fundamental_test.py -- Test 10, the one untested hypothesis class: value
and quality, the two most replicated fundamental anomalies in finance.
Nine price/event-based strategies have already been rejected
(RESEARCH_LOG.md).

Two pre-registered factors, one parameterization each, no variants:
  Value   -- Book-to-Market = StockholdersEquity / (price * shares out).
             Long top decile (cheapest).
  Quality -- Gross Profitability = (Revenues - CostOfRevenue) / Assets,
             per Novy-Marx 2013 (GrossProfit used directly when reported).
             Long top decile (most profitable).

Mechanics, different from factor_test.py's momentum/low-vol tests on
purpose: those signals are continuously recomputed and rebalanced every
month. Fundamentals are not -- a June 2015 Book-to-Market ratio is still
the best available estimate in, say, February 2016, because the next
10-K/proxy update hasn't been filed yet. So this ranks ANNUALLY each
June, using only fundamental facts already filed by that date, and holds
the resulting decile fixed from July through the following June (12
calendar months) before re-ranking. Monthly RETURNS are still measured
every month on the held, fixed portfolio -- that's what gives ~96
practice observations out of ~8 annual rebalances, instead of 8.

Reused verbatim from factor_test.py: monthly panel construction (close/
open aggregation, next_open/next2_open telescoping, eligibility filters),
perf_stats(), and alpha_regression(). The only structural change is the
factor engine loop, which assigns membership once a year instead of
every month -- everything downstream of "which stocks are held this
month" is identical.

CRITICAL -- lookahead protection: a fundamental fact is only usable from
its OWN 'filed' date onward, never its fiscal 'end' date. A Q4 2015
figure filed 2016-03-01 cannot be used to form the June 2015 decile, and
in fact can't be used until the *next* June (2016) formation, since June
2015 formation only sees facts filed by 2015-06-01. This is enforced by
filtering candidate facts on filed <= formation_date before selecting
the most-recent-by-end value, for every single company at every single
formation date -- there is no shortcut or global sort that could leak a
later filing into an earlier formation.

Evidence boundary: practice = filed/date < 2019-01-01, holdout >= 2019
sealed. Both the raw daily stock/SPY panels AND the raw fundamentals
table are truncated immediately after loading, before any other
computation, exactly as in factor_test.py.
"""
import os
import numpy as np
import pandas as pd

DATA_PATH = os.path.join("data", "yf_universe.parquet")
SPY_PATH = os.path.join("data", "spy_yf.parquet")
FUND_PATH = os.path.join("data", "edgar_fundamentals.parquet")
HOLDOUT_START = pd.Timestamp("2019-01-01")

PRICE_MIN, PRICE_MAX = 5.0, 100.0
MIN_ADV = 250_000
MIN_HISTORY_MONTHS = 24
COST_LEVELS = [0.001, 0.002, 0.004]
GAP_DAYS_THRESHOLD = 10
MOVE_THRESHOLD = 0.80
MIN_UNIVERSE_FOR_DECILE = 20
MIN_DECILE_SIZE_TRUSTED = 25

ACCOUNT_EQUITY = 4700.0
COMMISSION_PER_ORDER = 1.0
ANNUAL_DURATION_MIN_DAYS = 340
ANNUAL_DURATION_MAX_DAYS = 386


# ============================================================
# LOAD + HOLDOUT BOUNDARY
# ============================================================
raw = pd.read_parquet(DATA_PATH)
raw["date"] = pd.to_datetime(raw["date"])
raw_spy = pd.read_parquet(SPY_PATH)
raw_spy["date"] = pd.to_datetime(raw_spy["date"])
raw_fund = pd.read_parquet(FUND_PATH)

print("=" * 96)
print("HOLDOUT CONFIRMATION")
print("=" * 96)
print(f"Raw stock panel on disk spans: {raw['date'].min().date()} to {raw['date'].max().date()} "
      f"({len(raw)} rows, {raw['ticker'].nunique()} tickers)")
print(f"Raw SPY panel on disk spans: {raw_spy['date'].min().date()} to {raw_spy['date'].max().date()} "
      f"({len(raw_spy)} rows)")
print(f"Raw fundamentals table on disk spans filed dates: {raw_fund['filed'].min().date()} to "
      f"{raw_fund['filed'].max().date()} ({len(raw_fund)} rows, {raw_fund['ticker'].nunique()} tickers)")

practice = raw[raw["date"] < HOLDOUT_START].copy()
n_dropped_holdout = len(raw) - len(practice)
del raw
spy_practice = raw_spy[raw_spy["date"] < HOLDOUT_START].copy()
n_dropped_spy = len(raw_spy) - len(spy_practice)
del raw_spy
fund_practice = raw_fund[raw_fund["filed"] < HOLDOUT_START].copy()
n_dropped_fund = len(raw_fund) - len(fund_practice)
del raw_fund   # all three untruncated references are gone

print(f"Practice window kept: date/filed < {HOLDOUT_START.date()}  "
      f"(stocks: {len(practice)} rows, SPY: {len(spy_practice)} rows, fundamentals: {len(fund_practice)} rows)")
print(f"Rows excluded as holdout: stocks {n_dropped_holdout}, SPY {n_dropped_spy}, fundamentals {n_dropped_fund}")
print(f"Maximum date present in any DataFrame from this point forward: "
      f"{max(practice['date'].max(), spy_practice['date'].max(), fund_practice['filed'].max()).date()}")
print("CONFIRMED: the post-2018 holdout was not read by this script beyond the initial date checks above.")

df = practice.sort_values(["ticker", "date"]).reset_index(drop=True)
spy = spy_practice.sort_values("date").reset_index(drop=True)
fund = fund_practice.sort_values(["ticker", "concept", "end", "filed"]).reset_index(drop=True)


# ============================================================
# DATA QUALITY CHECKS (practice window only) -- verbatim from factor_test.py
# ============================================================
print()
print("=" * 96)
print("DATA QUALITY CHECKS (practice window only)")
print("=" * 96)

n0 = len(df)
bad_price = (df[["open", "high", "low", "close"]] <= 0).any(axis=1) | df["volume"].lt(0)
n_bad_price = int(bad_price.sum())
df = df[~bad_price].copy()

df["prev_close"] = df.groupby("ticker")["close"].shift(1)
df["daily_ret"] = df["close"] / df["prev_close"] - 1
big_move = df["daily_ret"].abs() > MOVE_THRESHOLD
n_big_move = int(big_move.fillna(False).sum())
df = df[~big_move.fillna(False)].copy()

df["prev_date"] = df.groupby("ticker")["date"].shift(1)
df["gap_days"] = (df["date"] - df["prev_date"]).dt.days
n_long_gaps = int((df["gap_days"] > GAP_DAYS_THRESHOLD).sum())

print(f"Rows before quality checks: {n0}")
print(f"Dropped -- zero/negative price or negative volume: {n_bad_price}")
print(f"Dropped -- single-day move > {MOVE_THRESHOLD*100:.0f}% (possible unadjusted split / bad tick): {n_big_move}")
print(f"Flagged, not dropped -- gaps > {GAP_DAYS_THRESHOLD} calendar days between consecutive rows: {n_long_gaps}")
print(f"Rows remaining: {len(df)} ({n0 - len(df)} dropped total, "
      f"{(n0 - len(df)) / n0 * 100:.3f}% of pre-check rows)")

df = df.drop(columns=["prev_close", "daily_ret", "prev_date", "gap_days"])


# ============================================================
# BUILD MONTHLY PANEL (stocks) -- verbatim from factor_test.py
# ============================================================
df["month"] = df["date"].dt.to_period("M")

close_last = df.sort_values("date").groupby(["ticker", "month"])["close"].last()
open_first = df.sort_values("date").groupby(["ticker", "month"])["open"].first()
vol_mean = df.groupby(["ticker", "month"])["volume"].mean()

monthly = pd.DataFrame({"close": close_last, "open": open_first, "avg_volume": vol_mean}).reset_index()
monthly = monthly.sort_values(["ticker", "month"]).reset_index(drop=True)

g = monthly.groupby("ticker")
monthly["next_open"] = g["open"].shift(-1)
monthly["next_month"] = g["month"].shift(-1)
monthly["next2_open"] = g["open"].shift(-2)
monthly["next2_month"] = g["month"].shift(-2)
monthly["ret"] = g["close"].pct_change()
monthly["hist_months"] = g.cumcount()

valid_next = monthly["next_month"] == (monthly["month"] + 1)
valid_next2 = monthly["next2_month"] == (monthly["month"] + 2)
monthly.loc[~valid_next, "next_open"] = np.nan
monthly.loc[~valid_next2, "next2_open"] = np.nan

monthly["eligible"] = (
    (monthly["close"] >= PRICE_MIN) & (monthly["close"] <= PRICE_MAX) &
    (monthly["avg_volume"] > MIN_ADV) &
    (monthly["hist_months"] >= MIN_HISTORY_MONTHS)
)

all_months = sorted(monthly["month"].unique())
avg_universe_size = monthly.loc[monthly["eligible"], "month"].value_counts().mean()

print()
print("=" * 96)
print("UNIVERSE")
print("=" * 96)
print(f"Monthly panel: {len(monthly)} (ticker, month) rows, {monthly['ticker'].nunique()} tickers, "
      f"{len(all_months)} months ({all_months[0]} to {all_months[-1]})")
print(f"Filters: close ${PRICE_MIN}-${PRICE_MAX}, monthly avg daily volume > {MIN_ADV:,}, "
      f"{MIN_HISTORY_MONTHS}+ months prior history")
print(f"Average eligible universe size per month: {avg_universe_size:.1f} "
      f"(implied average decile size: {avg_universe_size/10:.1f})")

monthly_idx = monthly.set_index(["ticker", "month"])


# ============================================================
# BUILD SPY MONTHLY RETURN SERIES -- verbatim from factor_test.py
# ============================================================
spy["month"] = spy["date"].dt.to_period("M")
spy_close_last = spy.groupby("month")["close"].last().reset_index().sort_values("month").reset_index(drop=True)
spy_close_last["spy_ret"] = spy_close_last["close"].pct_change()
SPY_MONTHLY = spy_close_last.set_index("month")["spy_ret"]


# ============================================================
# POINT-IN-TIME FUNDAMENTAL SELECTION
# ============================================================
INSTANT_CONCEPTS = ["Assets", "StockholdersEquity", "CommonStockSharesOutstanding"]
FLOW_CONCEPTS = ["Revenues", "CostOfRevenue", "GrossProfit", "NetIncomeLoss"]

fund_instant = fund[fund["concept"].isin(INSTANT_CONCEPTS)].copy()

flow = fund[fund["concept"].isin(FLOW_CONCEPTS)].copy()
duration_days = (flow["end"] - flow["start"]).dt.days
is_annual = duration_days.between(ANNUAL_DURATION_MIN_DAYS, ANNUAL_DURATION_MAX_DAYS)
n_flow_before = len(flow)
fund_flow_annual = flow[is_annual].copy()
print()
print("=" * 96)
print("FUNDAMENTAL FACT FILTERING")
print("=" * 96)
print(f"Flow concept rows (Revenues/CostOfRevenue/GrossProfit/NetIncomeLoss): {n_flow_before}")
print(f"Kept as annual-duration facts ({ANNUAL_DURATION_MIN_DAYS}-{ANNUAL_DURATION_MAX_DAYS} days, "
      f"i.e. fiscal-year figures, not quarterly YTD partials): {len(fund_flow_annual)}")
print(f"Instant concept rows (Assets/StockholdersEquity/CommonStockSharesOutstanding): {len(fund_instant)}")

fund_pit = pd.concat([fund_instant, fund_flow_annual], ignore_index=True)
fund_pit = fund_pit.sort_values(["ticker", "concept", "end"]).reset_index(drop=True)


def latest_known_value(ticker_facts_by_concept, concept, cutoff):
    """Most recent fiscal-period value for `concept` whose filed date is
    <= cutoff. Facts are pre-sorted by end ascending, so the last row
    passing the filed<=cutoff mask is both the most-recently-filed AND
    (almost always) the most recent fiscal period -- the two agree
    because filings become public in the same order their periods end."""
    sub = ticker_facts_by_concept.get(concept)
    if sub is None:
        return None
    known = sub[sub["filed"] <= cutoff]
    if known.empty:
        return None
    return float(known["val"].iloc[-1])


# Pre-group once per ticker for speed: {ticker: {concept: sub-dataframe sorted by end}}
facts_by_ticker = {}
for ticker, tdf in fund_pit.groupby("ticker"):
    facts_by_ticker[ticker] = {c: cdf for c, cdf in tdf.groupby("concept")}


# ============================================================
# ANNUAL FORMATION DATES (each June)
# ============================================================
june_months = [m for m in all_months if m.month == 6]
print()
print(f"Candidate June formation dates in practice window: {len(june_months)} "
      f"({june_months[0] if june_months else 'n/a'} to {june_months[-1] if june_months else 'n/a'})")


def build_formations(value_col_fn, long_is_high_signal=True):
    """value_col_fn(ticker, formation_date) -> (signal_value_or_None, extra_dict)
    Returns a DataFrame: one row per (formation_month, ticker) held stock,
    with the formation's decile membership and the signal value."""
    records = []
    for m in june_months:
        formation_date = pd.Timestamp(m.end_time.date())  # last calendar day of June
        month_df = monthly[(monthly["month"] == m) & monthly["eligible"]].copy()
        if month_df.empty:
            continue

        signals = []
        for row in month_df.itertuples(index=False):
            ticker_facts = facts_by_ticker.get(row.ticker)
            if ticker_facts is None:
                continue
            sig = value_col_fn(row.ticker, ticker_facts, formation_date, row.close)
            if sig is not None and np.isfinite(sig):
                signals.append((row.ticker, sig))

        if len(signals) < MIN_UNIVERSE_FOR_DECILE:
            continue

        sig_df = pd.DataFrame(signals, columns=["ticker", "signal"]).sort_values("signal")
        decile_n = max(1, len(sig_df) // 10)
        long_tickers = sig_df.iloc[-decile_n:]["ticker"] if long_is_high_signal else sig_df.iloc[:decile_n]["ticker"]

        records.append({
            "formation_month": m,
            "n_universe": len(sig_df),
            "n_long": len(long_tickers),
            "long_tickers": set(long_tickers),
        })

    return records


def value_signal(ticker, ticker_facts, cutoff, price):
    be = latest_known_value(ticker_facts, "StockholdersEquity", cutoff)
    shares = latest_known_value(ticker_facts, "CommonStockSharesOutstanding", cutoff)
    if be is None or shares is None or shares <= 0 or price <= 0 or be <= 0:
        return None  # negative/zero book equity excluded -- standard B/M convention
    market_cap = price * shares
    return be / market_cap


def quality_signal(ticker, ticker_facts, cutoff, price):
    assets = latest_known_value(ticker_facts, "Assets", cutoff)
    if assets is None or assets <= 0:
        return None
    gp = latest_known_value(ticker_facts, "GrossProfit", cutoff)
    if gp is None:
        rev = latest_known_value(ticker_facts, "Revenues", cutoff)
        cor = latest_known_value(ticker_facts, "CostOfRevenue", cutoff)
        if rev is None or cor is None:
            return None
        gp = rev - cor
    return gp / assets


# ============================================================
# HOLDING-PERIOD RETURN ENGINE
# ============================================================
def run_annual_factor(formations):
    """Expand each annual formation into 12 monthly holding-period
    records (or fewer at the tail, where practice-window truncation
    or panel end cuts a holding year short)."""
    period_records = []
    prev_weights = pd.Series(dtype=float)

    for f in formations:
        start_month = f["formation_month"] + 1  # hold from July of formation year
        long_tickers = f["long_tickers"]
        n_long = f["n_long"]

        w_new = pd.Series(1.0 / n_long, index=sorted(long_tickers))
        idx = w_new.index.union(prev_weights.index)
        turnover = 0.5 * float((w_new.reindex(idx, fill_value=0.0) -
                                 prev_weights.reindex(idx, fill_value=0.0)).abs().sum())
        prev_weights = w_new

        for k in range(12):
            hold_month = start_month + k
            if hold_month not in all_months:
                break
            month_rows = monthly[(monthly["month"] == hold_month) & monthly["ticker"].isin(long_tickers)]
            rets = month_rows.set_index("ticker")["ret"].reindex(sorted(long_tickers))
            valid = rets.dropna()
            if valid.empty:
                continue

            bench_rows = monthly[(monthly["month"] == hold_month) & monthly["eligible"]]
            bench_ret = bench_rows["ret"].mean()

            period_records.append({
                "formation_month": f["formation_month"],
                "month": hold_month,
                "is_rebalance_month": (k == 0),
                "n_long": n_long,
                "n_universe": f["n_universe"],
                "port_ret": float(valid.mean()),
                "bench_ret": float(bench_ret) if pd.notna(bench_ret) else np.nan,
                "turnover": turnover if k == 0 else 0.0,
            })

    res = pd.DataFrame(period_records)
    if len(res):
        res["spy_ret"] = res["month"].map(SPY_MONTHLY)
    return res


def perf_stats(period_returns, turnovers=None, cost=0.0):
    if turnovers is None:
        turnovers = np.zeros(len(period_returns))
    net = period_returns - turnovers * cost
    equity = np.cumprod(1 + net)
    n = len(net)
    total_return = float(equity[-1] - 1)
    annualized = float(equity[-1] ** (12.0 / n) - 1) if n > 0 else float("nan")
    running_max = np.maximum.accumulate(equity)
    max_dd = float((equity / running_max - 1).min())
    std = net.std(ddof=1)
    sharpe = float(net.mean() / std * np.sqrt(12)) if std > 0 else float("nan")
    return {"total_return": total_return, "annualized": annualized, "sharpe": sharpe, "max_dd": max_dd}


def alpha_regression(port_ret, spy_ret):
    mask = ~np.isnan(port_ret) & ~np.isnan(spy_ret)
    X = spy_ret[mask]
    Y = port_ret[mask]
    n = len(X)
    beta = float(np.cov(X, Y, ddof=1)[0, 1] / np.var(X, ddof=1))
    alpha_m = float(Y.mean() - beta * X.mean())
    resid = Y - (alpha_m + beta * X)
    ssr = float(np.sum(resid ** 2))
    sst = float(np.sum((Y - Y.mean()) ** 2))
    r2 = 1 - ssr / sst if sst > 0 else float("nan")
    sigma2 = ssr / (n - 2) if n > 2 else float("nan")
    sxx = float(np.sum((X - X.mean()) ** 2))
    se_alpha = float(np.sqrt(sigma2 * (1 / n + X.mean() ** 2 / sxx))) if sxx > 0 else float("nan")
    t_alpha = alpha_m / se_alpha if se_alpha and se_alpha > 0 else float("nan")
    alpha_annualized = float((1 + alpha_m) ** 12 - 1)
    return {"beta": beta, "alpha_annualized": alpha_annualized, "r2": r2, "t_alpha": t_alpha, "n": n}


def print_stats_table(rows, header_label="Variant"):
    header = f"{header_label:<20}{'TotalRet':>11}{'AnnRet':>10}{'Sharpe':>9}{'MaxDD':>9}"
    print(header)
    print("-" * len(header))
    for label, s in rows:
        print(f"{label:<20}{s['total_return']*100:>10.2f}%{s['annualized']*100:>9.2f}%"
              f"{s['sharpe']:>9.2f}{s['max_dd']*100:>8.2f}%")


def report_factor(name, formations):
    print()
    print("=" * 96)
    print(f"FACTOR: {name}")
    print("=" * 96)

    n_rebalances = len(formations)
    if n_rebalances == 0:
        print("No usable rebalance formations -- skipping.")
        return

    res = run_annual_factor(formations)
    n_periods = len(res)
    if n_periods == 0:
        print("No usable monthly holding-period observations -- skipping.")
        return

    port_ret = res["port_ret"].to_numpy()
    turnover = res["turnover"].to_numpy()
    bench_ret = res["bench_ret"].to_numpy()
    spy_ret = res["spy_ret"].to_numpy()
    n_long_per_period = res["n_long"].to_numpy()

    avg_decile = np.mean([f["n_long"] for f in formations])
    print(f"Rebalance count (annual formations): {n_rebalances} "
          f"({formations[0]['formation_month']} to {formations[-1]['formation_month']})")
    print(f"Monthly holding-period observations: {n_periods}")
    print(f"*** AVERAGE STOCKS PER DECILE: {avg_decile:.1f} ***")
    if avg_decile < MIN_DECILE_SIZE_TRUSTED:
        print(f"*** WARNING: {avg_decile:.1f} is BELOW the {MIN_DECILE_SIZE_TRUSTED}-stock trust threshold "
              f"set for this run -- treat this factor's result with corresponding caution. ***")
    else:
        print(f"Decile size clears the {MIN_DECILE_SIZE_TRUSTED}-stock bar set for this run.")

    avg_annual_turnover = res.loc[res["is_rebalance_month"], "turnover"].mean()
    print(f"Average turnover per annual rebalance: {avg_annual_turnover*100:.1f}%")

    # $1/order commission layer, expressed as a per-period cost rate that
    # multiplies elementwise against `turnover` inside perf_stats() --
    # perf_stats() itself is unmodified; only the cost argument varies.
    # turnover_frac * n_long stocks are replaced each rebalance (see
    # derivation in the module docstring's sibling comment below), so
    # 2 orders (1 sell + 1 buy) per replaced name, as a fraction of a
    # $ACCOUNT_EQUITY account:
    combined_cost_rate = 0.002 + (2.0 * n_long_per_period * COMMISSION_PER_ORDER) / ACCOUNT_EQUITY

    print()
    print_stats_table([
        ("gross", perf_stats(port_ret)),
        ("net_20bp", perf_stats(port_ret, turnover, 0.002)),
        (f"net_20bp+$1/ord", perf_stats(port_ret, turnover, combined_cost_rate)),
    ])

    print()
    print(f"--- Comparison: long decile vs equal-weighted universe vs SPY (gross), "
          f"same {n_periods} monthly observations ---")
    valid_spy = ~np.isnan(spy_ret)
    print_stats_table([
        ("Long decile", perf_stats(port_ret)),
        ("Universe EW", perf_stats(bench_ret[~np.isnan(bench_ret)])),
        ("SPY buy&hold", perf_stats(spy_ret[valid_spy])),
    ], header_label="Book")

    print()
    print("--- Formal alpha test: long decile monthly returns regressed on SPY monthly returns ---")
    reg = alpha_regression(port_ret, spy_ret)
    print(f"Beta:                 {reg['beta']:.3f}")
    print(f"Alpha (annualized):   {reg['alpha_annualized']*100:.3f}%")
    print(f"R-squared:            {reg['r2']:.3f}")
    print(f"t-stat on alpha:      {reg['t_alpha']:.2f}   (n={reg['n']} monthly observations)")
    if not np.isnan(reg['t_alpha']) and reg['t_alpha'] > 2 and reg['alpha_annualized'] > 0:
        print("-> Alpha is positive and statistically significant (t>2): not fully explained by SPY beta.")
    else:
        print("-> Alpha is NOT both positive and statistically significant: consistent with this being")
        print("   leveraged/adjusted market exposure rather than demonstrated skill.")


print()
print("=" * 96)
print("BUILDING ANNUAL FORMATIONS")
print("=" * 96)
value_formations = build_formations(value_signal, long_is_high_signal=True)
quality_formations = build_formations(quality_signal, long_is_high_signal=True)
print(f"Value: {len(value_formations)} usable June formations out of {len(june_months)} candidates")
print(f"Quality: {len(quality_formations)} usable June formations out of {len(june_months)} candidates")

report_factor("Value -- Book-to-Market (long top decile, cheapest)", value_formations)
report_factor("Quality -- Gross Profitability (long top decile, most profitable)", quality_formations)

print()
print("=" * 96)
print("This is a diagnostic run of two pre-registered specs only: no parameters were tuned,")
print("no annual-duration windows or decile cuts were varied, and no configuration was selected")
print("from results. Practice window ends 2018-12; 2019+ is a sealed holdout not read above.")
