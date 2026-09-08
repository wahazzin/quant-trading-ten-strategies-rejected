"""
Trade attribution -- the autopsy layer.

For every closed trade, decompose the P&L into WHY it happened rather than
just recording that it happened. Without this you cannot answer "was that
loss the signal's fault, the market's, or the execution's?" -- and no
learning layer can ever be built on top of outcomes alone.

DECOMPOSITION
    total_return = market_component + idiosyncratic_component - costs

    market_component  = beta * SPY_return_over_holding_period
        What you'd have earned just by being exposed to the market.
        A "winning" trade that underperformed its own beta is a LOSS
        in attribution terms -- this is the distinction that killed
        Tests 5, 7, 8 and 11, each of which looked profitable and was
        actually just market exposure.

    idiosyncratic     = total_return - market_component
        What was actually specific to this stock. THIS is the number
        a signal should be judged on, not raw P&L.

    costs             = commission + spread estimate

ALSO RECORDED PER TRADE
    signal scores at entry (every signal, not just the deciding one)
    sentiment score at entry (logged veto -- see architecture note 4)
    regime label at entry
    capacity state (were we at max positions? what got crowded out?)
    exit reason (target / stop / time limit / rebalance)

WHY OUTCOME-BASED LEARNING IS DANGEROUS HERE, IN ONE LINE:
    a correct decision can lose money and a bad decision can win. Training
    on outcomes alone teaches a model to avoid good bets that happened to
    lose. Attribution separates decision quality from luck, which is the
    only basis on which learning could ever be legitimate.
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ATTRIBUTION_LOG = Path("data/trade_attribution.parquet")


def estimate_beta(stock_returns: pd.Series, spy_returns: pd.Series,
                  min_obs: int = 60) -> float:
    """
    Trailing beta from aligned daily returns. Returns 1.0 (market-neutral
    assumption) when there is insufficient history -- deliberately
    conservative: assuming beta=1 attributes MORE of the return to the
    market, so a signal must clear a higher bar to look good.
    """
    aligned = pd.concat([stock_returns, spy_returns], axis=1).dropna()
    if len(aligned) < min_obs:
        return 1.0
    x = aligned.iloc[:, 1].values
    y = aligned.iloc[:, 0].values
    var_x = np.var(x, ddof=1)
    if var_x == 0:
        return 1.0
    return float(np.cov(y, x)[0, 1] / var_x)


def attribute_trade(
    ticker: str,
    entry_date, exit_date,
    entry_price: float, exit_price: float,
    shares: float,
    spy_entry: float, spy_exit: float,
    beta: float,
    costs: float = 0.0,
    signal_scores: dict | None = None,
    sentiment_score: float | None = None,
    regime: str | None = None,
    capacity_full: bool = False,
    exit_reason: str = "unknown",
) -> dict:
    """
    Decompose one closed trade. Returns a dict ready to append to the log.

    The number that matters is `idiosyncratic` -- return the signal is
    actually responsible for, after stripping out market exposure.
    """
    total = exit_price / entry_price - 1.0
    spy_ret = spy_exit / spy_entry - 1.0
    market_component = beta * spy_ret
    idiosyncratic = total - market_component
    net = total - costs

    return {
        "ticker": ticker,
        "entry_date": pd.Timestamp(entry_date),
        "exit_date": pd.Timestamp(exit_date),
        "holding_days": (pd.Timestamp(exit_date)
                         - pd.Timestamp(entry_date)).days,
        "shares": shares,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "total_return": total,
        "net_return": net,
        "spy_return": spy_ret,
        "beta": beta,
        "market_component": market_component,
        "idiosyncratic": idiosyncratic,
        "costs": costs,
        # did the signal actually add anything beyond market exposure?
        "signal_added_value": idiosyncratic > 0,
        "signal_scores": json.dumps(signal_scores or {}),
        "sentiment_score": sentiment_score,
        "regime": regime,
        "capacity_full": capacity_full,
        "exit_reason": exit_reason,
        "logged_at": datetime.utcnow(),
    }


def log_attribution(record: dict, path: Path = ATTRIBUTION_LOG) -> None:
    """Append one attributed trade. Creates the file on first write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([record])
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df_new], ignore_index=True)
    else:
        df = df_new
    df.to_parquet(path, index=False)


def analyze(path: Path = ATTRIBUTION_LOG, min_trades: int = 20) -> None:
    """
    The autopsy report. Answers, from logged trades:
      - is the edge real (idiosyncratic) or just market exposure (beta)?
      - which signals are actually contributing?
      - does the sentiment veto have any predictive value?
      - what is capacity costing us?
    """
    if not path.exists():
        print("No attribution log yet -- nothing to analyze.")
        return
    df = pd.read_parquet(path)
    n = len(df)
    print("=" * 62)
    print(f"TRADE ATTRIBUTION -- {n} closed trades")
    print("=" * 62)
    if n < min_trades:
        print(f"Only {n} trades. Need >= {min_trades} before any number here")
        print("is worth reading. Reporting raw counts only.")

    tot = df["total_return"].mean()
    mkt = df["market_component"].mean()
    idio = df["idiosyncratic"].mean()
    print(f"\n  mean total return    : {tot*100:+.3f}%")
    print(f"  mean market component: {mkt*100:+.3f}%   <- beta exposure")
    print(f"  mean idiosyncratic   : {idio*100:+.3f}%   <- THE SIGNAL")
    if n >= min_trades:
        se = df["idiosyncratic"].std(ddof=1) / np.sqrt(n)
        t = idio / se if se > 0 else float("nan")
        print(f"  idiosyncratic t-stat : {t:.2f}"
              f"   {'(significant)' if abs(t) > 2 else '(not significant)'}")
    share = (mkt / tot * 100) if tot != 0 else float("nan")
    print(f"  share of return from market: {share:.0f}%")
    print(f"  trades where signal added value: "
          f"{df['signal_added_value'].mean()*100:.0f}%")

    # ---- exit reason breakdown ----
    if "exit_reason" in df and df["exit_reason"].notna().any():
        print("\n  by exit reason (idiosyncratic return):")
        for reason, grp in df.groupby("exit_reason"):
            print(f"    {reason:<14} n={len(grp):>4}  "
                  f"idio={grp['idiosyncratic'].mean()*100:+.3f}%")

    # ---- regime breakdown ----
    if "regime" in df and df["regime"].notna().any():
        print("\n  by regime at entry:")
        for reg, grp in df.groupby("regime"):
            print(f"    {reg:<14} n={len(grp):>4}  "
                  f"idio={grp['idiosyncratic'].mean()*100:+.3f}%")

    # ---- sentiment veto: would it have helped? ----
    if "sentiment_score" in df and df["sentiment_score"].notna().sum() > 10:
        sub = df[df["sentiment_score"].notna()]
        neg = sub[sub["sentiment_score"] < -0.2]
        pos = sub[sub["sentiment_score"] > 0.2]
        print("\n  sentiment veto (logged, not acted on):")
        print(f"    negative-sentiment entries: n={len(neg)}  "
              f"idio={neg['idiosyncratic'].mean()*100:+.3f}%"
              if len(neg) else "    negative: none yet")
        print(f"    positive-sentiment entries: n={len(pos)}  "
              f"idio={pos['idiosyncratic'].mean()*100:+.3f}%"
              if len(pos) else "    positive: none yet")
        if len(neg) > 10 and len(pos) > 10:
            gap = pos["idiosyncratic"].mean() - neg["idiosyncratic"].mean()
            print(f"    gap (pos - neg): {gap*100:+.3f}pp  "
                  f"-- veto is only justified if this is clearly positive")

    # ---- capacity cost ----
    if "capacity_full" in df and df["capacity_full"].any():
        blocked = df["capacity_full"].mean()
        print(f"\n  entries taken while at max positions: {blocked*100:.0f}%")
        print("    (Test 19: capacity turned t=3.50 into negative alpha --"
              " watch this number)")

    print("\n" + "=" * 62)
    print("READ THIS AS: if 'mean idiosyncratic' is not clearly positive")
    print("with |t| > 2, the strategy is market exposure with extra steps.")
    print("=" * 62)


if __name__ == "__main__":
    analyze()
