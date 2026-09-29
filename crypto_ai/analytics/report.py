"""
report.py -- the human-readable interim / final report.

Enforces PREREGISTRATION.md section 10 in code: before the decision point it prints the
INTERIM banner and does NOT print a verdict, no matter how good or bad the numbers look.

Usage: python -m crypto_ai.analytics.report --state-dir DIR
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from crypto_ai.analytics import metrics as M
from crypto_ai.journal import Journal, now_utc, parse_iso
from crypto_ai.lock import load_config

BANNER = "INTERIM -- INFORMATIONAL ONLY. Interim performance is not evidence of edge."


def _f(x, pct=False, d=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100 * x:.{d}f}%" if pct else f"{x:.{d}f}"


def build_report(state_dir, cfg, now=None):
    now = now or now_utc()
    j = Journal(state_dir)
    lock = j.load_json("lock.json")
    cycles = j.read("cycles.jsonl")
    eq = j.read("equity.jsonl")
    lines = []
    ok = [c for c in cycles if c["status"] in ("OK", "OK_AI_FAULT")]
    start = parse_iso(lock["locked_at"]) if lock else (parse_iso(ok[0]["started_at"]) if ok else now)
    months = (now - start).days / 30.44
    decision = next((m for m in cfg["evaluation"]["decision_months"] if months >= m), None)
    enough = len(ok) >= cfg["evaluation"]["min_cycles_for_decision"]
    at_decision = decision is not None and enough

    if not at_decision:
        lines += ["=" * 90, BANNER, "=" * 90]
    lines.append(f"experiment={cfg['experiment_id']}  locked={'yes ' + lock['locked_at'] if lock else 'NO (dry run)'}"
                 f"  elapsed={months:.1f} months")
    lines.append(f"cycles: {len(ok)} completed, {sum(c['status'] == 'OK_AI_FAULT' for c in cycles)} AI faults, "
                 f"{sum(c['status'] == 'DATA_FAULT' for c in cycles)} data faults")
    toks = sum((c.get("tokens_in") or 0) + (c.get("tokens_out") or 0) for c in cycles)
    lines.append(f"LLM tokens used so far: {toks:,}")

    bench = M.daily_returns(eq, "btc_hold")
    lines.append("")
    lines.append(f"{'arm':12} {'return':>9} {'vol':>8} {'sharpe':>7} {'sortino':>8} {'maxDD':>8} "
                 f"{'beta':>6} {'alpha/yr':>9} {'fees$':>8} {'slip$':>7} {'turnover':>9}")
    for arm in cfg["arms"]:
        rows = [r for r in eq if r["arm"] == arm]
        if not rows:
            continue
        r = M.daily_returns(eq, arm)
        e = [x["equity"] for x in rows]
        ab = M.alpha_beta(r, bench) if arm != "btc_hold" else {"alpha_ann": float("nan"), "beta": 1.0}
        vol = r.std(ddof=1) * math.sqrt(365) if len(r) > 1 else float("nan")
        init = cfg["capital"]["initial_cash_usd"]
        lines.append(f"{arm:12} {_f(e[-1] / init - 1, True):>9} {_f(vol, True, 1):>8} {_f(M.sharpe(r)):>7} "
                     f"{_f(M.sortino(r)):>8} {_f(M.max_drawdown(e), True, 1):>8} {_f(ab['beta']):>6} "
                     f"{_f(ab['alpha_ann'], True, 1):>9} {rows[-1]['fees_usd']:>8.2f} "
                     f"{rows[-1]['slippage_usd']:>7.2f} {rows[-1]['traded_usd'] / init:>8.1f}x")

    dec, snaps = j.read("decisions.jsonl"), j.read("snapshots.jsonl")
    lag = cfg["evaluation"]["nw_lag_days"]
    ic24 = M.daily_ic_stats(M.ic_series(dec, snaps, 24), lag)
    ic72 = M.daily_ic_stats(M.ic_series(dec, snaps, 72), lag)
    lines.append("")
    lines.append("PRIMARY TEST -- AI outlook vs next return (cross-sectional Spearman IC)")
    for name, s in (("24h", ic24), ("72h", ic72)):
        lines.append(f"  IC{name}: mean={_f(s['mean_ic'], d=4)}  NW t={_f(s['t_nw'])}  days={s['n_days']}  "
                     f"cycles={s['n_cycles']}  undefined(all-equal outlook)={s['n_undefined']}")
    lines.append("")
    if at_decision:
        lines.append(f"DECISION POINT ({decision} months) -- primary verdict: {M.primary_verdict(ic24, ic72)}")
        lines.append("Secondary (economic) test applies only if primary = PREDICTIVE; see PREREGISTRATION.md s7.")
    else:
        need = cfg["evaluation"]["decision_months"][0]
        lines.append(f"No verdict: decision point is {need} months after lock AND >= "
                     f"{cfg['evaluation']['min_cycles_for_decision']} completed cycles. "
                     f"Numbers above are noise-dominated by design (PREREGISTRATION.md s6).")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", required=True)
    print(build_report(ap.parse_args().state_dir, load_config()))
