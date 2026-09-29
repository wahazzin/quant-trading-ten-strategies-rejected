"""
inspect_state.py -- quick human-readable look at a state directory (latest cycle).
Usage: python -m crypto_ai.inspect_state --state-dir DIR
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_ai.journal import Journal

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", required=True)
    j = Journal(ap.parse_args().state_dir)
    cycles = j.read("cycles.jsonl")
    if not cycles:
        sys.exit("no cycles yet")
    cid = cycles[-1]["cycle_id"]
    print(f"== latest cycle {cid}  status={cycles[-1]['status']}  total cycles={len(cycles)} ==")
    print("\n-- equity --")
    for r in [r for r in j.read("equity.jsonl") if r["cycle_id"] == cid]:
        print(f"  {r['arm']:12} equity=${r['equity']:10.2f}  cash=${r['cash']:9.2f}  "
              f"invested={100 * r['gross_exposure']:5.1f}%  fees=${r['fees_usd']:.2f}  "
              f"spread=${r['spread_usd']:.2f}  slip=${r['slippage_usd']:.2f}")
    print("\n-- fills this cycle --")
    for o in [o for o in j.read("orders.jsonl") if o["cycle_id"] == cid]:
        print(f"  {o['arm']:12} {o['side']:4} {o['asset']:9} ${o['notional']:9.2f} "
              f"@ {o['price']:.4f} (mid {o['mid']:.4f})  cause={o['cause']}")
    print("\n-- risk engine --")
    for r in [r for r in j.read("decisions.jsonl") if r["cycle_id"] == cid]:
        for x in r["risk_decisions"]:
            fw = None if x["final_weight"] is None else round(x["final_weight"], 4)
            print(f"  {r['arm']:12} {x['action']:6} {x['asset']:9} {x['status']:8} "
                  f"req={x['requested_weight']} final={fw} codes={x['codes']}")
        if r["arm"] == "ai_pv":
            print(f"  ai outlook: {r.get('outlook')}")
            if r.get("errors"):
                print(f"  ai ERRORS: {r['errors']}")
    print("\n-- events --")
    for e in [e for e in j.read("events.jsonl") if e.get("cycle_id") == cid]:
        print(" ", e)
    snap = [s for s in j.read("snapshots.jsonl") if s["cycle_id"] == cid][-1]["assets"]["BTC-USD"]
    print("\n-- BTC features --\n ", json.dumps(snap["features"]))
