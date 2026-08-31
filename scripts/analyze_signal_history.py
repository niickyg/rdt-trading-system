#!/usr/bin/env python3
"""
analyze_signal_history.py — offline, dependency-free audit of the signal record.

Purpose
-------
The operator loop needs an honest, reproducible answer to one question:
*does the bot know whether its own signals win or lose?* This script reads the
persisted signal files and reports:

  1. How many signals were generated, over what window, and their composition
     (direction, RRS distribution, symbol concentration).
  2. Whether any *outcome* labels exist on those signals (target hit / stopped
     out / realised PnL). Without outcome labels there is no feedback loop and
     profitability cannot be measured.
  3. The theoretical reward:risk the system is betting on (from entry/stop/
     target) and the win rate that structure would need just to break even.

It deliberately uses only the Python standard library so it runs in any
checkout with no network and no pip installs. It reads, never writes.

Usage
-----
    python3 scripts/analyze_signal_history.py
    python3 scripts/analyze_signal_history.py --history data/signals/signal_history.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


OUTCOME_KEYS = (
    "outcome",
    "result",
    "pnl",
    "realized_pnl",
    "exit_price",
    "exit_reason",
    "hit",
    "won",
    "closed_at",
    "target_hit",
    "stopped_out",
)


def _load(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"file not found: {path}")
    with path.open() as fh:
        return json.load(fh)


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def analyze_history(history: list[dict]) -> None:
    print("=" * 70)
    print("SIGNAL HISTORY")
    print("=" * 70)
    print(f"total signals logged : {len(history)}")
    if not history:
        return

    dirs = Counter(s.get("direction") for s in history)
    print(f"direction split      : {dict(dirs)}")

    dates = sorted(s.get("generated_at", "") for s in history if s.get("generated_at"))
    if dates:
        print(f"window               : {dates[0]}  ->  {dates[-1]}")

    syms = Counter(s.get("symbol") for s in history)
    print(f"unique symbols       : {len(syms)}  (top: {syms.most_common(5)})")

    rrs = [s["rrs"] for s in history if _num(s.get("rrs"))]
    if rrs:
        print(
            "rrs                  : "
            f"n={len(rrs)} mean={statistics.mean(rrs):.2f} "
            f"min={min(rrs):.2f} max={max(rrs):.2f}"
        )

    # --- outcome labels present? ---
    found = set()
    for s in history:
        for k in s:
            if k in OUTCOME_KEYS or "outcome" in k.lower():
                found.add(k)
    print()
    print("OUTCOME LABELS")
    print("-" * 70)
    if found:
        print(f"outcome-like keys present: {sorted(found)}")
    else:
        print("NO outcome labels found on any signal.")
        print("=> The signal record is unlabelled. Win/loss and realised PnL")
        print("   are never written back. The bot cannot measure profitability")
        print("   from this data. Closing this feedback loop is prerequisite #1.")

    # --- theoretical reward:risk from entry/stop/target ---
    rr = []
    for s in history:
        entry, stop, target, direction = (
            s.get("entry_price"),
            s.get("stop_price"),
            s.get("target_price"),
            s.get("direction"),
        )
        if not all(_num(v) for v in (entry, stop, target)):
            continue
        risk = abs(entry - stop)
        reward = abs(target - entry)
        if risk > 0:
            rr.append(reward / risk)
    print()
    print("THEORETICAL REWARD:RISK (from entry/stop/target)")
    print("-" * 70)
    if rr:
        med = statistics.median(rr)
        mean = statistics.mean(rr)
        breakeven_wr = 1.0 / (1.0 + med) if med > 0 else float("nan")
        print(f"planned R:R          : n={len(rr)} median={med:.2f} mean={mean:.2f}")
        print(
            "breakeven win rate   : "
            f"{breakeven_wr * 100:.1f}%  (at median R:R, before costs/slippage)"
        )
        print("NOTE: this is the *plan*, not the *result*. Actual expectancy")
        print("      requires realised exits, which are absent (see above).")
    else:
        print("insufficient entry/stop/target data to compute R:R.")


def analyze_metrics(metrics: dict) -> None:
    print()
    print("=" * 70)
    print("SIGNAL METRICS (signal_metrics.json)")
    print("=" * 70)
    total_outcomes = metrics.get("total_outcomes", 0)
    total_signals = metrics.get("total_signals", 0)
    print(f"total_scans          : {metrics.get('total_scans')}")
    print(f"total_signals        : {total_signals}")
    print(f"total_outcomes       : {total_outcomes}")
    print(f"target_hits/stop_outs: {metrics.get('target_hits')}/{metrics.get('stop_outs')}")
    if total_signals:
        cov = 100.0 * total_outcomes / total_signals
        print(f"outcome coverage     : {cov:.1f}% of signals have a recorded outcome")
        if cov < 50:
            print("=> Outcome coverage is effectively zero. record_outcome() is")
            print("   not wired into the exit path. No feedback loop exists.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--history", default="data/signals/signal_history.json")
    ap.add_argument("--metrics", default="data/signals/signal_metrics.json")
    args = ap.parse_args()

    history = _load(Path(args.history))
    if isinstance(history, dict):  # tolerate {"signals": [...]}
        history = history.get("signals", [])
    analyze_history(history)

    metrics_path = Path(args.metrics)
    if metrics_path.exists():
        analyze_metrics(_load(metrics_path))


if __name__ == "__main__":
    main()
