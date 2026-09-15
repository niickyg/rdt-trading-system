#!/usr/bin/env python3
"""
Offline signal-history analyzer (pure standard library).

Purpose
-------
Give the autonomous operator (and humans) a *reproducible* descriptive summary
of the signals the scanner has actually emitted, using only files committed to
the repo. It deliberately depends on nothing but the Python standard library so
it runs in a bare environment with no pandas/numpy, no market data, and no
network access.

What it does NOT do
-------------------
It does NOT compute realized P&L or win rates. Doing that honestly requires
forward price data for each signal (to see whether entry/stop/target were hit),
which is not committed to the repo. Anyone who wants profitability numbers must
run a backtest with real market data and honest transaction costs. This tool
only describes the *distribution* of signals — direction balance, RRS spread,
frequency, and coverage — so an assessment starts from facts, not vibes.

Usage
-----
    python3 scripts/analyze_signal_history.py
    python3 scripts/analyze_signal_history.py --file data/signals/signal_history.json --json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from statistics import mean, median, pstdev


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _quantile(sorted_vals, q):
    """Linear-interpolation quantile on an already-sorted list."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def analyze(records):
    """Return a plain-dict summary of a list of signal records."""
    total = len(records)
    directions = Counter((r.get("direction") or "unknown") for r in records)
    strategies = Counter((r.get("strategy") or "unknown") for r in records)
    strengths = Counter((r.get("strength") or "unknown") for r in records)

    rrs_vals = [r["rrs"] for r in records
                if isinstance(r.get("rrs"), (int, float)) and not math.isnan(r["rrs"])]
    rrs_sorted = sorted(rrs_vals)

    days = Counter((r.get("generated_at") or "")[:10] for r in records if r.get("generated_at"))
    symbols = Counter((r.get("symbol") or "unknown") for r in records)

    # Sanity flags: signals whose stop/target contradict their stated direction.
    inconsistent = 0
    for r in records:
        d = r.get("direction")
        entry, stop, target = r.get("entry_price"), r.get("stop_price"), r.get("target_price")
        if not all(isinstance(x, (int, float)) for x in (entry, stop, target)):
            continue
        if d == "long" and not (stop < entry < target):
            inconsistent += 1
        elif d == "short" and not (target < entry < stop):
            inconsistent += 1

    summary = {
        "total_signals": total,
        "direction_counts": dict(directions),
        "long_short_ratio": (round(directions.get("long", 0) / directions["short"], 2)
                             if directions.get("short") else None),
        "strategy_counts": dict(strategies),
        "strength_counts": dict(strengths),
        "rrs": {
            "count": len(rrs_vals),
            "mean": round(mean(rrs_vals), 3) if rrs_vals else None,
            "median": round(median(rrs_vals), 3) if rrs_vals else None,
            "stdev": round(pstdev(rrs_vals), 3) if len(rrs_vals) > 1 else None,
            "min": round(min(rrs_vals), 3) if rrs_vals else None,
            "p25": round(_quantile(rrs_sorted, 0.25), 3) if rrs_vals else None,
            "p75": round(_quantile(rrs_sorted, 0.75), 3) if rrs_vals else None,
            "max": round(max(rrs_vals), 3) if rrs_vals else None,
        },
        "coverage": {
            "unique_days": len(days),
            "first_day": min(days) if days else None,
            "last_day": max(days) if days else None,
            "signals_per_day": dict(sorted(days.items())),
            "avg_signals_per_active_day": round(total / len(days), 1) if days else None,
        },
        "unique_symbols": len(symbols),
        "top_symbols": dict(symbols.most_common(10)),
        "direction_consistency": {
            "inconsistent_stop_target_vs_direction": inconsistent,
            "note": "long expects stop<entry<target; short expects target<entry<stop",
        },
    }
    return summary


def _print_human(summary):
    def line(k, v):
        print(f"  {k:<34} {v}")

    print("=" * 68)
    print("SIGNAL HISTORY ANALYSIS (descriptive only — NOT profitability)".center(68))
    print("=" * 68)
    line("Total signals", summary["total_signals"])
    line("Direction counts", summary["direction_counts"])
    line("Long/short ratio", summary["long_short_ratio"])
    line("Strategy counts", summary["strategy_counts"])
    line("Strength counts", summary["strength_counts"])
    print("-" * 68)
    r = summary["rrs"]
    line("RRS n / mean / median / stdev", f"{r['count']} / {r['mean']} / {r['median']} / {r['stdev']}")
    line("RRS min / p25 / p75 / max", f"{r['min']} / {r['p25']} / {r['p75']} / {r['max']}")
    print("-" * 68)
    c = summary["coverage"]
    line("Unique active days", c["unique_days"])
    line("Date range", f"{c['first_day']} -> {c['last_day']}")
    line("Avg signals / active day", c["avg_signals_per_active_day"])
    line("Signals per day", c["signals_per_day"])
    print("-" * 68)
    line("Unique symbols", summary["unique_symbols"])
    line("Top symbols", summary["top_symbols"])
    dc = summary["direction_consistency"]
    line("Inconsistent stop/target rows", dc["inconsistent_stop_target_vs_direction"])
    print("=" * 68)
    print("NOTE: realized P&L / win rate require forward price data + honest costs,")
    print("which are NOT in the repo. This tool describes signal distribution only.")
    print("=" * 68)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        default=os.path.join(_project_root(), "data", "signals", "signal_history.json"),
        help="Path to signal history JSON (list of signal dicts).",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a human table.")
    args = parser.parse_args(argv)

    if not os.path.exists(args.file):
        print(f"ERROR: file not found: {args.file}", file=sys.stderr)
        return 2

    with open(args.file, "r") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        print(f"ERROR: expected a JSON list of signals, got {type(data).__name__}", file=sys.stderr)
        return 2

    summary = analyze(data)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        _print_human(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
