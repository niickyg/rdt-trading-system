#!/usr/bin/env python3
"""
Signal Outcome Backtest Harness (pure stdlib, no numpy/pandas required).

WHY THIS EXISTS
---------------
As of 2026-07-28 the system had generated ~1,986 raw signals but recorded
only 2 measured outcomes (see data/signals/signal_metrics.json). You cannot
make a strategy profitable if you never measure whether its signals make
money. This harness closes that gap: feed it the signal log and daily price
bars, and it computes honest forward-return outcomes for every distinct
setup, plus a SPY buy-and-hold benchmark over the matched holding window.

It is deliberately dependency-free so it runs in any environment (the remote
operator agent, CI, or the local bot container). The local bot has yfinance
and can generate the prices file; the remote operator can generate it from
the IBKR MCP `get_price_history` tool.

INPUTS
------
1. signals JSON: the repo's data/signals/signal_history.json format — a list
   of dicts with at least: symbol, direction ('long'/'short'), rrs,
   entry_price, stop_price, target_price, generated_at (ISO date/datetime).

2. prices JSON: {SYMBOL: [{"date": "YYYY-MM-DD", "open":.., "high":..,
   "low":.., "close":..}, ...], ...}. Must include "SPY" for the benchmark.
   Bars must be sorted ascending by date. Daily bars are expected.

METHOD & HONEST CAVEATS
-----------------------
* Setups are de-duplicated to one per (symbol, calendar-day, direction),
  keeping the FIRST signal of the day. The raw log is dominated by intraday
  re-emissions of the same setup, which would otherwise 20x-inflate counts.
* Entry is assumed at the signal's stated entry_price on the NEXT trading
  day's session (we scan bars strictly AFTER the signal date). If the entry
  price is never touched within `max_hold` days the setup is counted as
  "no_fill" and excluded from P&L (reported separately).
* Once filled, we walk forward day by day:
    - long:  stop if bar.low <= stop_price ; target if bar.high >= target.
    - short: stop if bar.high >= stop_price; target if bar.low <= target.
  INTRADAY PATH AMBIGUITY: if a single daily bar's range contains BOTH the
  stop and the target, we CONSERVATIVELY assume the stop hit first. This
  biases results downward (pessimistic), which is the right direction for a
  system whose job is to avoid false confidence. A future version should use
  intraday bars to resolve this precisely.
* If neither level is hit within `max_hold` trading days, the position is
  marked-to-market at the last available close ("time_exit").
* Outcomes are expressed in R (multiples of initial risk = |entry-stop|) and
  in raw percent. Commissions/slippage are applied as a per-side cost in R.

This tool does NOT prove a strategy works or fails on a thin sample. It
produces the measurement. Interpret the confidence in light of the sample
size and date span it prints.
"""
import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime


def _date(s):
    """Parse an ISO date or datetime to a YYYY-MM-DD string."""
    return str(s)[:10]


def load_setups(signals_path):
    with open(signals_path) as f:
        raw = json.load(f)
    seen = {}
    for s in raw:
        if not all(s.get(k) is not None for k in
                   ("symbol", "direction", "entry_price", "stop_price", "target_price", "generated_at")):
            continue
        key = (s["symbol"], _date(s["generated_at"]), s["direction"])
        if key not in seen:
            seen[key] = s
    return list(seen.values())


def simulate(setup, bars, max_hold, cost_r):
    """Return dict with outcome for one setup given its symbol's daily bars."""
    sig_date = _date(setup["generated_at"])
    direction = setup["direction"]
    entry = float(setup["entry_price"])
    stop = float(setup["stop_price"])
    target = float(setup["target_price"])
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    rr = abs(target - entry) / risk

    forward = [b for b in bars if b["date"] > sig_date][:max_hold]
    if not forward:
        return {"status": "no_data"}

    filled = False
    entry_date = None
    for b in forward:
        # fill check: entry touched intraday
        if not filled:
            if b["low"] <= entry <= b["high"]:
                filled = True
                entry_date = b["date"]
            else:
                continue
        # after fill, evaluate stop/target on this and subsequent bars
        hi, lo = b["high"], b["low"]
        if direction == "long":
            hit_stop = lo <= stop
            hit_tgt = hi >= target
        else:
            hit_stop = hi >= stop
            hit_tgt = lo <= target
        if hit_stop and hit_tgt:
            # ambiguous -> conservative: stop first
            return _result("stop", -1.0, entry, stop, direction, entry_date, b["date"], rr, cost_r)
        if hit_stop:
            return _result("stop", -1.0, entry, stop, direction, entry_date, b["date"], rr, cost_r)
        if hit_tgt:
            return _result("target", rr, entry, target, direction, entry_date, b["date"], rr, cost_r)

    if not filled:
        return {"status": "no_fill"}
    # time exit at last close
    last = forward[-1]
    px = last["close"]
    r_mult = ((px - entry) if direction == "long" else (entry - px)) / risk
    return _result("time_exit", r_mult, entry, px, direction, entry_date, last["date"], rr, cost_r)


def _result(status, r_gross, entry, exit_px, direction, entry_date, exit_date, rr, cost_r):
    r_net = r_gross - cost_r
    pct = ((exit_px - entry) if direction == "long" else (entry - exit_px)) / entry * 100.0
    return {"status": status, "r_gross": r_gross, "r_net": r_net, "pct": pct,
            "entry_date": entry_date, "exit_date": exit_date, "rr_planned": rr}


def spy_benchmark(bars_by_sym, entry_date, exit_date):
    spy = bars_by_sym.get("SPY")
    if not spy:
        return None
    entry_bar = next((b for b in spy if b["date"] >= entry_date), None)
    exit_bar = next((b for b in reversed(spy) if b["date"] <= exit_date), None)
    if not entry_bar or not exit_bar or entry_bar["date"] > exit_bar["date"]:
        return None
    return (exit_bar["close"] - entry_bar["close"]) / entry_bar["close"] * 100.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signals", default="data/signals/signal_history.json")
    ap.add_argument("--prices", required=True, help="JSON: {SYMBOL: [daily bars]}. Must include SPY.")
    ap.add_argument("--max-hold", type=int, default=10, help="Max trading days to hold (default 10)")
    ap.add_argument("--cost-r", type=float, default=0.05,
                    help="Round-trip cost expressed in R (default 0.05R ~ commissions+slippage)")
    ap.add_argument("--json-out", default=None, help="Optional path to write per-setup results")
    args = ap.parse_args()

    setups = load_setups(args.signals)
    with open(args.prices) as f:
        prices = json.load(f)
    bars_by_sym = {sym: sorted(bars, key=lambda b: b["date"]) for sym, bars in prices.items()}

    covered = [s for s in setups if s["symbol"] in bars_by_sym]
    results = []
    skipped_no_price = 0
    for s in setups:
        bars = bars_by_sym.get(s["symbol"])
        if not bars:
            skipped_no_price += 1
            continue
        r = simulate(s, bars, args.max_hold, args.cost_r)
        if r is None or r.get("status") in ("no_data", "no_fill"):
            results.append({"symbol": s["symbol"], "direction": s["direction"],
                            "date": _date(s["generated_at"]), "rrs": s.get("rrs"),
                            **(r or {"status": "bad_setup"})})
            continue
        bench = spy_benchmark(bars_by_sym, r["entry_date"], r["exit_date"])
        results.append({"symbol": s["symbol"], "direction": s["direction"],
                        "date": _date(s["generated_at"]), "rrs": s.get("rrs"),
                        "spy_bench_pct": bench, **r})

    filled = [r for r in results if r["status"] in ("stop", "target", "time_exit")]
    dates = sorted(_date(s["generated_at"]) for s in setups)

    print("=" * 66)
    print("SIGNAL OUTCOME BACKTEST")
    print("=" * 66)
    print(f"Distinct setups (symbol,day,dir): {len(setups)}")
    print(f"Signal date span: {dates[0]} -> {dates[-1]}  ({len(set(dates))} unique days)")
    print(f"Setups with price data: {len(covered)}   (no price data: {skipped_no_price})")
    print(f"Filled & resolved: {len(filled)}")
    if not filled:
        print("\nNo filled setups to score. Provide price data covering the signal window.")
        _maybe_dump(args.json_out, results)
        return

    wins = [r for r in filled if r["r_net"] > 0]
    win_rate = len(wins) / len(filled) * 100
    total_r = sum(r["r_net"] for r in filled)
    avg_r = total_r / len(filled)
    gross_win = sum(r["r_net"] for r in filled if r["r_net"] > 0)
    gross_loss = -sum(r["r_net"] for r in filled if r["r_net"] < 0)
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    avg_pct = sum(r["pct"] for r in filled) / len(filled)

    benched = [r for r in filled if r.get("spy_bench_pct") is not None]
    if benched:
        strat_avg = sum(r["pct"] for r in benched) / len(benched)
        spy_avg = sum(r["spy_bench_pct"] for r in benched) / len(benched)
        edge = strat_avg - spy_avg

    by_status = defaultdict(int)
    for r in filled:
        by_status[r["status"]] += 1

    print("-" * 66)
    print(f"Win rate:            {win_rate:5.1f}%")
    print(f"Expectancy:          {avg_r:+.3f} R / trade   (net of {args.cost_r}R cost)")
    print(f"Total R:             {total_r:+.2f} R")
    print(f"Profit factor:       {pf:.2f}")
    print(f"Avg raw return:      {avg_pct:+.2f}% / trade")
    print(f"Outcomes:            {dict(by_status)}")
    if benched:
        print("-" * 66)
        print(f"vs SPY buy&hold over matched holding windows (n={len(benched)}):")
        print(f"  strategy avg/trade: {strat_avg:+.2f}%")
        print(f"  SPY avg/trade:      {spy_avg:+.2f}%")
        print(f"  edge over SPY:      {edge:+.2f}% / trade")
    print("=" * 66)
    print("NOTE: intraday stop/target ambiguity resolved pessimistically "
          "(stop first).\nInterpret confidence in light of sample size and date span above.")
    _maybe_dump(args.json_out, results)


def _maybe_dump(path, results):
    if path:
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nPer-setup results written to {path}")


if __name__ == "__main__":
    sys.exit(main())
