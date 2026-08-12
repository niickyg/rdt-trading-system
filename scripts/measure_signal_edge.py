#!/usr/bin/env python3
"""
measure_signal_edge.py — the honest edge measurement the mandate demands.

Takes the historical RRS signals the live bot generated (data/signals/signal_history.json)
and, using daily OHLC bars pulled from IBKR (scratchpad/hist/<SYM>.json), computes what
ACTUALLY happened to each signal's plan (target vs stop, forward returns), net of realistic
costs, and compares it head-to-head with SPY buy-and-hold over the identical windows.

This is deliberately dependency-free (stdlib only) so it runs anywhere. It is NOT a
portfolio simulator — it measures per-signal edge. Overlapping signals and capital limits
are ignored on purpose; the question here is narrower and prior: does acting on one of
these signals, on average, beat holding SPY over the same horizon, after costs?

Assumptions (all conservative, all stated):
- Entry fills at the signal's own entry_price on the signal date (the strategy's plan).
- Bracket: for longs, stop if day low <= stop_price, target if day high >= target_price.
  If both happen the same day, assume STOP first (pessimistic). Shorts mirrored.
- Max hold = HOLD_DAYS trading days; if neither level hits, exit at that day's close
  (time stop, marked to market).
- Costs: SLIPPAGE_BPS per side applied to notional (round trip = 2x). Commission ~0 for
  IBKR US equities. A frictionless pass is also reported for reference.
"""
import json
import glob
import os
import statistics as st
from datetime import datetime
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST_DIR = os.path.join(ROOT, "scratchpad", "hist")
SIGNALS = os.path.join(ROOT, "data", "signals", "signal_history.json")

HOLD_DAYS = 20          # ~1 trading month max hold
SLIPPAGE_BPS = 5.0      # per side, in basis points of notional (10 bps round trip)


def load_bars(path):
    """Return dict date_str -> (open, high, low, close), sorted date list."""
    d = json.load(open(path))
    if "time" not in d:
        return None, None
    out = {}
    for i, t in enumerate(d["time"]):
        day = t[:10]
        try:
            out[day] = (d["open"][i], d["high"][i], d["low"][i], d["close"][i])
        except (IndexError, TypeError):
            continue
    return out, sorted(out.keys())


def sig_date(s):
    return s["generated_at"][:10]


def eval_signal(sig, bars, days):
    """Return dict with outcome, r_multiple (gross), ret_pct (gross), hold_days, or None."""
    d0 = sig_date(sig)
    # first trading day on/after signal date
    idx = None
    for i, day in enumerate(days):
        if day >= d0:
            idx = i
            break
    if idx is None:
        return None
    entry = sig["entry_price"]
    stop = sig["stop_price"]
    target = sig["target_price"]
    direction = sig["direction"]
    if not all(isinstance(x, (int, float)) and x > 0 for x in (entry, stop, target)):
        return None
    risk = abs(entry - stop)
    if risk <= 0:
        return None

    outcome, exit_price, hold = "timeout", None, 0
    end = min(idx + days, len(days))
    for j in range(idx, end):
        o, h, l, c = bars[days[j]]
        hold = j - idx + 1
        if direction == "long":
            hit_stop = l <= stop
            hit_tgt = h >= target
            if hit_stop:                 # pessimistic: stop first on ambiguous bars
                outcome, exit_price = "stop", stop
                break
            if hit_tgt:
                outcome, exit_price = "target", target
                break
        else:  # short
            hit_stop = h >= stop
            hit_tgt = l <= target
            if hit_stop:
                outcome, exit_price = "stop", stop
                break
            if hit_tgt:
                outcome, exit_price = "target", target
                break
    if exit_price is None:
        exit_price = bars[days[end - 1]][3]  # close of last day
        hold = end - idx

    if direction == "long":
        gross = (exit_price - entry)
    else:
        gross = (entry - exit_price)
    r_mult = gross / risk
    ret_pct = gross / entry * 100.0
    return {"outcome": outcome, "r_gross": r_mult, "ret_gross_pct": ret_pct,
            "hold": hold, "entry_idx": idx, "end_idx": min(idx + hold, len(days) - 1),
            "direction": direction}


def spy_return(spy_bars, spy_days, start_date, hold):
    """SPY buy-and-hold % return over the same window (entry day open-ish to exit close)."""
    idx = None
    for i, day in enumerate(spy_days):
        if day >= start_date:
            idx = i
            break
    if idx is None:
        return None
    entry_c = spy_bars[spy_days[idx]][0]  # open of entry day
    j = min(idx + hold - 1, len(spy_days) - 1)
    exit_c = spy_bars[spy_days[j]][3]     # close of exit day
    return (exit_c - entry_c) / entry_c * 100.0


def main():
    files = [f for f in glob.glob(os.path.join(HIST_DIR, "*.json"))]
    bars_by_sym, days_by_sym = {}, {}
    for f in files:
        sym = os.path.splitext(os.path.basename(f))[0]
        b, d = load_bars(f)
        if b:
            bars_by_sym[sym] = b
            days_by_sym[sym] = d
    if "SPY" not in bars_by_sym:
        print("!! SPY.json missing — cannot benchmark. Aborting.")
        return
    spy_bars, spy_days = bars_by_sym["SPY"], days_by_sym["SPY"]

    signals = json.load(open(SIGNALS))
    covered = set(bars_by_sym) - {"SPY"}
    evald = [s for s in signals if s["symbol"] in covered]

    results = []
    net_slip = SLIPPAGE_BPS / 10000.0
    for s in evald:
        r = eval_signal(s, bars_by_sym[s["symbol"]], days_by_sym[s["symbol"]])
        if not r:
            continue
        # net of round-trip slippage as % of entry
        r["ret_net_pct"] = r["ret_gross_pct"] - 2 * SLIPPAGE_BPS / 100.0
        # net R: subtract slippage cost expressed in R units
        risk_pct = abs(s["entry_price"] - s["stop_price"]) / s["entry_price"] * 100.0
        slip_r = (2 * SLIPPAGE_BPS / 100.0) / risk_pct if risk_pct > 0 else 0
        r["r_net"] = r["r_gross"] - slip_r
        r["spy_ret_pct"] = spy_return(spy_bars, spy_days, sig_date(s), r["hold"])
        r["excess_pct"] = (r["ret_net_pct"] - r["spy_ret_pct"]) if r["spy_ret_pct"] is not None else None
        r["symbol"] = s["symbol"]
        results.append(r)

    n = len(results)
    if n == 0:
        print("No evaluable signals.")
        return

    longs = [r for r in results if r["direction"] == "long"]
    shorts = [r for r in results if r["direction"] == "short"]
    wins = [r for r in results if r["outcome"] == "target"]
    stops = [r for r in results if r["outcome"] == "stop"]
    timeouts = [r for r in results if r["outcome"] == "timeout"]

    def avg(xs):
        xs = [x for x in xs if x is not None]
        return st.mean(xs) if xs else float("nan")

    print("=" * 70)
    print("HONEST SIGNAL EDGE MEASUREMENT")
    print("=" * 70)
    print(f"Symbols with data: {len(covered)} of 48  |  Signals evaluated: {n} of {len(signals)}")
    print(f"Hold cap: {HOLD_DAYS} trading days  |  Slippage: {SLIPPAGE_BPS} bps/side")
    print(f"Direction: {len(longs)} long / {len(shorts)} short")
    print("-" * 70)
    print(f"Outcomes: target {len(wins)} ({len(wins)/n*100:.1f}%) | "
          f"stop {len(stops)} ({len(stops)/n*100:.1f}%) | timeout {len(timeouts)} ({len(timeouts)/n*100:.1f}%)")
    print(f"Win rate (target before stop): {len(wins)/n*100:.1f}%")
    print("-" * 70)
    print("PER-TRADE, NET OF COSTS:")
    print(f"  Avg R-multiple (net):     {avg([r['r_net'] for r in results]):+.3f}")
    print(f"  Avg return per trade:     {avg([r['ret_net_pct'] for r in results]):+.3f}%")
    print(f"  Median return per trade:  {st.median([r['ret_net_pct'] for r in results]):+.3f}%")
    print(f"  Avg SPY over same window: {avg([r['spy_ret_pct'] for r in results]):+.3f}%")
    print(f"  Avg EXCESS vs SPY:        {avg([r['excess_pct'] for r in results]):+.3f}%")
    print(f"  Avg hold (days):          {avg([r['hold'] for r in results]):.1f}")
    print("-" * 70)
    print("FRICTIONLESS (for reference only):")
    print(f"  Avg R-multiple (gross):   {avg([r['r_gross'] for r in results]):+.3f}")
    print(f"  Avg return per trade:     {avg([r['ret_gross_pct'] for r in results]):+.3f}%")
    print("-" * 70)
    # Illustrative sequential equity: 1% risk/trade, compounding, ignores overlap/capital
    eq = 1.0
    for r in sorted(results, key=lambda x: x["entry_idx"]):
        eq *= (1 + 0.01 * r["r_net"])
    print(f"Illustrative equity (1% risk/trade, {n} trades, compounded, NO overlap cap):")
    print(f"  Strategy multiple: {eq:.3f}x  (={(eq-1)*100:+.1f}%)")
    # SPY buy-and-hold over the full signal-to-now window
    first_sig = min(sig_date(s) for s in evald)
    spy_full = spy_return(spy_bars, spy_days, first_sig, len(spy_days))
    print(f"  SPY buy&hold from {first_sig} to {spy_days[-1]}: {spy_full:+.1f}%")
    print("=" * 70)
    # by-direction excess
    print(f"Long avg excess vs SPY:  {avg([r['excess_pct'] for r in longs]):+.3f}%  (n={len(longs)})")
    print(f"Short avg excess vs SPY: {avg([r['excess_pct'] for r in shorts]):+.3f}%  (n={len(shorts)})")
    print("=" * 70)
    # persist machine-readable summary
    summ = {
        "n_signals": n, "n_symbols": len(covered),
        "win_rate_pct": len(wins) / n * 100,
        "avg_r_net": avg([r["r_net"] for r in results]),
        "avg_ret_net_pct": avg([r["ret_net_pct"] for r in results]),
        "avg_spy_pct": avg([r["spy_ret_pct"] for r in results]),
        "avg_excess_pct": avg([r["excess_pct"] for r in results]),
        "long_excess_pct": avg([r["excess_pct"] for r in longs]),
        "short_excess_pct": avg([r["excess_pct"] for r in shorts]),
        "illustrative_equity_mult": eq,
        "spy_buyhold_pct": spy_full,
        "hold_days_cap": HOLD_DAYS, "slippage_bps": SLIPPAGE_BPS,
    }
    out = os.path.join(ROOT, "scratchpad", "edge_summary.json")
    json.dump(summ, open(out, "w"), indent=2)
    print(f"Summary written to {out}")


if __name__ == "__main__":
    main()
