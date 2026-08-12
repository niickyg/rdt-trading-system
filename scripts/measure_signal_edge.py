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


def eval_signal(sig, bars, day_list, hold_days=HOLD_DAYS, fill_mode="plan"):
    """Return dict with outcome, r_multiple (gross), ret_pct (gross), hold_days, or None.

    fill_mode="plan": fill at the signal's planned entry_price (optimistic — assumes the
        level is always reachable). fill_mode="open": fill at the entry day's OPEN, which
        is what you could realistically get acting on an overnight-generated signal. Stop
        and target remain the planned absolute prices; if the open is already past a level
        the trade resolves immediately at that level.
    """
    d0 = sig_date(sig)
    # first trading day on/after signal date
    idx = None
    for i, day in enumerate(day_list):
        if day >= d0:
            idx = i
            break
    if idx is None:
        return None
    stop = sig["stop_price"]
    target = sig["target_price"]
    direction = sig["direction"]
    entry = sig["entry_price"]
    if fill_mode == "open":
        entry = bars[day_list[idx]][0]  # entry-day open
    if not all(isinstance(x, (int, float)) and x > 0 for x in (entry, stop, target)):
        return None
    risk = abs(entry - stop)
    if risk <= 0:
        return None

    outcome, exit_price, hold = "timeout", None, 0
    end = min(idx + hold_days, len(day_list))
    for j in range(idx, end):
        o, h, l, c = bars[day_list[j]]
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
        exit_price = bars[day_list[end - 1]][3]  # close of last day
        hold = end - idx

    if direction == "long":
        gross = (exit_price - entry)
    else:
        gross = (entry - exit_price)
    r_mult = gross / risk
    ret_pct = gross / entry * 100.0
    return {"outcome": outcome, "r_gross": r_mult, "ret_gross_pct": ret_pct,
            "hold": hold, "entry_idx": idx, "end_idx": min(idx + hold, len(day_list) - 1),
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
    # Dedupe: the scanner re-emits the same signal every scan (~60s), so identical
    # (symbol, date, direction, entry, stop, target) rows are ONE tradeable signal.
    seen, deduped = set(), []
    for s in signals:
        key = (s["symbol"], sig_date(s), s["direction"],
               round(s.get("entry_price") or 0, 2), round(s.get("stop_price") or 0, 2),
               round(s.get("target_price") or 0, 2))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)
    print(f"Raw signals: {len(signals)}  ->  deduped distinct signals: {len(deduped)}")
    from collections import Counter as _C
    _dc = sorted(_C(sig_date(s) for s in deduped).items())
    print(f"Signal-date concentration (EFFECTIVE sample size): {_dc}")
    print("  ^^ If signals cluster on a few dates, N is misleading — trades on the same day")
    print("     are correlated bets on one market event, not independent samples.")
    signals = deduped
    covered = set(bars_by_sym) - {"SPY"}
    evald = [s for s in signals if s["symbol"] in covered]

    def avg(xs):
        xs = [x for x in xs if x is not None]
        return st.mean(xs) if xs else float("nan")

    def build_results(fill_mode):
        out = []
        for s in evald:
            r = eval_signal(s, bars_by_sym[s["symbol"]], days_by_sym[s["symbol"]],
                            fill_mode=fill_mode)
            if not r:
                continue
            r["ret_net_pct"] = r["ret_gross_pct"] - 2 * SLIPPAGE_BPS / 100.0
            # entry actually used depends on fill_mode; recompute risk% from it
            entry_used = (bars_by_sym[s["symbol"]][days_by_sym[s["symbol"]][r["entry_idx"]]][0]
                          if fill_mode == "open" else s["entry_price"])
            risk_pct = abs(entry_used - s["stop_price"]) / entry_used * 100.0 if entry_used else 0
            slip_r = (2 * SLIPPAGE_BPS / 100.0) / risk_pct if risk_pct > 0 else 0
            r["r_net"] = r["r_gross"] - slip_r
            r["spy_ret_pct"] = spy_return(spy_bars, spy_days, sig_date(s), r["hold"])
            r["excess_pct"] = (r["ret_net_pct"] - r["spy_ret_pct"]) if r["spy_ret_pct"] is not None else None
            r["symbol"] = s["symbol"]
            r["sigdate"] = sig_date(s)
            out.append(r)
        return out

    def tstat(xs):
        xs = [x for x in xs if x is not None]
        if len(xs) < 2:
            return float("nan")
        m, sd = st.mean(xs), st.pstdev(xs)
        return m / (sd / (len(xs) ** 0.5)) if sd > 0 else float("nan")

    def report(results, label):
        n = len(results)
        longs = [r for r in results if r["direction"] == "long"]
        shorts = [r for r in results if r["direction"] == "short"]
        wins = [r for r in results if r["outcome"] == "target"]
        excess = [r["excess_pct"] for r in results if r["excess_pct"] is not None]
        long_ex = [r["excess_pct"] for r in longs if r["excess_pct"] is not None]
        print("-" * 70)
        print(f"[{label}]  n={n}  ({len(longs)} long / {len(shorts)} short)")
        print(f"  Win rate (target<stop):   {len(wins)/n*100:.1f}%")
        print(f"  Avg R-multiple (net):     {avg([r['r_net'] for r in results]):+.3f}")
        print(f"  Avg return per trade:     {avg([r['ret_net_pct'] for r in results]):+.3f}%")
        print(f"  Median return per trade:  {st.median([r['ret_net_pct'] for r in results]):+.3f}%")
        print(f"  Avg SPY over same window: {avg([r['spy_ret_pct'] for r in results]):+.3f}%")
        print(f"  Avg EXCESS vs SPY:        {avg(excess):+.3f}%   t={tstat(excess):+.2f} (naive, ignores overlap)")
        print(f"  Long-only avg EXCESS:     {avg(long_ex):+.3f}%   t={tstat(long_ex):+.2f}")
        print(f"  Avg hold (days):          {avg([r['hold'] for r in results]):.1f}")
        return {"n": n, "win_rate_pct": len(wins)/n*100,
                "avg_r_net": avg([r['r_net'] for r in results]),
                "avg_ret_net_pct": avg([r['ret_net_pct'] for r in results]),
                "avg_excess_pct": avg(excess), "excess_t": tstat(excess),
                "long_excess_pct": avg(long_ex), "long_excess_t": tstat(long_ex)}

    def portfolio_sim(results, max_concurrent, risk_frac=0.01):
        """Realistic-ish equity: process signals in date order, cap concurrent positions,
        risk risk_frac of equity per trade, realize P&L in R at exit. Approximates calendar
        overlap by using entry_idx/end_idx on each symbol's own trading-day index. Because
        symbols share the market calendar, we use the signal DATE for slotting."""
        # order by entry date, then simulate slot occupancy by date index on SPY calendar
        date_index = {d: i for i, d in enumerate(spy_days)}
        evs = sorted(results, key=lambda r: r["sigdate"])
        equity = 1.0
        # track open positions as (release_date_index)
        open_slots = []  # list of release indices
        taken = skipped = 0
        for r in evs:
            di = date_index.get(r["sigdate"])
            if di is None:
                continue
            # free finished slots
            open_slots = [rel for rel in open_slots if rel > di]
            if len(open_slots) >= max_concurrent:
                skipped += 1
                continue
            taken += 1
            equity *= (1 + risk_frac * r["r_net"])
            open_slots.append(di + r["hold"])
        return equity, taken, skipped

    print("=" * 70)
    print("HONEST SIGNAL EDGE MEASUREMENT")
    print("=" * 70)
    print(f"Symbols with data: {len(covered)} of 48  |  Signals evaluated (of {len(evald)} coverable)")
    print(f"Hold cap: {HOLD_DAYS} trading days  |  Slippage: {SLIPPAGE_BPS} bps/side round-trip 2x")
    print("Two fill assumptions: 'plan' = fill at signal's planned entry_price (optimistic);")
    print("'open' = fill at entry-day open (realistic for an overnight-generated signal).")

    res_plan = build_results("plan")
    res_open = build_results("open")
    if not res_plan:
        print("No evaluable signals.")
        return
    s_plan = report(res_plan, "FILL @ PLANNED ENTRY (optimistic)")
    s_open = report(res_open, "FILL @ ENTRY-DAY OPEN (realistic)")

    print("=" * 70)
    first_sig = min(sig_date(s) for s in evald)
    spy_full = spy_return(spy_bars, spy_days, first_sig, len(spy_days))
    print(f"BENCHMARK: SPY buy&hold {first_sig} -> {spy_days[-1]}: {spy_full:+.1f}%")
    print("-" * 70)
    print("CAPPED PORTFOLIO EQUITY (realistic fills, 1% risk/trade, R realized at exit):")
    for cap in (3, 5, 8):
        eq, taken, skipped = portfolio_sim(res_open, cap)
        print(f"  max {cap} concurrent: {eq:.3f}x ({(eq-1)*100:+.1f}%)  "
              f"[{taken} taken, {skipped} skipped by cap]")
    print("  ^ still optimistic: assumes every planned stop/target fills at its exact level,")
    print("    no gaps through stops, no borrow cost on shorts, single 1-month signal regime.")
    print("=" * 70)

    summ = {
        "window": f"{first_sig}..{spy_days[-1]}",
        "spy_buyhold_pct": spy_full,
        "fill_plan": s_plan, "fill_open": s_open,
        "portfolio_open_fill": {
            str(c): dict(zip(("equity_mult", "taken", "skipped"), portfolio_sim(res_open, c)))
            for c in (3, 5, 8)},
        "hold_days_cap": HOLD_DAYS, "slippage_bps": SLIPPAGE_BPS,
        "n_symbols": len(covered),
    }
    out = os.path.join(ROOT, "scratchpad", "edge_summary.json")
    json.dump(summ, open(out, "w"), indent=2)
    print(f"Summary written to {out}")


if __name__ == "__main__":
    main()
