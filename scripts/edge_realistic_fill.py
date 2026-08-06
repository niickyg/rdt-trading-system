#!/usr/bin/env python3
"""
Realistic-fill re-evaluation. The naive test in evaluate_signal_edge.py enters at the signal's
stated entry_price, but diagnostics show that price is unfillable ~76% of the time (the stock has
already gapped a mean +10.5% past it). That manufactures fake edge. Here we enter at the NEXT
BAR's OPEN -- a price actually achievable -- and test two interpretations:

  A) KEEP the signal's own stop_price / target_price levels. (You chase the gap but keep the
     original bracket. Often you're already at/past target at entry.)
  B) REBUILD a fresh 2:1 bracket from the realistic entry using the signal's risk distance
     (stop = entry -/+ 1R, target = entry +/- 2R). (You take the RRS *direction* with a normal
     bracket off a real fill.) This is the "can you actually trade this" test.

Compares against the random-date control expectancy (~0R) on equal footing (both use real fills).
"""
import json
import statistics

HORIZON = 10
COST_R = 0.10


def load(p):
    with open(p) as f:
        return json.load(f)


def barrier(bars, start_idx, direction, entry, stop, tgt, horizon):
    risk = abs(entry - stop)
    if risk <= 1e-9:
        return None
    rr = abs(tgt - entry) / risk
    fwd = bars[start_idx:start_idx + horizon]
    if not fwd:
        return None
    for b in fwd:
        hi, lo = b["h"], b["l"]
        if direction == "long":
            hit_stop, hit_tgt = lo <= stop, hi >= tgt
        else:
            hit_stop, hit_tgt = hi >= stop, lo <= tgt
        if hit_stop:
            return -1.0
        if hit_tgt:
            return rr
    last = fwd[-1]["c"]
    return (last - entry) / risk if direction == "long" else (entry - last) / risk


def main():
    signals = load("data/signals/signal_history.json")
    prices = load("data/operator_journal/artifacts/2026-08-06_ibkr_daily_prices.json")

    a_all, b_all = [], []
    a_long, b_long, a_short, b_short = [], [], [], []
    already_past_target = 0
    n = 0

    for s in signals:
        sym, direction = s.get("symbol"), s.get("direction")
        e, stop, tgt = s.get("entry_price"), s.get("stop_price"), s.get("target_price")
        gen = s.get("generated_at", "")
        bars = prices.get(sym)
        if not (sym and direction and e and stop and tgt and gen and bars):
            continue
        risk0 = abs(e - stop)
        if risk0 <= 1e-9:
            continue
        sd = gen[:10]
        idxs = [i for i, b in enumerate(bars) if b["date"] > sd]
        if not idxs:
            continue
        entry_idx = idxs[0]
        fill = bars[entry_idx]["o"]  # realistic fill = next bar's open
        n += 1

        # A) keep signal's stop/target levels, enter at realistic fill, barrier from same bar's
        #    intraday after open. Approximate: test from entry_idx (same day can hit).
        # If already past target at fill, that's an instant (fill vs target) outcome.
        if direction == "long":
            if fill >= tgt:
                already_past_target += 1
                ra = (bars[entry_idx]["c"] - fill) / risk0  # can't get target edge; MTM close
            else:
                ra = barrier(bars, entry_idx, direction, fill, stop, tgt, HORIZON)
        else:
            if fill <= tgt:
                already_past_target += 1
                ra = (fill - bars[entry_idx]["c"]) / risk0
            else:
                ra = barrier(bars, entry_idx, direction, fill, stop, tgt, HORIZON)
        if ra is not None:
            a_all.append(ra)
            (a_long if direction == "long" else a_short).append(ra)

        # B) fresh 2:1 bracket off the realistic fill using original risk distance
        if direction == "long":
            sb, tb = fill - risk0, fill + 2 * risk0
        else:
            sb, tb = fill + risk0, fill - 2 * risk0
        rb = barrier(bars, entry_idx, direction, fill, sb, tb, HORIZON)
        if rb is not None:
            b_all.append(rb)
            (b_long if direction == "long" else b_short).append(rb)

    def stat(name, arr):
        if not arr:
            print(f"{name}: (empty)")
            return
        exp = statistics.mean(arr)
        print(f"{name:34s} n={len(arr):5d}  expectancy={exp:+.3f}R  net_of_cost={exp-COST_R:+.3f}R")

    print(f"HORIZON={HORIZON}, cost={COST_R}R. Realistic fill = next bar's OPEN.")
    print(f"signals evaluated: {n}; already past target at realistic fill: "
          f"{already_past_target} ({100*already_past_target/n:.1f}%)\n")
    print("A) enter at real fill, KEEP signal's stop/target levels:")
    stat("  A all", a_all); stat("  A long", a_long); stat("  A short", a_short)
    print("\nB) enter at real fill, FRESH 2:1 bracket (the tradable version):")
    stat("  B all", b_all); stat("  B long", b_long); stat("  B short", b_short)


if __name__ == "__main__":
    main()
