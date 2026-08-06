#!/usr/bin/env python3
"""
Adversarial control for evaluate_signal_edge.py.

Question it answers: is the RRS signal's forward-return expectancy real STOCK-SELECTION /
TIMING edge, or is it just market beta (drift) captured by a 2:1 barrier in a bull market?

Control design (isolates the timing signal, holds everything else constant):
  For each real signal, keep the SAME symbol, SAME direction, and SAME barrier geometry
  (dollar risk = |entry-stop|, target distance = |target-entry|). Replace ONLY the entry
  DATE with a uniformly random trading day drawn from that symbol's price window, and set
  entry_ctrl = that day's close, stop/target at the same dollar distances. Then run the
  identical first-touch barrier test forward.

If RRS timing has edge, the REAL expectancy should clearly exceed the CONTROL expectancy.
If they are similar, the "edge" is drift + barrier geometry, not the signal.

Also reports a naive-cost haircut in R units.
"""
import json
import random
import statistics

HORIZON = 10
N_CONTROL_DRAWS = 20  # average many random-date draws per signal to shrink control variance
SEED = 42
# round-trip cost as a fraction of 1R (risk). Assume ~2-5% of the stop distance per side.
COST_R = 0.10


def load(p):
    with open(p) as f:
        return json.load(f)


def barrier(bars, start_idx, direction, entry, stop, tgt, horizon):
    """First-touch from bars[start_idx:] over `horizon` bars. Returns R multiple."""
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
        if hit_stop:  # conservative: stop checked first, and if both, loss
            return -1.0
        if hit_tgt:
            return rr
    last = fwd[-1]["c"]
    return (last - entry) / risk if direction == "long" else (entry - last) / risk


def main():
    rng = random.Random(SEED)
    signals = load("data/signals/signal_history.json")
    prices = load("data/operator_journal/artifacts/2026-08-06_ibkr_daily_prices.json")

    real_r, ctrl_r = [], []
    long_real, short_real, long_ctrl, short_ctrl = [], [], [], []

    for s in signals:
        sym, direction = s.get("symbol"), s.get("direction")
        e, stop, tgt = s.get("entry_price"), s.get("stop_price"), s.get("target_price")
        gen = s.get("generated_at", "")
        bars = prices.get(sym)
        if not (sym and direction and e and stop and tgt and gen and bars):
            continue
        risk = abs(e - stop)
        tgt_dist = abs(tgt - e)
        if risk <= 1e-9:
            continue

        # REAL: enter first bar strictly after signal date
        sig_date = gen[:10]
        idxs = [i for i, b in enumerate(bars) if b["date"] > sig_date]
        if not idxs:
            continue
        r = barrier(bars, idxs[0], direction, e, stop, tgt, HORIZON)
        if r is not None:
            real_r.append(r)
            (long_real if direction == "long" else short_real).append(r)

        # CONTROL: random entry dates, same geometry (dollar distances), same direction
        # valid start indices leave room for at least 1 forward bar
        valid = list(range(0, len(bars) - 1))
        if not valid:
            continue
        for _ in range(N_CONTROL_DRAWS):
            t = rng.choice(valid)
            ec = bars[t]["c"]
            if direction == "long":
                sc, tc = ec - risk, ec + tgt_dist
            else:
                sc, tc = ec + risk, ec - tgt_dist
            rc = barrier(bars, t + 1, direction, ec, sc, tc, HORIZON)
            if rc is not None:
                ctrl_r.append(rc)
                (long_ctrl if direction == "long" else short_ctrl).append(rc)

    def stat(name, arr):
        if not arr:
            print(f"{name}: (empty)")
            return
        exp = statistics.mean(arr)
        print(f"{name:26s} n={len(arr):6d}  expectancy={exp:+.3f}R  "
              f"net_of_cost={exp - COST_R:+.3f}R")

    print(f"HORIZON={HORIZON} trading days, control draws/signal={N_CONTROL_DRAWS}, "
          f"cost={COST_R}R round-trip\n")
    stat("REAL   all", real_r)
    stat("CONTROL(random-date) all", ctrl_r)
    print()
    stat("REAL   long", long_real)
    stat("CONTROL long", long_ctrl)
    print()
    stat("REAL   short", short_real)
    stat("CONTROL short", short_ctrl)
    print()
    if real_r and ctrl_r:
        edge = statistics.mean(real_r) - statistics.mean(ctrl_r)
        print(f">>> RRS timing edge over random-date control (all): {edge:+.3f}R per signal")
        el = (statistics.mean(long_real) - statistics.mean(long_ctrl)) if long_real and long_ctrl else float('nan')
        print(f">>> RRS timing edge, LONGS only: {el:+.3f}R per signal")


if __name__ == "__main__":
    main()
