#!/usr/bin/env python3
"""
Honest forward-return edge evaluation for historical RRS signals.

Reads:
  - data/signals/signal_history.json  (list of emitted signals with entry/stop/target/direction/generated_at)
  - a prices JSON: {SYMBOL: [{"date":"YYYY-MM-DD","o":..,"h":..,"l":..,"c":..}, ...]} ascending by date

Method (first-touch barrier test, no lookahead):
  * Entry on the FIRST daily bar strictly AFTER the signal's calendar date (next-day fill),
    to avoid intraday-of-signal-day lookahead. Entry assumed at the signal's stated entry_price.
  * Walk forward up to HORIZON trading days. For a LONG: win if bar.high >= target,
    loss if bar.low <= stop. For a SHORT: win if bar.low <= target, loss if bar.high >= stop.
  * If both target and stop are touched on the same bar, count as a LOSS (conservative).
  * If neither is touched within HORIZON, outcome = "timeout"; R marked to the last bar's close.

Reports win rate, expectancy in R units, and breakeven win rate implied by the R:R.
This is a screening estimate: daily bars can't resolve true intrabar sequencing, and it
ignores slippage/commissions. Treat numbers as an optimistic-to-neutral upper bound on edge.
"""
import json
import sys
from datetime import datetime
from collections import defaultdict

HORIZONS = [5, 10, 20]


def load(path):
    with open(path) as f:
        return json.load(f)


def evaluate(signals, prices, horizon):
    results = []
    skipped = 0
    for s in signals:
        sym = s.get("symbol")
        direction = s.get("direction")
        e = s.get("entry_price")
        stop = s.get("stop_price")
        tgt = s.get("target_price")
        gen = s.get("generated_at", "")
        if not (sym and direction and e and stop and tgt and gen):
            skipped += 1
            continue
        bars = prices.get(sym)
        if not bars:
            skipped += 1
            continue
        sig_date = gen[:10]
        # risk per share (stop distance) must be positive
        risk = abs(e - stop)
        if risk <= 1e-9:
            skipped += 1
            continue
        rr = abs(tgt - e) / risk
        # find first bar strictly after signal date
        fwd = [b for b in bars if b["date"] > sig_date]
        fwd = fwd[:horizon]
        if not fwd:
            skipped += 1
            continue
        outcome = "timeout"
        r_mult = None
        for b in fwd:
            hi, lo = b["h"], b["l"]
            if direction == "long":
                hit_stop = lo <= stop
                hit_tgt = hi >= tgt
            else:
                hit_stop = hi >= stop
                hit_tgt = lo <= tgt
            if hit_stop and hit_tgt:
                outcome, r_mult = "loss", -1.0
                break
            if hit_stop:
                outcome, r_mult = "loss", -1.0
                break
            if hit_tgt:
                outcome, r_mult = "win", rr
                break
        if outcome == "timeout":
            last_close = fwd[-1]["c"]
            if direction == "long":
                r_mult = (last_close - e) / risk
            else:
                r_mult = (e - last_close) / risk
        results.append({"symbol": sym, "direction": direction, "outcome": outcome,
                        "r": r_mult, "rr": rr, "date": sig_date})
    return results, skipped


def summarize(results, skipped, horizon):
    n = len(results)
    if n == 0:
        print(f"[H={horizon}] no evaluable signals (skipped {skipped})")
        return
    wins = [r for r in results if r["outcome"] == "win"]
    losses = [r for r in results if r["outcome"] == "loss"]
    timeouts = [r for r in results if r["outcome"] == "timeout"]
    total_r = sum(r["r"] for r in results)
    exp_r = total_r / n
    # win rate among resolved (win/loss) trades
    resolved = len(wins) + len(losses)
    wr_resolved = len(wins) / resolved if resolved else 0.0
    avg_rr = sum(r["rr"] for r in results) / n
    breakeven_wr = 1.0 / (1.0 + avg_rr)  # p*rr - (1-p) = 0 -> p = 1/(1+rr)
    print(f"\n===== HORIZON {horizon} trading days =====")
    print(f"evaluable signals: {n}  (skipped {skipped})")
    print(f"wins: {len(wins)}  losses: {len(losses)}  timeouts: {len(timeouts)}")
    print(f"win rate (resolved only): {wr_resolved*100:.1f}%")
    print(f"avg target R:R: {avg_rr:.2f}  -> breakeven win rate: {breakeven_wr*100:.1f}%")
    print(f"EXPECTANCY: {exp_r:+.3f} R per signal  (total {total_r:+.1f}R over {n} signals)")
    # by direction
    for d in ("long", "short"):
        sub = [r for r in results if r["direction"] == d]
        if sub:
            er = sum(r["r"] for r in sub) / len(sub)
            w = sum(1 for r in sub if r["outcome"] == "win")
            l = sum(1 for r in sub if r["outcome"] == "loss")
            rr_res = w / (w + l) if (w + l) else 0
            print(f"  {d:5s}: n={len(sub):4d}  wr(resolved)={rr_res*100:4.1f}%  expectancy={er:+.3f}R")


def spy_benchmark(signals, prices):
    """SPY buy-and-hold return over the signal window (first signal date -> last bar available),
    for context on the 'beat SPY' mission bar."""
    spy = prices.get("SPY")
    if not spy:
        print("\n[SPY benchmark] no SPY price data available")
        return
    dates = sorted(s["generated_at"][:10] for s in signals if s.get("generated_at"))
    start_date = dates[0]
    entry_bars = [b for b in spy if b["date"] >= start_date]
    if not entry_bars:
        print("\n[SPY benchmark] no SPY bars in window")
        return
    start_px = entry_bars[0]["c"]
    end_px = spy[-1]["c"]
    ret = (end_px - start_px) / start_px
    print(f"\n[SPY benchmark] buy-and-hold {entry_bars[0]['date']} ({start_px:.2f}) -> "
          f"{spy[-1]['date']} ({end_px:.2f}) = {ret*100:+.1f}%")


def main():
    prices_path = sys.argv[1] if len(sys.argv) > 1 else "data/operator_journal/artifacts/2026-08-06_ibkr_daily_prices.json"
    signals = load("data/signals/signal_history.json")
    prices = load(prices_path)
    print(f"loaded {len(signals)} signals, prices for {len(prices)} symbols")
    for h in HORIZONS:
        results, skipped = evaluate(signals, prices, h)
        summarize(results, skipped, h)
    spy_benchmark(signals, prices)


if __name__ == "__main__":
    main()
