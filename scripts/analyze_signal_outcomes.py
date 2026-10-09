#!/usr/bin/env python3
"""
Signal Outcome Backfill & Honest Edge Measurement
==================================================

The live scanner logged 1,986 signals to ``data/signals/signal_history.json``
but ``SignalMetrics.record_outcome()`` was never wired to anything, so the
system never measured whether those signals actually worked. This script fills
that gap: it replays each distinct signal against real historical daily OHLC
(yfinance) and determines whether the target or the stop was hit first, then
reports the realized edge — net of honest costs — against SPY buy-and-hold over
the same window.

This is a MEASUREMENT tool, not a trading tool. It does not place orders.

Honesty caveats (printed in the output too):
  * ENTRY PRICE: By default we fill at the NEXT session's OPEN, not the
    signal's logged ``entry_price``. The logged entry_price is a snapshot taken
    mid-move; for 73% of long signals in the Feb-Mar 2026 data the next open is
    ABOVE it (mean +9%). "Filling" at that stale price is look-ahead bias and
    inflates the edge massively (PF 3.4 / +505R vs the realistic PF 1.8 /
    +149R). Use --entry-at-signal-price only to demonstrate that bias, never to
    make an edge claim.
  * Daily bars cannot resolve intraday ordering when BOTH stop and target are
    touched on the same day. We resolve such days PESSIMISTICALLY (stop first).
  * The fixed-risk dollar sim IGNORES concurrency/capital limits (you cannot
    hold hundreds of positions on $25K). Read the R-expectancy, not the headline
    dollar figure, as the edge estimate.
  * The logged signals span only ~1 month (Feb-Mar 2026) in one market regime,
    so this is a small, in-the-wild sample — indicative, not conclusive. The
    capital-and-execution-aware 2-year walk-forward (scripts/run_walkforward_v2)
    is the authoritative performance number.

Usage:
    /usr/local/bin/python3 scripts/analyze_signal_outcomes.py \
        [--hold-days 5] [--slippage-bps 5] [--risk-per-trade 0.01] \
        [--capital 25000] [--entry-at-signal-price]
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HISTORY_FILE = PROJECT_ROOT / "data" / "signals" / "signal_history.json"


def load_signals():
    with open(HISTORY_FILE) as f:
        raw = json.load(f)
    # Dedupe to distinct trade ideas: same (symbol, direction, rounded entry)
    # keeps the EARLIEST occurrence. The scanner re-emits persisting signals on
    # every scan, so the raw 1,986 massively over-counts distinct ideas.
    seen = {}
    for s in raw:
        try:
            key = (s["symbol"], s["direction"], round(float(s["entry_price"]), 2))
        except (KeyError, TypeError, ValueError):
            continue
        ts = s.get("generated_at", "")
        if key not in seen or ts < seen[key].get("generated_at", ""):
            seen[key] = s
    return raw, list(seen.values())


def fetch_history(symbols, start, end):
    """Download daily OHLC for all symbols, return {symbol: DataFrame}."""
    data = {}
    syms = sorted(set(symbols))
    print(f"  Downloading daily OHLC for {len(syms)} symbols "
          f"({start.date()} → {end.date()})...")
    df = yf.download(syms, start=start, end=end, interval="1d",
                     progress=False, auto_adjust=False, group_by="ticker")
    for sym in syms:
        try:
            if len(syms) == 1:
                sub = df
            else:
                sub = df[sym]
            sub = sub.dropna(subset=["High", "Low", "Close"])
            if len(sub):
                data[sym] = sub
        except (KeyError, Exception):
            continue
    print(f"  Got usable history for {len(data)}/{len(syms)} symbols")
    return data


def evaluate(sig, hist, hold_days, slippage_bps, entry_at_signal_price=False):
    """
    Return dict with outcome for one signal, or None if unresolvable.

    Entry: realistic default is the NEXT session's OPEN (the first bar strictly
    after generated_at). With entry_at_signal_price=True, uses the logged
    entry_price instead — biased, for demonstration only.

    Resolution: walk forward up to `hold_days` trading days. Long: stop if
    Low<=stop, target if High>=target (pessimistic: stop wins ties). Short
    mirrored. If neither triggers, exit at the close of the last bar (time stop).
    Signals already invalid at entry (open past stop or past target) are skipped.
    """
    sym = sig["symbol"]
    if sym not in hist:
        return None
    direction = sig["direction"]
    stop = float(sig["stop_price"])
    target = float(sig["target_price"])
    if stop <= 0 or target <= 0:
        return None
    try:
        gen = datetime.fromisoformat(sig["generated_at"].replace("Z", "+00:00"))
    except ValueError:
        return None
    gen_naive = gen.replace(tzinfo=None)

    df = hist[sym]
    future = df[df.index > pd.Timestamp(gen_naive.date())]
    if len(future) == 0:
        return None
    future = future.head(hold_days)

    if entry_at_signal_price:
        entry = float(sig["entry_price"])
    else:
        entry = float(future.iloc[0]["Open"])  # realistic fill
    if entry <= 0:
        return None
    # Skip signals already past stop/target at entry (can't be taken honestly)
    if direction == "long" and (entry <= stop or entry >= target):
        return None
    if direction == "short" and (entry >= stop or entry <= target):
        return None

    risk = abs(entry - stop)
    if risk <= 0:
        return None
    slip = entry * slippage_bps / 10000.0  # price slippage per fill

    for _, bar in future.iterrows():
        hi, lo = float(bar["High"]), float(bar["Low"])
        if direction == "long":
            hit_stop = lo <= stop
            hit_tgt = hi >= target
            if hit_stop:  # pessimistic tie-break
                exit_px = stop - slip
                return _result(sig, entry, exit_px, risk, "stop", slip)
            if hit_tgt:
                exit_px = target - slip
                return _result(sig, entry, exit_px, risk, "target", slip)
        else:  # short
            hit_stop = hi >= stop
            hit_tgt = lo <= target
            if hit_stop:
                exit_px = stop + slip
                return _result(sig, entry, exit_px, risk, "stop", slip)
            if hit_tgt:
                exit_px = target + slip
                return _result(sig, entry, exit_px, risk, "target", slip)

    # Time stop: mark-to-market at last close
    last_close = float(future.iloc[-1]["Close"])
    exit_px = last_close - slip if direction == "long" else last_close + slip
    return _result(sig, entry, exit_px, risk, "timeout", slip)


def _result(sig, entry, exit_px, risk, how, slip):
    direction = sig["direction"]
    # Effective entry includes slippage against us
    eff_entry = entry + slip if direction == "long" else entry - slip
    if direction == "long":
        pnl_per_share = exit_px - eff_entry
    else:
        pnl_per_share = eff_entry - exit_px
    r_multiple = pnl_per_share / risk
    return {
        "symbol": sig["symbol"],
        "direction": direction,
        "how": how,
        "pnl_per_share": pnl_per_share,
        "risk_per_share": risk,
        "r_multiple": r_multiple,
        "win": pnl_per_share > 0,
        "rrs": sig.get("rrs"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold-days", type=int, default=5)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--risk-per-trade", type=float, default=0.01)
    ap.add_argument("--capital", type=float, default=25000.0)
    ap.add_argument("--entry-at-signal-price", action="store_true",
                    help="Use logged entry_price (BIASED) instead of next open")
    args = ap.parse_args()

    raw, signals = load_signals()
    dates = [s.get("generated_at", "") for s in signals if s.get("generated_at")]
    dmin, dmax = min(dates)[:10], max(dates)[:10]
    print("=" * 78)
    print("SIGNAL OUTCOME BACKFILL — honest edge measurement")
    print("=" * 78)
    print(f"Raw logged signals:      {len(raw)}")
    print(f"Distinct trade ideas:    {len(signals)}  (deduped by symbol/dir/entry)")
    print(f"Signal window:           {dmin} → {dmax}")
    entry_mode = ("logged entry_price (BIASED — demo only)"
                  if args.entry_at_signal_price else "next session open (realistic)")
    print(f"Entry fill:              {entry_mode}")
    print(f"Assumptions: hold<= {args.hold_days}d, slippage {args.slippage_bps}bps/fill, "
          f"pessimistic same-day tie-break (stop wins)")
    print()

    start = datetime.fromisoformat(dmin) - timedelta(days=5)
    end = datetime.fromisoformat(dmax) + timedelta(days=args.hold_days * 2 + 10)
    hist = fetch_history([s["symbol"] for s in signals], start, end)

    results = [r for s in signals
               if (r := evaluate(s, hist, args.hold_days, args.slippage_bps,
                                 args.entry_at_signal_price))]
    n = len(results)
    if n == 0:
        print("No signals could be resolved against price history.")
        sys.exit(1)

    wins = [r for r in results if r["win"]]
    losses = [r for r in results if not r["win"]]
    gross_win_r = sum(r["r_multiple"] for r in wins)
    gross_loss_r = -sum(r["r_multiple"] for r in losses)
    win_rate = len(wins) / n
    avg_win_r = (gross_win_r / len(wins)) if wins else 0
    avg_loss_r = (gross_loss_r / len(losses)) if losses else 0
    profit_factor = (gross_win_r / gross_loss_r) if gross_loss_r > 0 else float("inf")
    total_r = sum(r["r_multiple"] for r in results)
    expectancy_r = total_r / n

    # Fixed-fractional-risk equity sim (each trade risks risk_per_trade of capital)
    risk_dollars = args.capital * args.risk_per_trade
    net_dollars = total_r * risk_dollars
    pct_return = net_dollars / args.capital * 100

    how = defaultdict(int)
    for r in results:
        how[r["how"]] += 1

    print("-" * 78)
    print("RESULTS (realized, net of slippage)")
    print("-" * 78)
    print(f"Resolved trades:    {n}  (long {sum(1 for r in results if r['direction']=='long')}, "
          f"short {sum(1 for r in results if r['direction']=='short')})")
    print(f"Exit breakdown:     {dict(how)}")
    print(f"Win rate:           {win_rate*100:.1f}%")
    print(f"Avg win:            +{avg_win_r:.2f}R    Avg loss: -{avg_loss_r:.2f}R")
    print(f"Profit factor:      {profit_factor:.2f}")
    print(f"Expectancy:         {expectancy_r:+.3f}R per trade")
    print(f"Total:              {total_r:+.1f}R")
    print()
    print(f"Fixed-risk sim:     {args.risk_per_trade*100:.1f}% risk/trade "
          f"(${risk_dollars:.0f}) on ${args.capital:.0f}")
    print(f"  Net P&L:          ${net_dollars:+,.0f}  ({pct_return:+.1f}%)  "
          f"<-- IGNORES concurrency; NOT a real account return")
    print(f"  ({n} trades at {args.risk_per_trade*100:.1f}% each = "
          f"{n*args.risk_per_trade*100:.0f}% nominal risk, impossible on one account)")
    print()

    # SPY buy-and-hold over the same calendar window (context, not a fair A/B)
    try:
        spy = yf.download("SPY", start=dmin, end=dmax, interval="1d",
                          progress=False, auto_adjust=True)
        if spy.columns.nlevels > 1:
            spy.columns = spy.columns.get_level_values(0)
        c = spy["Close"].dropna()
        spy_ret = (float(c.iloc[-1]) / float(c.iloc[0]) - 1) * 100
        print(f"SPY buy-and-hold over {dmin}→{dmax}: {spy_ret:+.1f}% (one flat month)")
    except Exception as e:
        print(f"(Could not fetch SPY benchmark: {e})")

    print()
    print("HONEST READ: the edge estimate is the per-trade expectancy "
          f"({expectancy_r:+.3f}R).")
    print("  A positive one-month, single-regime expectancy is encouraging but NOT")
    print("  conclusive. The authoritative, capital/execution-aware number is the")
    print("  2-year walk-forward (scripts/run_walkforward_v2.py): ~PF 1.44, ~6%/yr,")
    print("  which TRAILS SPY buy-and-hold (~20%/yr). Do not quote the dollar figure.")
    print("=" * 78)


if __name__ == "__main__":
    main()
