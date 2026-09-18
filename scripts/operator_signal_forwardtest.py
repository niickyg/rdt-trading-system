#!/usr/bin/env python3
"""Operator honest forward-test of recorded RRS signals.

Turns the untracked signal snapshots in data/signals/signal_history.json into
*measured* outcomes using real historical daily bars, so the system can be
judged on actual P&L net of honest costs versus SPY buy-and-hold.

Methodology (deliberately conservative):
  * Dedupe raw snapshots to unique trade ideas by (symbol, direction,
    rounded entry_price), keeping the earliest generated_at.
  * Entry fill = OPEN of the first trading session strictly after the
    signal's generated_at date (signals are generated after-hours, so they
    are actionable next session). No look-ahead on the signal price.
  * Walk forward up to MAX_HOLD_DAYS sessions of daily bars:
      - long:  low <= stop  -> stopped out at stop
               high >= target -> target hit at target
               if a single bar spans both, assume STOP first (worst case)
      - short: mirror image
      - neither within the window -> mark-to-market exit at the last close
  * Costs: round-trip COST_BPS applied to every trade.
  * Equal-dollar per trade. Reported as average per-trade net return and as a
    naive equal-weight aggregate. This is an approximation using DAILY bars,
    not intraday fills — read the caveats in the operator journal entry.

Not a promise of live performance. A floor-level reality check that the repo
could not previously produce because outcomes were never recorded.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
HISTORY = ROOT / "data" / "signals" / "signal_history.json"

MAX_HOLD_DAYS = 10
COST_BPS = 10.0  # 0.10% round-trip (commission + slippage) on liquid names


def load_unique_signals() -> list[dict]:
    raw = json.loads(HISTORY.read_text())
    uniq: dict[tuple, dict] = {}
    for s in raw:
        try:
            entry = round(float(s.get("entry_price") or 0), 2)
        except (TypeError, ValueError):
            continue
        if not (entry and s.get("stop_price") and s.get("target_price")):
            continue
        key = (s.get("symbol"), s.get("direction"), entry)
        g = s.get("generated_at", "")
        if key not in uniq or g < uniq[key].get("generated_at", ""):
            uniq[key] = s
    return list(uniq.values())


def fetch_bars(symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    # auto_adjust=False: raw OHLC that matches the signals' raw entry/stop/target.
    # Adjusted prices would spuriously trip stops on dividend payers / splits.
    data = yf.download(
        symbols, start=start, end=end, progress=False,
        auto_adjust=False, group_by="ticker", threads=True,
    )
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = data[sym] if len(symbols) > 1 else data
            df = df.dropna(subset=["Open", "High", "Low", "Close"])
            if len(df):
                out[sym] = df
        except (KeyError, Exception):
            continue
    return out


def evaluate(sig: dict, bars: pd.DataFrame) -> dict | None:
    gen = sig.get("generated_at", "")[:10]
    try:
        gen_date = datetime.strptime(gen, "%Y-%m-%d").date()
    except ValueError:
        return None
    future = bars[bars.index.date > gen_date]
    if len(future) < 1:
        return None

    direction = sig["direction"]
    entry = float(future.iloc[0]["Open"])  # fill at next session open
    stop = float(sig["stop_price"])
    target = float(sig["target_price"])
    if entry <= 0:
        return None

    # Split/data-artifact guard: yfinance always split-adjusts OHLC, but the
    # signal's raw entry/stop/target are pre-split. If the next-session open
    # diverges wildly from the recorded signal entry, the bars are on a
    # different price basis (e.g. DD reverse split) -> unmeasurable, skip.
    sig_entry = float(sig.get("entry_price") or 0)
    if sig_entry > 0:
        ratio = entry / sig_entry
        if ratio < 0.6 or ratio > 1.6:
            return None

    # Realistic gap fills: if a bar OPENS beyond the level, fill at the open
    # (a gap-down through a long stop fills worse than the stop; a gap-up
    # through a target fills better). Only when price merely trades through
    # the level intrabar do we assume a fill at the level itself.
    outcome, exit_px = "timeout", None
    window = future.iloc[:MAX_HOLD_DAYS]
    for _, bar in window.iterrows():
        op, hi, lo = float(bar["Open"]), float(bar["High"]), float(bar["Low"])
        if direction == "long":
            if lo <= stop:
                outcome, exit_px = "stop", min(op, stop)
                break
            if hi >= target:
                outcome, exit_px = "target", max(op, target)
                break
        else:  # short
            if hi >= stop:
                outcome, exit_px = "stop", max(op, stop)
                break
            if lo <= target:
                outcome, exit_px = "target", min(op, target)
                break
    if exit_px is None:
        exit_px = float(window.iloc[-1]["Close"])

    gross = (exit_px - entry) / entry if direction == "long" else (entry - exit_px) / entry
    net = gross - COST_BPS / 10000.0
    return {"symbol": sig["symbol"], "direction": direction, "outcome": outcome,
            "gross": gross, "net": net}


def main() -> int:
    sigs = load_unique_signals()
    if not sigs:
        print("No valid signals with entry/stop/target found.")
        return 1
    dates = sorted(s.get("generated_at", "")[:10] for s in sigs)
    start = (datetime.strptime(dates[0], "%Y-%m-%d").date() - timedelta(days=3)).isoformat()
    end = (datetime.strptime(dates[-1], "%Y-%m-%d").date() + timedelta(days=MAX_HOLD_DAYS * 2 + 10)).isoformat()
    symbols = sorted({s["symbol"] for s in sigs})
    print(f"Unique trade ideas: {len(sigs)} across {len(symbols)} symbols")
    print(f"Signal date span: {dates[0]} .. {dates[-1]}  | bar window {start}..{end}")
    print(f"Costs: {COST_BPS} bps round-trip | max hold {MAX_HOLD_DAYS} sessions\n")

    bars = fetch_bars(symbols + ["SPY"], start, end)
    print(f"Fetched bars for {len(bars)}/{len(symbols) + 1} symbols\n")

    results, skipped = [], 0
    for s in sigs:
        b = bars.get(s["symbol"])
        if b is None:
            skipped += 1
            continue
        r = evaluate(s, b)
        if r is None:
            skipped += 1
            continue
        results.append(r)

    if not results:
        print("No evaluable trades (data gaps).")
        return 1

    nets = [r["net"] for r in results]
    grosses = [r["gross"] for r in results]
    oc = Counter(r["outcome"] for r in results)
    wins = sum(1 for n in nets if n > 0)

    print("=" * 60)
    print(f"Evaluated trades: {len(results)}  (skipped {skipped})")
    print(f"Outcomes: {dict(oc)}")
    print(f"Win rate (net>0): {wins/len(nets)*100:.1f}%")
    print(f"Avg net return/trade:   {statistics.mean(nets)*100:+.3f}%")
    print(f"Median net return/trade:{statistics.median(nets)*100:+.3f}%")
    print(f"Avg gross return/trade: {statistics.mean(grosses)*100:+.3f}%")
    print(f"Sum of net returns (equal-$ book): {sum(nets)*100:+.2f}%")
    print(f"Best/Worst: {max(nets)*100:+.2f}% / {min(nets)*100:+.2f}%")

    # SPY buy-and-hold over the signal span
    spy = bars.get("SPY")
    if spy is not None and len(spy) > 1:
        s0, s1 = float(spy.iloc[0]["Close"]), float(spy.iloc[-1]["Close"])
        print(f"\nSPY buy-and-hold over window: {(s1/s0-1)*100:+.2f}%  ({s0:.2f} -> {s1:.2f})")
        # per-trade profit factor
        gains = sum(n for n in nets if n > 0)
        losses = -sum(n for n in nets if n < 0)
        pf = gains / losses if losses else float("inf")
        print(f"Profit factor (net): {pf:.2f}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
