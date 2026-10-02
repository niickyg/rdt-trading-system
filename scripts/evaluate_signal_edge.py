#!/usr/bin/env python3
"""
Signal Edge Evaluator — measure the REALIZED performance of live-generated signals.

This is the operator's ground-truth tool. Unlike the walk-forward backtest (which
re-simulates a strategy on historical data and is easy to over-fit), this tool takes
the signals the bot ACTUALLY emitted (persisted in data/signals/signal_history.json)
and asks a single honest question: if you had taken each signal at its stated entry,
stop, and target, did it make money?

Method (first-touch, daily bars):
  For each unique signal setup (symbol, direction, date, entry), walk forward up to
  HORIZON trading days. On each day, if the daily range touches the stop -> -1R; if it
  touches the target -> +(target-entry)/risk R. If a day touches both, assume the stop
  filled first (conservative). If neither is touched within the horizon, mark-to-close
  at the horizon's last bar.

Caveats (read before trusting the number):
  * Daily bars cannot see intraday path; a same-day touch of both stop and target is
    scored as a stop (pessimistic by design).
  * Uses the signal's stated entry_price as the fill. Real fills include slippage and
    gap risk, so live results are typically WORSE, not better, than this estimate.
  * No commissions/fees modeled. The R figures are gross.
  * Price data is adjusted (splits/divs). Short borrow costs are not modeled.

Usage:
    python scripts/evaluate_signal_edge.py                 # default horizon 15d
    python scripts/evaluate_signal_edge.py --horizon 5     # intraday/swing horizon
    python scripts/evaluate_signal_edge.py --no-cache      # force fresh download

Requires: yfinance, pandas, numpy. Writes a CSV + JSON summary to
data/operator_journal/evidence/.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HISTORY = PROJECT_ROOT / "data" / "signals" / "signal_history.json"
EVIDENCE_DIR = PROJECT_ROOT / "data" / "operator_journal" / "evidence"
CACHE = Path(os.environ.get("TMPDIR", "/tmp")) / "rdt_signal_prices.pkl"


def load_unique_signals():
    raw = json.loads(HISTORY.read_text())
    seen, sigs = set(), []
    for s in raw:
        req = ("symbol", "direction", "entry_price", "stop_price",
               "target_price", "generated_at")
        if not all(s.get(k) is not None for k in req):
            continue
        key = (s["symbol"], s["direction"], s["generated_at"][:10],
               round(float(s["entry_price"]), 2))
        if key in seen:
            continue
        seen.add(key)
        sigs.append(s)
    return sigs


def download_prices(symbols, start, end, use_cache=True):
    import yfinance as yf
    if use_cache and CACHE.exists():
        try:
            return pd.read_pickle(CACHE)
        except Exception:
            pass
    frames = {}
    for sym in symbols:
        for attempt in range(4):
            try:
                d = yf.download(sym, start=start, end=end, progress=False,
                                auto_adjust=True)
                if len(d) > 0:
                    d.columns = [c[0] if isinstance(c, tuple) else c
                                 for c in d.columns]
                    frames[sym] = d[["Open", "High", "Low", "Close"]]
                    break
            except Exception:
                pass
            time.sleep(3)
    if not frames:
        raise RuntimeError("No price data could be downloaded (rate limited?).")
    data = pd.concat(frames, axis=1)
    try:
        data.to_pickle(CACHE)
    except Exception:
        pass
    print(f"  downloaded {len(frames)}/{len(symbols)} symbols")
    return data


def simulate(sigs, data, horizon):
    rows = []
    for s in sigs:
        try:
            d = data[s["symbol"]].dropna()
        except Exception:
            continue
        if d.empty:
            continue
        gdate = pd.Timestamp(s["generated_at"][:10])
        fwd = d[d.index.date > gdate.date()].iloc[:horizon]
        if fwd.empty:
            continue
        entry = float(s["entry_price"])
        stop = float(s["stop_price"])
        tgt = float(s["target_price"])
        long = s["direction"] == "long"
        risk = abs(entry - stop)
        if risk <= 0:
            continue
        outcome, rmult = None, None
        for _, bar in fwd.iterrows():
            hi, lo = float(bar["High"]), float(bar["Low"])
            hit_stop = (lo <= stop) if long else (hi >= stop)
            hit_tgt = (hi >= tgt) if long else (lo <= tgt)
            if hit_stop:  # checked first => conservative on same-day ambiguity
                outcome, rmult = "stop", -1.0
                break
            if hit_tgt:
                outcome, rmult = "target", abs(tgt - entry) / risk
                break
        if outcome is None:
            last = float(fwd.iloc[-1]["Close"])
            pnl = (last - entry) if long else (entry - last)
            outcome, rmult = "horizon", pnl / risk
        rows.append(dict(
            symbol=s["symbol"], direction=s["direction"],
            stock_change_pct=s.get("stock_change_pct"), rrs=s.get("rrs"),
            strength=s.get("strength"), rr_ratio=round(abs(tgt - entry) / risk, 2),
            outcome=outcome, R=round(rmult, 3),
        ))
    return pd.DataFrame(rows)


def report(df, horizon):
    def agg(g):
        return g.agg(n=("R", "size"),
                     winrate=("R", lambda x: round((x > 0).mean() * 100, 1)),
                     expR=("R", lambda x: round(x.mean(), 3)),
                     totalR=("R", lambda x: round(x.sum(), 1)))
    out = []
    out.append(f"Horizon: {horizon} trading days | signals evaluated: {len(df)}")
    out.append(f"Win rate: {(df.R > 0).mean() * 100:.1f}%  "
               f"(breakeven needs 1/(1+avg_rr) = "
               f"{100 / (1 + df.rr_ratio.mean()):.1f}% at avg R:R {df.rr_ratio.mean():.2f})")
    out.append(f"Expectancy: {df.R.mean():+.3f}R/trade | total: {df.R.sum():+.1f}R")
    out.append(f"Outcomes: {dict(Counter(df.outcome))}")
    out.append("\nBy direction:\n" + agg(df.groupby("direction")).to_string())
    if df.stock_change_pct.notna().any():
        df = df.copy()
        df["abschg"] = df.stock_change_pct.abs()
        df["extension"] = pd.cut(df.abschg, [0, 3, 5, 8, 100],
                                 labels=["0-3%", "3-5%", "5-8%", "8%+"])
        out.append("\nBy entry extension (|intraday move| at signal):\n"
                   + agg(df.groupby("extension", observed=True)).to_string())
    out.append("\nBy strength:\n" + agg(df.groupby("strength")).to_string())
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=15,
                    help="forward trading-day horizon (default 15)")
    ap.add_argument("--no-cache", action="store_true",
                    help="force fresh price download")
    args = ap.parse_args()

    sigs = load_unique_signals()
    if not sigs:
        print("No usable signals in history.")
        return 1
    symbols = sorted({s["symbol"] for s in sigs})
    dates = [s["generated_at"][:10] for s in sigs]
    start = (datetime.fromisoformat(min(dates)) - timedelta(days=7)).date().isoformat()
    end = (datetime.fromisoformat(max(dates)) + timedelta(days=45)).date().isoformat()
    print(f"Evaluating {len(sigs)} unique signals on {len(symbols)} symbols "
          f"({min(dates)} .. {max(dates)})")

    data = download_prices(symbols, start, end, use_cache=not args.no_cache)
    df = simulate(sigs, data, args.horizon)
    if df.empty:
        print("No signals could be evaluated (missing price data).")
        return 1

    text = report(df, args.horizon)
    print("\n" + text)

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    df.to_csv(EVIDENCE_DIR / f"signal_outcomes_{stamp}_h{args.horizon}.csv",
              index=False)
    summary = dict(
        generated=datetime.now().isoformat(), horizon=args.horizon,
        n=len(df), win_rate_pct=round((df.R > 0).mean() * 100, 1),
        expectancy_R=round(df.R.mean(), 3), total_R=round(df.R.sum(), 1),
        signal_window=f"{min(dates)}..{max(dates)}",
    )
    (EVIDENCE_DIR / f"signal_edge_summary_{stamp}_h{args.horizon}.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\nWrote evidence to {EVIDENCE_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
