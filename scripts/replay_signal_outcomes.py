#!/usr/bin/env python3
"""
Replay historical signals against actual market data to measure REALIZED edge.

Why this exists
---------------
The bot generates signals into ``data/signals/signal_history.json`` but has
historically tracked the realized outcome of almost none of them
(``signal_metrics.json`` recorded only 2 outcomes against ~2000 signals).
You cannot manage — or improve — what you do not measure. This tool closes that
gap: it takes the signals the system actually produced and asks, honestly, "did
they make money?" by walking each one forward against real OHLC bars.

Methodology & honest caveats
----------------------------
* Deduplicates to one trade per (symbol, date, direction) so repeated intra-day
  re-signals of the same setup are not double-counted.
* First-touch simulation on DAILY bars. The live strategy is intraday (5-minute
  entries); intraday history is not available from free providers more than
  ~60 days back, so for older signals this is necessarily a *daily-bar swing
  approximation*, not a faithful intraday replay. Treat it as directional
  evidence, not ground truth.
* Tie-break: if a single day's range spans BOTH stop and target, the outcome is
  scored as a STOP (conservative — never flatters the strategy).
* ``--cost`` applies a per-trade friction in R (commission + spread + slippage).
  For liquid US stocks 0.05-0.10R is a reasonable base case; higher for thin
  intraday fills or shorts with borrow cost.
* The position-capped portfolio sim models the real constraint that you can only
  hold N concurrent positions, so clustered signals compete for slots.

Returns a nonzero exit code only on hard errors, never on a "bad" result.

Usage
-----
    python3 scripts/replay_signal_outcomes.py                 # defaults
    python3 scripts/replay_signal_outcomes.py --fwd 10 --cost 0.10 --max-positions 8
    python3 scripts/replay_signal_outcomes.py --json out.json

Requires: pandas, numpy, yfinance (network access for price history).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_signals(path: Path) -> list[dict]:
    with path.open() as fh:
        raw = json.load(fh)
    seen: dict[tuple, dict] = {}
    for s in raw:
        sym, direction, generated = s.get("symbol"), s.get("direction"), s.get("generated_at")
        entry, stop, target = s.get("entry_price"), s.get("stop_price"), s.get("target_price")
        if not all([sym, direction, generated]) or None in (entry, stop, target):
            continue
        if entry <= 0 or stop <= 0 or target <= 0:
            continue
        key = (sym, generated[:10], direction)
        seen.setdefault(key, dict(
            symbol=sym, date=generated[:10], direction=direction,
            entry=float(entry), stop=float(stop), target=float(target),
        ))
    return list(seen.values())


def run(trades, data, fwd_days, cost_r):
    import pandas as pd

    def ohlc(sym):
        try:
            return data[sym][["Open", "High", "Low", "Close"]].dropna()
        except Exception:
            return None

    results = []
    for t in trades:
        df = ohlc(t["symbol"])
        if df is None or df.empty:
            continue
        fwd = df[df.index >= pd.Timestamp(t["date"])].iloc[:fwd_days]
        if fwd.empty:
            continue
        entry, stop, target, d = t["entry"], t["stop"], t["target"], t["direction"]
        risk = abs(entry - stop)
        if risk <= 0:
            continue
        outcome = exit_px = None
        exit_date = str(fwd.index[-1].date())
        for ts, bar in fwd.iterrows():
            hi, lo = bar["High"], bar["Low"]
            hit_stop = lo <= stop if d == "long" else hi >= stop
            hit_tgt = hi >= target if d == "long" else lo <= target
            if hit_stop:  # checked before target => conservative tie-break
                outcome, exit_px, exit_date = "stop", stop, str(ts.date())
                break
            if hit_tgt:
                outcome, exit_px, exit_date = "target", target, str(ts.date())
                break
        if outcome is None:
            outcome, exit_px = "timeout", float(fwd.iloc[-1]["Close"])
        pnl = (exit_px - entry) if d == "long" else (entry - exit_px)
        results.append(dict(**t, outcome=outcome, exit=float(exit_px),
                            exit_date=exit_date, R=float(pnl / risk - cost_r)))
    return results


def summarize(results):
    import numpy as np
    R = np.array([x["R"] for x in results]) if results else np.array([])
    if len(R) == 0:
        return dict(trades=0)
    wins, losses = R[R > 0], R[R <= 0]
    pf = (wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf")
    return dict(
        trades=int(len(R)), win_rate=float((R > 0).mean()),
        avg_R=float(R.mean()), profit_factor=float(pf), sum_R=float(R.sum()),
        outcomes=dict(Counter(x["outcome"] for x in results)),
        directions=dict(Counter(x["direction"] for x in results)),
    )


def portfolio_sim(results, max_positions, risk_frac, start_equity):
    ordered = sorted(results, key=lambda x: x["date"])
    equity = start_equity
    open_until: list[str] = []
    taken = 0
    for x in ordered:
        open_until = [e for e in open_until if e > x["date"]]
        if len(open_until) >= max_positions:
            continue
        open_until.append(x["exit_date"])
        equity += equity * risk_frac * x["R"]
        taken += 1
    return dict(final_equity=equity, pct=(equity / start_equity - 1) * 100,
                taken=taken, total=len(results))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signals", default=None, help="Path to signal_history.json")
    ap.add_argument("--fwd", type=int, default=10, help="Max trading days to hold (default 10)")
    ap.add_argument("--cost", type=float, default=0.10, help="Per-trade friction in R (default 0.10)")
    ap.add_argument("--max-positions", type=int, default=8, help="Concurrent position cap (default 8)")
    ap.add_argument("--risk-frac", type=float, default=0.01, help="Risk per trade as fraction of equity")
    ap.add_argument("--account", type=float, default=25000.0, help="Starting equity for portfolio sim")
    ap.add_argument("--start", default="2026-02-01", help="Price history start (YYYY-MM-DD)")
    ap.add_argument("--end", default="2026-05-01", help="Price history end (YYYY-MM-DD)")
    ap.add_argument("--json", default=None, help="Write machine-readable results to this path")
    args = ap.parse_args()

    try:
        import pandas as pd  # noqa: F401
        import numpy as np  # noqa: F401
        import yfinance as yf
    except ImportError as exc:
        print(f"ERROR: missing dependency ({exc}). Install pandas numpy yfinance.", file=sys.stderr)
        return 2

    sig_path = Path(args.signals) if args.signals else project_root() / "data/signals/signal_history.json"
    if not sig_path.exists():
        print(f"ERROR: signals file not found: {sig_path}", file=sys.stderr)
        return 2

    trades = load_signals(sig_path)
    if not trades:
        print("No usable signals found.", file=sys.stderr)
        return 2
    symbols = sorted({t["symbol"] for t in trades})
    print(f"Deduped trades: {len(trades)} across {len(symbols)} symbols "
          f"({min(t['date'] for t in trades)} .. {max(t['date'] for t in trades)})")

    data = yf.download(symbols + ["SPY"], start=args.start, end=args.end,
                       progress=False, auto_adjust=False, group_by="ticker")

    results = run(trades, data, args.fwd, args.cost)
    stats = summarize(results)
    print(f"\n=== REALIZED-SIGNAL REPLAY (fwd={args.fwd}d, cost={args.cost}R) ===")
    print(f"  Evaluated      : {stats['trades']}")
    print(f"  Win rate       : {stats['win_rate']*100:.1f}%")
    print(f"  Expectancy     : {stats['avg_R']:+.3f} R/trade")
    print(f"  Profit factor  : {stats['profit_factor']:.3f}")
    print(f"  Outcomes       : {stats['outcomes']}")
    print(f"  Directions     : {stats['directions']}")

    port = portfolio_sim(results, args.max_positions, args.risk_frac, args.account)
    print(f"\n=== POSITION-CAPPED PORTFOLIO (max {args.max_positions}, "
          f"{args.risk_frac*100:.0f}% risk/trade) ===")
    print(f"  Signals taken  : {port['taken']}/{port['total']} (capacity constrained)")
    print(f"  Final equity   : ${port['final_equity']:,.0f}  ({port['pct']:+.2f}% over window)")

    import pandas as pd
    try:
        spy = data["SPY"][["Close"]].dropna()
        ws = pd.Timestamp(min(t["date"] for t in trades))
        spyw = spy[spy.index >= ws]
        spy_ret = (spyw.iloc[-1]["Close"] / spyw.iloc[0]["Close"] - 1) * 100
        print(f"  SPY buy & hold : {spy_ret:+.2f}% "
              f"({ws.date()}..{spyw.index[-1].date()})")
    except Exception:
        spy_ret = None

    if args.json:
        payload = dict(params=vars(args), stats=stats, portfolio=port, spy_bh_pct=spy_ret)
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
