#!/usr/bin/env python3
"""Honest forward-return evaluation of historically generated RRS signals.

WHY THIS EXISTS
---------------
The production system logged 1986 signal snapshots to
data/signals/signal_history.json but only ever recorded *2* outcomes in
signal_metrics.json. In other words, the bot has generated signals for months
without ever measuring whether they made money. This script closes that gap: it
replays every real signal against actual forward prices (yfinance) and reports an
honest, reproducible edge estimate with confidence intervals and a SPY benchmark.

It is intentionally conservative:
  * Entry = OPEN of the first regular session on/after the signal date. The scanner
    runs overnight / after-hours, so the recorded `entry_price` is not fillable; the
    next regular-session open is the realistic fill.
  * Bracket exit: each day, check STOP first then TARGET (stop-first on same-bar
    ambiguity = conservative). If neither is hit within --horizon trading days, exit
    at that day's close.
  * Slippage applied against us on both entry and exit (--slip bps per side).
  * R-multiple winsorized to [-3, 5]: when the next open gaps to near the stop the
    raw risk denominator becomes tiny and produces absurd R outliers that are an
    artifact of the fill model, not real edge.
  * Benchmark: SPY bought at the same entry open and sold at the matched exit date.

Usage:
  python scripts/evaluate_signal_history.py [--horizon 10] [--slip 10] [--json out.json]

Requires: yfinance, pandas, numpy. Read-only; places no orders.
"""
import argparse
import json
import sys
from collections import OrderedDict, Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "data" / "signals" / "signal_history.json"


def load_signals():
    raw = json.loads(HISTORY.read_text())
    sig = OrderedDict()
    for x in raw:
        g = (x.get("generated_at") or "")[:10]
        key = (x.get("symbol"), x.get("direction"), g)
        if key not in sig and x.get("entry_price") and x.get("stop_price") and x.get("target_price"):
            sig[key] = x
    return list(sig.values()), len(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=10, help="max trading days to hold")
    ap.add_argument("--slip", type=float, default=10.0, help="slippage in bps per side")
    ap.add_argument("--start", default="2026-02-01")
    ap.add_argument("--end", default="2026-04-20")
    ap.add_argument("--json", default=None, help="write machine-readable results here")
    args = ap.parse_args()

    try:
        import numpy as np
        import pandas as pd
        import yfinance as yf
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}. pip install yfinance pandas numpy")

    slip = args.slip / 1e4
    signals, n_raw = load_signals()
    syms = sorted({s["symbol"] for s in signals}) + ["SPY"]
    print(f"Deduped {n_raw} snapshots -> {len(signals)} unique (symbol,dir,day) signals, "
          f"{len(syms)-1} symbols. horizon={args.horizon}d slip={args.slip:.0f}bps/side\n")

    data = yf.download(syms, start=args.start, end=args.end, progress=False, auto_adjust=False)

    def bars(sym):
        try:
            df = pd.DataFrame({
                "open": data["Open"][sym], "high": data["High"][sym],
                "low": data["Low"][sym], "close": data["Close"][sym],
            }).dropna()
        except Exception:
            return None
        return df if not df.empty else None

    spy = bars("SPY")

    def evaluate(s):
        sym, d, sd = s["symbol"], s["direction"], (s["generated_at"] or "")[:10]
        stop, tgt = float(s["stop_price"]), float(s["target_price"])
        df = bars(sym)
        if df is None:
            return None
        sday = datetime.strptime(sd, "%Y-%m-%d").date()
        future = df[df.index.date >= sday]
        if len(future) < 2:
            return None
        long = d == "long"
        entry_open = float(future.iloc[0]["open"])
        entry = entry_open * (1 + slip) if long else entry_open * (1 - slip)
        risk = abs(entry - stop)
        if risk <= 0:
            return None
        window = future.iloc[1:args.horizon + 1]
        exit_px = exit_dt = reason = None
        for dt, row in window.iterrows():
            hi, lo = float(row["high"]), float(row["low"])
            if long:
                if lo <= stop: exit_px, reason, exit_dt = stop, "stop", dt; break
                if hi >= tgt: exit_px, reason, exit_dt = tgt, "target", dt; break
            else:
                if hi >= stop: exit_px, reason, exit_dt = stop, "stop", dt; break
                if lo <= tgt: exit_px, reason, exit_dt = tgt, "target", dt; break
        if exit_px is None:
            exit_px, reason, exit_dt = float(window.iloc[-1]["close"]), "time", window.index[-1]
        exit_fill = exit_px * (1 - slip) if long else exit_px * (1 + slip)
        pnl = (exit_fill - entry) if long else (entry - exit_fill)
        spy_ret = None
        if spy is not None:
            sf = spy[spy.index.date >= sday]
            if len(sf) >= 2:
                se = float(sf.iloc[0]["open"])
                match = sf[sf.index <= exit_dt]
                sx = float(match.iloc[-1]["close"]) if len(match) else float(sf.iloc[1]["close"])
                spy_ret = (sx - se) / se
        return dict(sym=sym, dir=d, date=sd, reason=reason, R=pnl / risk,
                    ret=pnl / entry, spy=spy_ret,
                    hold=(exit_dt.date() - future.index[0].date()).days)

    res = [r for s in signals if (r := evaluate(s))]
    if not res:
        sys.exit("No signals could be evaluated (no price data).")

    R = np.clip([r["R"] for r in res], -3, 5)
    rets = np.array([r["ret"] for r in res])
    paired = [(r["ret"], r["spy"]) for r in res if r["spy"] is not None]
    sig_r = np.array([p[0] for p in paired])
    spy_r = np.array([p[1] for p in paired])
    wins = int((rets > 0).sum())
    gw, gl = rets[rets > 0].sum(), -rets[rets < 0].sum()
    pf = gw / gl if gl > 0 else float("inf")

    rng = np.random.default_rng(42)
    boot = np.array([rng.choice(rets, len(rets), replace=True).mean() for _ in range(10000)])
    alpha = sig_r - spy_r
    boota = np.array([rng.choice(alpha, len(alpha), replace=True).mean() for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    alo, ahi = np.percentile(boota, [2.5, 97.5])

    print("=== BRACKET BACKTEST (per-trade, equal weight) ===")
    print(f"Trades:              {len(res)}")
    print(f"Win rate:            {wins/len(res)*100:.1f}%  ({wins}W / {len(res)-wins}L)")
    print(f"Avg R (winsorized):  {R.mean():+.3f}R   median {np.median(R):+.3f}R")
    print(f"Avg return/trade:    {rets.mean()*100:+.3f}%  (after {args.slip:.0f}bps slip/side)")
    print(f"Profit factor:       {pf:.2f}")
    print(f"Exit reasons:        {dict(Counter(r['reason'] for r in res))}")
    print(f"By direction:        {dict(Counter(r['dir'] for r in res))}")
    print(f"Median hold:         {int(np.median([r['hold'] for r in res]))} cal.days")
    print("\n=== VS SPY (matched windows) ===")
    print(f"Avg signal ret:      {sig_r.mean()*100:+.3f}%")
    print(f"Avg SPY ret:         {spy_r.mean()*100:+.3f}%")
    print(f"Avg alpha:           {alpha.mean()*100:+.3f}%")
    print(f"Beat SPY:            {int((sig_r>spy_r).sum())}/{len(paired)} ({(sig_r>spy_r).mean()*100:.0f}%)")
    print("\n=== SIGNIFICANCE (10k bootstrap) ===")
    print(f"Mean ret/trade 95% CI: [{lo*100:+.3f}%, {hi*100:+.3f}%]  one-sided p(mean<=0)={(boot<=0).mean():.3f}")
    print(f"Mean alpha 95% CI:     [{alo*100:+.3f}%, {ahi*100:+.3f}%]  one-sided p(alpha<=0)={(boota<=0).mean():.3f}")
    print("\nCAVEAT: per-trade stats, not a portfolio curve (sizing/overlap ignored). "
          "Sample is tiny and single-regime; treat as a hypothesis, not proof.")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "n_trades": len(res), "win_rate": wins / len(res), "avg_R": float(R.mean()),
            "avg_ret": float(rets.mean()), "profit_factor": pf,
            "avg_alpha": float(alpha.mean()), "ret_ci": [float(lo), float(hi)],
            "p_ret_le_0": float((boot <= 0).mean()), "horizon": args.horizon,
            "slip_bps": args.slip, "trades": res,
        }, indent=2, default=str))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
