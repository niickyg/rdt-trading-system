#!/usr/bin/env python3
"""
Signal outcome evaluator — the missing measurement layer.

The system generates thousands of signals (data/signals/signal_history.json)
but records almost no *outcomes*, so it cannot answer the only question that
matters: are the signals profitable, and do they beat SPY buy-and-hold?

This script closes that loop. For each historical signal it fetches real daily
OHLC bars (Yahoo Finance, daily), simulates the trade against the signal's own
entry/stop/target with a *conservative* path assumption, and reports realised
expectancy in R-multiples plus a SPY buy-and-hold benchmark over the same window.

It is deliberately dependency-light (stdlib only; honours HTTPS_PROXY) so it runs
in constrained/CI environments without pandas/numpy.

LIMITATIONS (read before trusting the numbers):
  * Daily bars, not intraday. RDT is an intraday methodology, so exits are an
    approximation. Path dependency within a day is unknowable from a daily bar,
    so when a single bar's range straddles BOTH stop and target we assume the
    STOP filled first. This biases results PESSIMISTIC on purpose — an honest
    evaluator should not flatter the strategy.
  * Uses the signal's recorded entry_price as the fill (no slippage/commission
    unless --cost-bps given). Real fills differ.
  * A signal whose forward window has no market data yet (future-dated) is
    skipped and counted as "unresolvable". If everything is unresolvable, that
    is itself the finding: you are evaluating signals for dates that haven't
    happened.

Usage:
  python scripts/evaluate_signals.py --history data/signals/signal_history.json
  python scripts/evaluate_signals.py --max-hold 10 --cost-bps 5 --limit-symbols 60
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
CACHE_DIR = os.environ.get("EVAL_CACHE_DIR", "/tmp/eval_signals_cache")


def _parse_dt(s: str) -> datetime:
    """Parse ISO timestamps, tolerating trailing Z and offsets; return UTC-naive date anchor."""
    s = s.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def fetch_daily(symbol: str, start: datetime, end: datetime, retries: int = 3):
    """Return list of (date, open, high, low, close) daily bars from Yahoo, cached to disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    p1 = int(start.replace(tzinfo=timezone.utc).timestamp())
    p2 = int(end.replace(tzinfo=timezone.utc).timestamp())
    cache = os.path.join(CACHE_DIR, f"{symbol}_{p1}_{p2}.json")
    raw = None
    if os.path.exists(cache) and os.path.getsize(cache) > 50:
        raw = open(cache).read()
    else:
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
               f"?period1={p1}&period2={p2}&interval=1d")
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                raw = urllib.request.urlopen(req, timeout=30).read().decode()
                open(cache, "w").write(raw)
                break
            except Exception as e:  # noqa: BLE001 - network best-effort
                if attempt == retries - 1:
                    print(f"  ! fetch failed {symbol}: {e}", file=sys.stderr)
                    return []
                time.sleep(2.0 * (attempt + 1))
    try:
        d = json.loads(raw)
        res = d["chart"]["result"][0]
        ts = res["timestamp"]
        q = res["indicators"]["quote"][0]
        bars = []
        for i, t in enumerate(ts):
            o, h, low, c = q["open"][i], q["high"][i], q["low"][i], q["close"][i]
            if None in (o, h, low, c):
                continue
            bars.append((datetime.utcfromtimestamp(t).date(), o, h, low, c))
        return bars
    except Exception:  # noqa: BLE001
        return []


def simulate(signal: dict, bars, max_hold: int, cost_bps: float):
    """Simulate one signal on daily bars. Returns R-multiple realised, or None if unresolvable."""
    try:
        entry = float(signal["entry_price"])
        stop = float(signal["stop_price"])
        target = float(signal["target_price"])
        direction = signal["direction"]
        sig_date = _parse_dt(signal["generated_at"]).date()
    except (KeyError, ValueError, TypeError):
        return None

    risk = abs(entry - stop)
    if risk <= 0:
        return None

    fwd = [b for b in bars if b[0] >= sig_date][:max_hold]
    if not fwd:
        return None  # unresolvable: no forward data (e.g. future-dated signal)

    cost_r = (entry * cost_bps / 10000.0) / risk  # round-trip cost expressed in R

    for _, o, h, low, c in fwd:
        if direction == "long":
            hit_stop = low <= stop
            hit_tgt = h >= target
            if hit_stop and hit_tgt:
                return -1.0 - cost_r          # conservative: assume stop first
            if hit_stop:
                return -1.0 - cost_r
            if hit_tgt:
                return (target - entry) / risk - cost_r
        else:  # short
            hit_stop = h >= stop
            hit_tgt = low <= target
            if hit_stop and hit_tgt:
                return -1.0 - cost_r
            if hit_stop:
                return -1.0 - cost_r
            if hit_tgt:
                return (entry - target) / risk - cost_r
    # time exit at last available close
    last_close = fwd[-1][4]
    if direction == "long":
        return (last_close - entry) / risk - cost_r
    return (entry - last_close) / risk - cost_r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", default="data/signals/signal_history.json")
    ap.add_argument("--max-hold", type=int, default=10, help="max trading days held")
    ap.add_argument("--cost-bps", type=float, default=0.0,
                    help="round-trip cost in basis points of notional (slippage+commission)")
    ap.add_argument("--risk-frac", type=float, default=0.01,
                    help="fraction of equity risked per trade for the illustrative $ curve")
    ap.add_argument("--limit-symbols", type=int, default=100)
    ap.add_argument("--account", type=float, default=25000.0)
    args = ap.parse_args()

    signals = json.load(open(args.history))
    if isinstance(signals, dict):
        signals = next((v for v in signals.values() if isinstance(v, list)), [])
    print(f"Loaded {len(signals)} signals from {args.history}")

    dates = [_parse_dt(s["generated_at"]).date() for s in signals if s.get("generated_at")]
    if not dates:
        print("No dated signals; nothing to evaluate.")
        return 0
    start = datetime.combine(min(dates), datetime.min.time()) - timedelta(days=5)
    end = datetime.combine(max(dates), datetime.min.time()) + timedelta(days=args.max_hold * 2 + 10)
    print(f"Signal window {min(dates)} .. {max(dates)}; fetching bars {start.date()} .. {end.date()}")

    symbols = sorted({s["symbol"] for s in signals})[: args.limit_symbols]
    bars_by_sym = {}
    for i, sym in enumerate(symbols):
        bars_by_sym[sym] = fetch_daily(sym, start, end)
        if i % 10 == 0:
            print(f"  fetched {i+1}/{len(symbols)} symbols")
        time.sleep(0.8)

    results, unresolvable, no_data = [], 0, 0
    for s in signals:
        sym = s.get("symbol")
        bars = bars_by_sym.get(sym)
        if not bars:
            no_data += 1
            continue
        r = simulate(s, bars, args.max_hold, args.cost_bps)
        if r is None:
            unresolvable += 1
        else:
            results.append((s, r))

    # Non-overlapping trades: the realistic accounting. The same symbol is often
    # re-flagged every day it stays strong, so counting every raw signal as an
    # independent trade double-counts one position many times over and inflates
    # both trade count and P&L. Here we take one position per symbol at a time
    # and do not re-enter until the prior position's max-hold window has elapsed.
    nonoverlap = []
    by_sym = {}
    for s in signals:
        by_sym.setdefault(s.get("symbol"), []).append(s)
    for sym, slist in by_sym.items():
        bars = bars_by_sym.get(sym)
        if not bars:
            continue
        bdates = [b[0] for b in bars]
        slist = sorted(slist, key=lambda x: _parse_dt(x["generated_at"]).date())
        blocked_until = None
        for s in slist:
            d = _parse_dt(s["generated_at"]).date()
            if blocked_until and d < blocked_until:
                continue
            r = simulate(s, bars, args.max_hold, args.cost_bps)
            if r is None:
                continue
            nonoverlap.append(r)
            fwd = [x for x in bdates if x >= d][: args.max_hold]
            blocked_until = fwd[-1] if fwd else None

    print("\n" + "=" * 60)
    print("SIGNAL OUTCOME EVALUATION")
    print("=" * 60)
    print(f"Symbols evaluated:   {len(symbols)}")
    print(f"Signals total:       {len(signals)}")
    print(f"  no market data:    {no_data}")
    print(f"  unresolvable:      {unresolvable}  (no forward bars — often future-dated)")
    print(f"  resolved:          {len(results)}")

    if not results:
        print("\nNo resolvable signals. If these are future-dated, forward outcomes")
        print("do not exist yet and the strategy cannot be evaluated on this data.")
        return 0

    rs = [r for _, r in results]
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    gross_win = sum(wins)
    gross_loss = -sum(losses)
    expectancy = sum(rs) / len(rs)
    win_rate = len(wins) / len(rs)
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    # illustrative non-compounding $ curve at fixed fractional risk
    risk_dollars = args.account * args.risk_frac
    pnl_dollars = sum(rs) * risk_dollars

    print("\n-- Per raw signal (each signal counted independently; INFLATED, "
          "one position is re-counted every day it is re-flagged) --")
    print(f"Win rate:            {win_rate:6.1%}")
    print(f"Expectancy:          {expectancy:+.3f} R per trade")
    print(f"Profit factor:       {pf:.3f}")
    print(f"Avg win / avg loss:  {(sum(wins)/len(wins) if wins else 0):+.2f}R / "
          f"{(sum(losses)/len(losses) if losses else 0):+.2f}R")
    print(f"Sum of R:            {sum(rs):+.1f}R over {len(rs)} trades")
    print(f"Illustrative P&L:    ${pnl_dollars:,.0f} "
          f"(non-compounding, {args.risk_frac:.1%} risk = ${risk_dollars:,.0f}/trade)")

    # The honest number: one position per symbol at a time.
    if nonoverlap:
        nw = [r for r in nonoverlap if r > 0]
        nl = [r for r in nonoverlap if r <= 0]
        n_pf = (sum(nw) / -sum(nl)) if nl else float("inf")
        n_exp = sum(nonoverlap) / len(nonoverlap)
        print("\n-- Non-overlapping trades (REALISTIC: one position per symbol "
          "at a time) --")
        print(f"Trades:              {len(nonoverlap)}")
        print(f"Win rate:            {len(nw)/len(nonoverlap):6.1%}")
        print(f"Expectancy:          {n_exp:+.3f} R per trade")
        print(f"Profit factor:       {n_pf:.3f}")
        print(f"Sum of R:            {sum(nonoverlap):+.1f}R")
        print(f"Illustrative P&L:    ${sum(nonoverlap)*risk_dollars:,.0f} "
          f"(non-compounding, {args.risk_frac:.1%} risk)")
        # this is the number that decides the verdict below
        expectancy, pf = n_exp, n_pf
        pnl_dollars = sum(nonoverlap) * risk_dollars

    # SPY buy-and-hold benchmark over the same window
    spy = fetch_daily("SPY", start, end)
    if spy:
        spy_start, spy_end = spy[0][4], spy[-1][4]
        spy_ret = spy_end / spy_start - 1.0
        days = (spy[-1][0] - spy[0][0]).days or 1
        spy_annual = (1 + spy_ret) ** (365.0 / days) - 1
        print(f"\nSPY buy-and-hold:    {spy_ret:+.1%} over {days} days "
              f"({spy_annual:+.1%} annualised)")
        strat_ret = pnl_dollars / args.account
        print(f"\nVERDICT (on realistic non-overlapping accounting): expectancy is "
              f"{'POSITIVE' if expectancy > 0 else 'NON-POSITIVE'} "
              f"(PF {pf:.3f}); illustrative return {strat_ret:+.1%} "
              f"{'beats' if strat_ret > spy_ret else 'does NOT beat'} "
              f"SPY buy-and-hold ({spy_ret:+.1%}) over this window.")
        print("NOTE: single ~2-month window, small sample, daily-bar approximation "
              "with pessimistic path assumption. Not out-of-sample validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
