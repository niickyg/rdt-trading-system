#!/usr/bin/env python3
"""
operator_backtest.py — Honest, reusable evaluation harness for the RDT bot's own signals.

WHY THIS EXISTS
---------------
As of 2026-08-11 the system had persisted ~1,986 raw scanner signals in
data/signals/signal_history.json but almost NO outcome tracking (2 outcomes total
in signal_metrics.json). You cannot make a bot profitable if you cannot measure
whether its signals actually make money. This script closes that gap: it takes the
bot's own historical signals, attaches REAL forward prices, and produces an honest
scorecard against SPY buy-and-hold — net of commissions and slippage.

It deliberately reports the numbers that flatter a strategy AND the numbers that
kill it, side by side:
  * per-trade barrier-touch P&L (stop/target) at several max-hold horizons
  * per-trade raw directional edge (hold N days, exit at close)
  * ALPHA vs BETA: signal return minus SPY return over identical holding windows
  * a CAPACITY-CONSTRAINED portfolio sim (max N concurrent, no leverage, compounded)
    — this is the number that actually matters for a real account
  * SPY buy-and-hold benchmark over the same window

DATA REALITY (important, learned 2026-08-11)
--------------------------------------------
The bot operates in a 2026 market timeline. In the user's live environment the
correct forward prices come from IBKR (via brokers/ibkr/client.py). Public yfinance
data reachable from some sandboxes can be ~1 year behind and WILL NOT match 2026
signals — do not trust a run whose price dates don't overlap the signal dates.

To stay provider-agnostic, this script reads an OHLC price file you supply via
--prices (JSON: {SYMBOL: [{"d":"YYYY-MM-DD","o":..,"h":..,"l":..,"c":..}, ...]}).
Produce that file from whatever source is authoritative in your environment (IBKR
history, a data vendor, or yfinance if your clock matches). A CONTAMINATION GUARD
drops any symbol whose signal entry price is wildly off-scale versus its own price
series in the signal month (catches split/spinoff/wrong-instrument breaks — e.g.
the DuPont "DD" 3x scale mismatch found on 2026-08-11).

USAGE
-----
    python scripts/operator_backtest.py \
        --signals data/signals/signal_history.json \
        --prices  path/to/ohlc_prices.json \
        --account 25000 --risk-pct 0.015 --max-pos 8

Exit code is 0 always; this is a reporting tool, not a gate.
"""
from __future__ import annotations
import argparse, json, datetime as dt, statistics, collections, sys


def load_prices(path):
    raw = json.load(open(path))
    P = {}
    for s, bars in raw.items():
        clean = []
        for b in bars:
            try:
                clean.append({"d": dt.date.fromisoformat(str(b["d"])[:10]),
                              "o": float(b["o"]), "h": float(b["h"]),
                              "l": float(b["l"]), "c": float(b["c"])})
            except (KeyError, ValueError, TypeError):
                continue
        P[s] = sorted(clean, key=lambda x: x["d"])
    return P


def dedupe_signals(signals):
    """One trade per (symbol, date, direction), earliest generated_at wins."""
    best = {}
    for s in signals:
        g = s.get("generated_at", "")
        key = (s.get("symbol"), g[:10], s.get("direction"))
        if key not in best or g < best[key].get("generated_at", ""):
            best[key] = s
    return [t for t in best.values()
            if t.get("entry_price") and t.get("stop_price") and t.get("target_price")]


def contamination_guard(trades, P, max_ratio=1.5):
    """Drop symbols whose signal entry price is off-scale vs their own series in the
    signal month — a robust flag for split/spinoff/wrong-instrument data breaks."""
    bad = set()
    for t in trades:
        s = t["symbol"]
        gen = dt.date.fromisoformat(t["generated_at"][:10])
        month = [b["c"] for b in P.get(s, []) if b["d"].year == gen.year and b["d"].month == gen.month]
        if month:
            med = statistics.median(month)
            r = t["entry_price"] / med if med else 1.0
            if r > max_ratio or r < 1.0 / max_ratio:
                bad.add(s)
    if bad:
        print(f"  [contamination guard] dropping {sorted(bad)} (entry/data scale mismatch)")
    return [t for t in trades if t["symbol"] not in bad], bad


def fwd_bars(P, sym, gen):
    return [b for b in P.get(sym, []) if b["d"] > gen]


def barrier_test(trades, P, account, risk_pct, max_hold, comm=1.0, slip=0.0005):
    risk = account * risk_pct
    res = []
    for t in trades:
        s, dirn = t["symbol"], t["direction"]
        sign = 1 if dirn == "long" else -1
        E, S, T = t["entry_price"], t["stop_price"], t["target_price"]
        gen = dt.date.fromisoformat(t["generated_at"][:10])
        fwd = fwd_bars(P, s, gen)
        if not fwd:
            continue
        rps = abs(E - S)
        if rps <= 0:
            continue
        sh = risk / rps
        exitp = oc = None
        for b in fwd[:max_hold]:
            hs = (b["l"] <= S) if dirn == "long" else (b["h"] >= S)
            ht = (b["h"] >= T) if dirn == "long" else (b["l"] <= T)
            if hs:            # ambiguous same-bar -> stop first (conservative)
                exitp, oc = S, "stop"; break
            if ht:
                exitp, oc = T, "target"; break
        if exitp is None:
            exitp, oc = fwd[min(max_hold, len(fwd)) - 1]["c"], "timeout"
        gross = sign * (exitp - E) * sh
        net = gross - (2 * comm + slip * (E + exitp) * sh)
        res.append({"net": net, "gross": gross, "oc": oc, "dir": dirn})
    return res


def summarize(res, account, risk, label):
    n = len(res)
    if not n:
        print(f"  {label}: NO TRADES"); return
    wins = [r for r in res if r["net"] > 0]
    losses = [r for r in res if r["net"] <= 0]
    net = sum(r["net"] for r in res)
    gl = -sum(r["net"] for r in losses)
    pf = (sum(r["net"] for r in wins) / gl) if gl > 0 else float("inf")
    oc = dict(collections.Counter(r["oc"] for r in res))
    print(f"  {label}: n={n} win={len(wins)/n*100:.0f}% PF={pf:.2f} "
          f"NET=${net:,.0f} ({net/account*100:+.1f}%) exp={net/n/risk:+.2f}R {oc}")


def alpha_vs_beta(trades, P, hold=5):
    spy = P.get("SPY", [])

    def spy_ret(gen):
        fs = [b for b in spy if b["d"] > gen]
        if not fs:
            return None
        return (fs[min(hold, len(fs)) - 1]["c"] - fs[0]["o"]) / fs[0]["o"]

    out = {}
    for direction in ("long", "short"):
        raws, betas, diffs = [], [], []
        for t in trades:
            if t["direction"] != direction:
                continue
            s = t["symbol"]; sign = 1 if direction == "long" else -1
            gen = dt.date.fromisoformat(t["generated_at"][:10])
            fwd = fwd_bars(P, s, gen)
            sret = spy_ret(gen)
            if not fwd or sret is None:
                continue
            r = sign * (fwd[min(hold, len(fwd)) - 1]["c"] - t["entry_price"]) / t["entry_price"]
            raws.append(r); betas.append(sret * sign); diffs.append(r - sret * sign)
        if diffs:
            out[direction] = (len(diffs), statistics.mean(raws), statistics.mean(betas),
                              statistics.mean(diffs), sum(1 for d in diffs if d > 0) / len(diffs))
    return out


def portfolio_sim(trades, P, account, risk_pct, max_pos, max_hold,
                  exit_mode="barrier", long_only=False, comm=1.0, slip=0.0005):
    if long_only:
        trades = [t for t in trades if t["direction"] == "long"]
    evs = []
    for t in trades:
        gen = dt.date.fromisoformat(t["generated_at"][:10])
        fwd = fwd_bars(P, t["symbol"], gen)
        if fwd:
            evs.append({**t, "entry_date": fwd[0]["d"], "entry_open": fwd[0]["o"], "bars": fwd})
    if not evs:
        return None
    days = sorted({b["d"] for s in P for b in P[s] if b["d"] >= min(e["entry_date"] for e in evs)})
    equity = float(account); open_pos = []; closed = []
    pending = sorted(evs, key=lambda e: (e["entry_date"], -abs(e.get("rrs", 0)))); pi = 0
    for day in days:
        still = []
        for pos in open_pos:
            bar = next((b for b in pos["bars"] if b["d"] == day), None)
            if bar:
                pos["held"] += 1
            exitp = None; dirn = pos["direction"]
            if bar:
                hs = (bar["l"] <= pos["stop"]) if dirn == "long" else (bar["h"] >= pos["stop"])
                ht = (bar["h"] >= pos["target"]) if dirn == "long" else (bar["l"] <= pos["target"])
                if hs:
                    exitp = pos["stop"]
                elif exit_mode == "barrier" and ht:
                    exitp = pos["target"]
                elif pos["held"] >= max_hold:
                    exitp = bar["c"]
            if exitp is not None:
                sign = 1 if dirn == "long" else -1
                pnl = sign * (exitp - pos["entry"]) * pos["shares"] \
                    - (2 * comm + slip * (pos["entry"] + exitp) * pos["shares"])
                equity += pnl; closed.append(pnl)
            else:
                still.append(pos)
        open_pos = still
        while pi < len(pending) and pending[pi]["entry_date"] == day:
            e = pending[pi]; pi += 1
            if len(open_pos) >= max_pos:
                continue
            E = e["entry_open"] if e["entry_open"] > 0 else e["entry_price"]
            S = e["stop_price"]; rps = abs(E - S)
            if rps <= 0:
                continue
            shares = min(equity * risk_pct / rps, (equity / max_pos) / E)  # no leverage
            open_pos.append({**e, "entry": E, "stop": S, "target": e["target_price"],
                             "shares": shares, "held": 0})
    for pos in open_pos:
        lb = pos["bars"][-1]; sign = 1 if pos["direction"] == "long" else -1
        pnl = sign * (lb["c"] - pos["entry"]) * pos["shares"] \
            - (2 * comm + slip * (pos["entry"] + lb["c"]) * pos["shares"])
        equity += pnl; closed.append(pnl)
    wr = (sum(1 for p in closed if p > 0) / len(closed) * 100) if closed else 0
    return {"equity": equity, "ret": (equity - account) / account,
            "trades": len(closed), "win": wr}


def spy_buyhold(trades, P):
    spy = P.get("SPY", [])
    if not spy or not trades:
        return None
    fg = min(dt.date.fromisoformat(t["generated_at"][:10]) for t in trades)
    sf = [b for b in spy if b["d"] >= fg]
    if not sf:
        return None
    ret = sf[-1]["c"] / sf[0]["c"] - 1
    return {"start": sf[0]["d"], "end": sf[-1]["d"], "ret": ret}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signals", default="data/signals/signal_history.json")
    ap.add_argument("--prices", required=True,
                    help="JSON {SYMBOL:[{d,o,h,l,c}]} of REAL forward OHLC (IBKR/vendor).")
    ap.add_argument("--account", type=float, default=25000.0)
    ap.add_argument("--risk-pct", type=float, default=0.015)
    ap.add_argument("--max-pos", type=int, default=8)
    args = ap.parse_args()

    signals = json.load(open(args.signals))
    P = load_prices(args.prices)
    trades = dedupe_signals(signals)
    print(f"Signals: {len(signals)} raw -> {len(trades)} distinct (symbol/date/dir)")
    trades, _ = contamination_guard(trades, P)
    have = sum(1 for t in trades if fwd_bars(P, t["symbol"], dt.date.fromisoformat(t["generated_at"][:10])))
    print(f"Trades with forward price data: {have}/{len(trades)}")
    if have == 0:
        print("!! No overlap between signal dates and price dates. Wrong price file / timeline. Aborting.")
        return 0

    risk = args.account * args.risk_pct
    print("\n== BARRIER-TOUCH (stop/target, net of costs) ==")
    for mh in (5, 10, 20):
        summarize(barrier_test(trades, P, args.account, args.risk_pct, mh), args.account, risk, f"max_hold={mh}d")

    print("\n== ALPHA vs BETA (5-day, signal return minus SPY over same window) ==")
    for d, (n, sig, beta, alpha, aw) in alpha_vs_beta(trades, P).items():
        print(f"  {d:5}: n={n} signal={sig*100:+.2f}% beta={beta*100:+.2f}% "
              f"ALPHA={alpha*100:+.2f}% alpha_win={aw*100:.0f}%")

    bh = spy_buyhold(trades, P)
    print("\n== REALISTIC PORTFOLIO (max concurrent, no leverage, compounded) vs SPY B&H ==")
    if bh:
        print(f"  SPY buy&hold {bh['start']}->{bh['end']}: {bh['ret']*100:+.2f}%")
    for label, lo, em, mh in (
        ("all signals, barrier, 10d", False, "barrier", 10),
        ("long-only, barrier, 10d",   True,  "barrier", 10),
        ("long-only, 5d time-exit",   True,  "time",    5),
    ):
        r = portfolio_sim(trades, P, args.account, args.risk_pct, args.max_pos, mh,
                          exit_mode=em, long_only=lo)
        if r:
            verdict = "BEATS SPY" if (bh and r["ret"] > bh["ret"]) else "lags SPY"
            print(f"  {label:28}: {r['ret']*100:+5.1f}% ({r['trades']} trades, {r['win']:.0f}% win) -> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
