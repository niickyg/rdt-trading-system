#!/usr/bin/env python3
"""Independent first-touch backtest of RDT signal_history.json vs real IBKR prices.

Measures: does the raw RRS momentum signal have edge? For each historical signal we
enter at the signal's stated entry_price and walk subsequent DAILY bars, recording
whether the stop or target is touched first (first-touch, daily high/low). Same-bar
stop+target ambiguity resolves to STOP (conservative). Signals that resolve neither
within MAX_HOLD trading days are marked-to-market at the last close.

This measures SIGNAL EDGE, not a tradeable portfolio (the raw feed has ~90 signals/day,
far more than $25k could hold). Per-trade expectancy in R is the honest edge metric.
"""
import json, os, sys, statistics
from collections import defaultdict, Counter

PRICES = "/tmp/claude-0/-home-user-rdt-trading-system/5d2878f6-f93c-5bd4-8676-deb3f6d17e4e/scratchpad/prices"
SIGNALS = "/home/user/rdt-trading-system/data/signals/signal_history.json"

# cost model (per share, round trip): IBKR commission ~$0.005/sh each way + ~1 tick slippage each way
COMMISSION_RT = 0.01      # $/share round trip
SLIPPAGE_RT   = 0.04      # $/share round trip (2 cents each side)
COST_RT = COMMISSION_RT + SLIPPAGE_RT

def load_prices():
    px = {}
    for fn in os.listdir(PRICES):
        if not fn.endswith(".json") or fn.startswith("_"):
            continue
        sym = fn[:-5]
        d = json.load(open(os.path.join(PRICES, fn)))
        if not d.get("time"):
            continue
        rows = []
        highs = d.get("high") or d["close"]  # SPY file has only time/close (benchmark use)
        lows = d.get("low") or d["close"]
        for i, t in enumerate(d["time"]):
            rows.append((t[:10], highs[i], lows[i], d["close"][i]))
        rows.sort(key=lambda r: r[0])
        px[sym] = rows
    return px

def simulate(signals, px, max_hold, same_bar="stop"):
    results = []
    skipped = Counter()
    for s in signals:
        sym = s["symbol"]; d = s["direction"]
        e = s.get("entry_price"); st = s.get("stop_price"); tg = s.get("target_price")
        gen = (s.get("generated_at") or "")[:10]
        if sym not in px: skipped["no_price_file"] += 1; continue
        if not (e and st and tg and gen): skipped["missing_fields"] += 1; continue
        # risk per share
        if d == "long":
            risk = e - st
            if risk <= 0 or tg <= e: skipped["bad_levels"] += 1; continue
        else:
            risk = st - e
            if risk <= 0 or tg >= e: skipped["bad_levels"] += 1; continue
        rr = (tg - e)/risk if d == "long" else (e - tg)/risk
        # forward bars strictly after gen date
        bars = [b for b in px[sym] if b[0] > gen][:max_hold]
        if not bars: skipped["no_forward_bars"] += 1; continue
        outcome = None; exit_px = None
        for (dt, hi, lo, cl) in bars:
            if d == "long":
                hit_stop = lo <= st; hit_tgt = hi >= tg
            else:
                hit_stop = hi >= st; hit_tgt = lo <= tg
            if hit_stop and hit_tgt:
                outcome = same_bar
            elif hit_stop:
                outcome = "stop"
            elif hit_tgt:
                outcome = "target"
            if outcome:
                exit_px = st if outcome == "stop" else tg
                break
        if outcome is None:
            outcome = "timeout"; exit_px = bars[-1][3]  # last close
        # R multiple (gross)
        if d == "long":
            r_gross = (exit_px - e)/risk
        else:
            r_gross = (e - exit_px)/risk
        cost_R = COST_RT / risk
        r_net = r_gross - cost_R
        results.append({"sym": sym, "dir": d, "outcome": outcome, "rr": rr,
                        "r_gross": r_gross, "r_net": r_net, "risk": risk, "entry": e})
    return results, skipped

def report(results, label):
    n = len(results)
    oc = Counter(r["outcome"] for r in results)
    tgt = oc["target"]; stp = oc["stop"]; to = oc["timeout"]
    resolved = tgt + stp
    wr = tgt/resolved*100 if resolved else float("nan")
    wr_incl_to = tgt/n*100 if n else float("nan")
    exp_g = statistics.mean(r["r_gross"] for r in results) if n else 0
    exp_n = statistics.mean(r["r_net"] for r in results) if n else 0
    pos = sum(r["r_net"] for r in results if r["r_net"] > 0)
    neg = sum(r["r_net"] for r in results if r["r_net"] < 0)
    pf = pos/abs(neg) if neg else float("inf")
    avg_rr = statistics.mean(r["rr"] for r in results) if n else 0
    be_wr = 1/(1+avg_rr)*100  # breakeven win rate for this R:R (gross)
    print(f"\n===== {label} (n={n}) =====")
    print(f"  outcomes: target={tgt} stop={stp} timeout={to}")
    print(f"  win rate (resolved only): {wr:.1f}%   (incl timeouts as non-win: {wr_incl_to:.1f}%)")
    print(f"  avg R:R structure: {avg_rr:.2f}  -> breakeven win rate needed (gross): {be_wr:.1f}%")
    print(f"  expectancy/trade:  gross {exp_g:+.3f}R   net {exp_n:+.3f}R")
    print(f"  profit factor (net): {pf:.3f}")
    # $ at 1% risk of $25k = $250/trade
    print(f"  => at $250 risk/trade: net {exp_n*250:+.2f} $/trade, total over {n} signals {exp_n*250*n:+,.0f} $ (NOT a portfolio, see caveat)")
    return {"n": n, "wr": wr, "exp_net": exp_n, "pf": pf, "avg_rr": avg_rr, "be_wr": be_wr,
            "target": tgt, "stop": stp, "timeout": to, "exp_gross": exp_g}

def alpha_test(signals, px, horizon=10):
    """Directional-alpha test: for each signal, hold the signalled direction for exactly
    `horizon` trading days (entry at first close after gen date, exit at close h days later,
    NO stop/target). Compute stock return and SPY return over the SAME dates; alpha =
    directional (stock - SPY). If RRS is real relative-strength alpha, mean alpha > 0 for
    BOTH longs and shorts. If it's just beta, longs win and shorts lose in a bull market."""
    spy = {d: cl for (d, hi, lo, cl) in px.get("SPY", [])}
    spy_dates = sorted(spy)
    def spy_ret(d0, d1):
        if d0 in spy and d1 in spy:
            return spy[d1]/spy[d0]-1
        return None
    rows = {"long": [], "short": []}
    for s in signals:
        sym = s["symbol"]; d = s["direction"]; gen = (s.get("generated_at") or "")[:10]
        if sym not in px or d not in rows: continue
        bars = [b for b in px[sym] if b[0] > gen]
        if len(bars) <= horizon: continue
        d0 = bars[0][0]; c0 = bars[0][3]
        d1 = bars[horizon][0]; c1 = bars[horizon][3]
        stock_r = c1/c0 - 1
        sr = spy_ret(d0, d1)
        if sr is None: continue
        # directional: for shorts, a profitable move is stock DOWN, so flip sign
        dir_stock = stock_r if d == "long" else -stock_r
        dir_spy = sr if d == "long" else -sr
        alpha = dir_stock - dir_spy            # did the pick beat SPY in the signalled direction?
        rows[d].append({"stock": dir_stock, "spy": dir_spy, "alpha": alpha})
    print(f"\n===== DIRECTIONAL ALPHA TEST (hold {horizon}d, no stop/target) =====")
    for d in ("long", "short"):
        r = rows[d]
        if not r: continue
        ms = statistics.mean(x["stock"] for x in r)
        mspy = statistics.mean(x["spy"] for x in r)
        ma = statistics.mean(x["alpha"] for x in r)
        win_alpha = sum(1 for x in r if x["alpha"] > 0)/len(r)*100
        print(f"  {d.upper()}s (n={len(r)}): mean directional stock move {ms*100:+.2f}%, "
              f"SPY move {mspy*100:+.2f}%, ALPHA {ma*100:+.2f}%  ({win_alpha:.0f}% beat SPY)")
    print("  interpretation: alpha>0 for BOTH sides = real relative-strength edge; "
          "long alpha>0 & short alpha<0 = just market beta.")

def main():
    signals = json.load(open(SIGNALS))
    px = load_prices()
    print(f"loaded {len(signals)} signals, {len(px)} price files: {sorted(px)}")
    summary = {}
    for mh in (5, 10, 20):
        res, sk = simulate(signals, px, mh, same_bar="stop")
        summary[mh] = report(res, f"MAX_HOLD={mh} days, same-bar=STOP (conservative)")
        if mh == 10:
            print(f"  skipped: {dict(sk)}")
            # direction split
            longs = [r for r in res if r["dir"] == "long"]
            shorts = [r for r in res if r["dir"] == "short"]
            report(longs, "  -> LONGS only (MH=10)")
            report(shorts, "  -> SHORTS only (MH=10)")
            # optimistic bound
            res_o, _ = simulate(signals, px, 10, same_bar="target")
            report(res_o, "MAX_HOLD=10, same-bar=TARGET (optimistic bound)")
    alpha_test(signals, px, horizon=10)
    # SPY buy-and-hold benchmark
    spy = px.get("SPY")
    if spy:
        by_date = {d: cl for (d, hi, lo, cl) in spy}
        start = by_date.get("2026-02-03")
        print(f"\n===== SPY BUY & HOLD BENCHMARK =====")
        print(f"  SPY 2026-02-03 close: {start}")
        print(f"  SPY 2026-08-14 close: 775.82 (from full-year fetch)")
        if start:
            print(f"  buy-and-hold Feb 3 -> Aug 14: {(775.82/start-1)*100:+.2f}%  (= {(775.82/start-1)*25000:+,.0f} on $25k)")
    json.dump(summary, open(os.path.join(PRICES, "_summary.json"), "w"), indent=2)

if __name__ == "__main__":
    main()
