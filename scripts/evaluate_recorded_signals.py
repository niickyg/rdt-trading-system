#!/usr/bin/env python3
"""
Out-of-sample forward evaluation of the bot's OWN recorded signals.

Context
-------
`data/signals/signal_history.json` contains 1,986 real signals the live scanner
emitted between 2026-02-03 and 2026-03-05. Because we are now well past that
window, every one of those signals has a *known* real outcome. This script
replays them against real daily OHLC data (fetched separately via the IBKR MCP
connector into a prices JSON file) and measures whether the strategy actually
made money net of honest costs, benchmarked against SPY buy-and-hold over the
same span.

This is deliberately independent of the project's own backtesting engine — it
tests the signals the bot *actually produced*, not a re-simulation of the
strategy logic.

Methodology (documented so it can be criticised)
------------------------------------------------
1. De-duplication. The scanner re-emits the same live signal every scan, so the
   1,986 rows are NOT independent trades (e.g. DOW appears 93 times in a month).
   We collapse them: for each symbol we walk signals in time order and open a
   trade only when flat in that symbol, using the FIRST signal's
   entry/stop/target. All further signals for that symbol are ignored until the
   trade exits. This mirrors a trader holding one position per name at a time.

2. Entry. Fill at the OPEN of the first trading day strictly after the signal's
   calendar date (no look-ahead; the signal is known only after its bar).

3. Exit, checked day by day on daily bars (max hold = MAX_HOLD_DAYS):
     LONG : stop if low  <= stop; target if high >= target.
     SHORT: stop if high >= stop; target if low  <= target.
   If both touched same day -> assume STOP first (conservative).
   If neither by the horizon -> exit at that day's close (time stop).

4. Costs. Round-trip friction of COST_BPS basis points of notional applied to
   every trade. NOTE: with daily bars this UNDERSTATES real slippage because
   gaps through stops are filled at the stop level, not the gap open.

5. Portfolio. Sequential, entry-date ordered, fixed-fractional RISK_PCT of
   current equity risked per trade (1R = |entry-stop|), capped at MAX_CONCURRENT
   open positions. Equity compounds. Benchmark = SPY buy-and-hold from the first
   entry date to the last exit date.

Usage:
    python scripts/evaluate_recorded_signals.py <prices.json> [out.md]
"""
import json
import sys
from datetime import datetime, date
from pathlib import Path

# ---- Tunable, honest assumptions -------------------------------------------
MAX_HOLD_DAYS = 10       # DEPLOYMENT_SUMMARY documents a 5-10 day time stop
COST_BPS = 10.0          # round-trip friction, basis points of notional
RISK_PCT = 0.01          # 1% of equity risked per trade
MAX_CONCURRENT = 8       # MAX_OPEN_POSITIONS default
START_EQUITY = 25000.0


def d(s):
    return datetime.strptime(s[:10], "%Y-%m-%d").date()


def load_prices(path):
    raw = json.load(open(path))
    out = {}
    for sym, obj in raw.items():
        bars = obj["bars"] if isinstance(obj, dict) and "bars" in obj else obj
        # index by date -> ohlc, and keep an ordered date list
        by_date = {}
        for b in bars:
            by_date[d(b["date"])] = (b["open"], b["high"], b["low"], b["close"])
        out[sym] = {"by_date": by_date, "dates": sorted(by_date)}
    return out


def next_trading_day(dates, after):
    for dt in dates:
        if dt > after:
            return dt
    return None


def simulate_trade(px, direction, entry_price, stop, target, signal_date):
    """Return dict with entry/exit/outcome/R or None if not simulable."""
    dates = px["dates"]
    by = px["by_date"]
    entry_dt = next_trading_day(dates, signal_date)
    if entry_dt is None:
        return None
    o, h, l, c = by[entry_dt]
    fill = o  # enter at next open
    if fill <= 0:
        return None
    risk_per_share = abs(fill - stop)
    if risk_per_share <= 0:
        return None
    # forward walk
    idx = dates.index(entry_dt)
    horizon = dates[idx: idx + MAX_HOLD_DAYS + 1]
    exit_price = None
    outcome = None
    exit_dt = horizon[-1]
    for dt in horizon:
        oo, hh, ll, cc = by[dt]
        if direction == "long":
            hit_stop = ll <= stop
            hit_tgt = hh >= target
        else:
            hit_stop = hh >= stop
            hit_tgt = ll <= target
        if hit_stop:  # conservative: stop wins ties
            exit_price, outcome, exit_dt = stop, "stop", dt
            break
        if hit_tgt:
            exit_price, outcome, exit_dt = target, "target", dt
            break
    if exit_price is None:  # time stop at last bar close
        exit_price, outcome, exit_dt = by[horizon[-1]][3], "time", horizon[-1]

    if direction == "long":
        gross_r = (exit_price - fill) / risk_per_share
        gross_ret = (exit_price - fill) / fill
    else:
        gross_r = (fill - exit_price) / risk_per_share
        gross_ret = (fill - exit_price) / fill
    cost_ret = COST_BPS / 10000.0
    net_ret = gross_ret - cost_ret
    net_r = gross_r - (cost_ret * fill / risk_per_share)
    return {
        "direction": direction, "entry_dt": entry_dt, "exit_dt": exit_dt,
        "fill": fill, "stop": stop, "target": target, "exit_price": exit_price,
        "outcome": outcome, "gross_r": gross_r, "net_r": net_r,
        "net_ret": net_ret, "risk_per_share": risk_per_share,
    }


def build_trades(signals, prices):
    # group signals by symbol, sorted by time
    by_sym = {}
    for s in signals:
        by_sym.setdefault(s["symbol"], []).append(s)
    trades = []
    skipped_no_px = set()
    for sym, sigs in by_sym.items():
        if sym not in prices:
            skipped_no_px.add(sym)
            continue
        sigs.sort(key=lambda s: s["generated_at"])
        px = prices[sym]
        busy_until = date.min
        for s in sigs:
            sdate = d(s["generated_at"])
            if sdate <= busy_until:
                continue
            t = simulate_trade(px, s["direction"], float(s["entry_price"]),
                               float(s["stop_price"]), float(s["target_price"]), sdate)
            if t is None:
                continue
            t["symbol"] = sym
            t["rrs"] = s.get("rrs")
            trades.append(t)
            busy_until = t["exit_dt"]
    trades.sort(key=lambda t: t["entry_dt"])
    return trades, skipped_no_px


def portfolio_sim(trades, order_key=None):
    """Sequential portfolio. order_key controls how competing signals are ranked
    (default = entry date only, i.e. arbitrary arrival order for same-day signals).
    order_key must use only information known at signal time (no look-ahead)."""
    if order_key is None:
        order_key = lambda t: (t["entry_dt"],)
    trades = sorted(trades, key=order_key)
    equity = START_EQUITY
    open_positions = []  # list of (exit_dt, pnl)
    curve = []
    for t in trades:
        # close any positions that exited on/before this entry date
        still = []
        for exit_dt, pnl in open_positions:
            if exit_dt <= t["entry_dt"]:
                equity += pnl
            else:
                still.append((exit_dt, pnl))
        open_positions = still
        if len(open_positions) >= MAX_CONCURRENT:
            t["taken"] = False
            continue
        t["taken"] = True
        risk_dollars = equity * RISK_PCT
        pnl = risk_dollars * t["net_r"]
        open_positions.append((t["exit_dt"], pnl))
        curve.append((t["entry_dt"], equity))
    for exit_dt, pnl in open_positions:
        equity += pnl
    return equity, curve


def spy_buy_hold(prices, start, end):
    spy = prices.get("SPY")
    if not spy:
        return None
    dates = spy["dates"]
    s = next((dt for dt in dates if dt >= start), None)
    e = next((dt for dt in reversed(dates) if dt <= end), None)
    if not s or not e:
        return None
    o = spy["by_date"][s][0]
    c = spy["by_date"][e][3]
    return (c - o) / o, s, e


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    prices = load_prices(sys.argv[1])
    root = Path(__file__).resolve().parent.parent
    signals = json.load(open(root / "data/signals/signal_history.json"))

    trades, skipped = build_trades(signals, prices)
    if not trades:
        print("No trades simulable.")
        sys.exit(1)

    wins = [t for t in trades if t["net_r"] > 0]
    losses = [t for t in trades if t["net_r"] <= 0]
    gross_win = sum(t["net_r"] for t in wins)
    gross_loss = -sum(t["net_r"] for t in losses)
    n = len(trades)
    win_rate = len(wins) / n
    expectancy_r = sum(t["net_r"] for t in trades) / n
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    final_eq, curve = portfolio_sim(trades)
    taken = [t for t in trades if t.get("taken")]
    port_ret = (final_eq - START_EQUITY) / START_EQUITY

    first_entry = min(t["entry_dt"] for t in trades)
    last_exit = max(t["exit_dt"] for t in trades)
    bh = spy_buy_hold(prices, first_entry, last_exit)

    long_t = [t for t in trades if t["direction"] == "long"]
    short_t = [t for t in trades if t["direction"] == "short"]
    outcomes = {}
    for t in trades:
        outcomes[t["outcome"]] = outcomes.get(t["outcome"], 0) + 1

    L = []
    L.append("# Out-of-Sample Evaluation of Recorded Signals")
    L.append("")
    L.append(f"Signals file window: {d(signals[0]['generated_at'])} .. {d(signals[-1]['generated_at'])}")
    L.append(f"Raw signals: {len(signals)}  |  De-duplicated trades: {n}  "
             f"(one position per symbol at a time)")
    if skipped:
        L.append(f"Symbols skipped (no price data): {sorted(skipped)}")
    L.append("")
    L.append("## Assumptions")
    L.append(f"- Entry: next-day open after signal | Max hold: {MAX_HOLD_DAYS} days | "
             f"Cost: {COST_BPS} bps round-trip | Ties -> stop")
    L.append(f"- Portfolio: {RISK_PCT:.1%} risk/trade, max {MAX_CONCURRENT} concurrent, "
             f"${START_EQUITY:,.0f} start")
    L.append("")
    L.append("## Trade statistics (per-trade, equal weight in R)")
    L.append(f"- Trades: {n}  ({len(long_t)} long / {len(short_t)} short)")
    L.append(f"- Win rate: {win_rate:.1%}")
    L.append(f"- Expectancy: {expectancy_r:+.3f} R per trade")
    L.append(f"- Profit factor: {pf:.2f}")
    L.append(f"- Avg win: {(gross_win/len(wins) if wins else 0):+.2f} R | "
             f"Avg loss: {(-gross_loss/len(losses) if losses else 0):+.2f} R")
    L.append(f"- Outcomes: {outcomes}")
    if long_t:
        L.append(f"- LONG expectancy: {sum(t['net_r'] for t in long_t)/len(long_t):+.3f} R "
                 f"(n={len(long_t)}, win {sum(1 for t in long_t if t['net_r']>0)/len(long_t):.1%})")
    if short_t:
        L.append(f"- SHORT expectancy: {sum(t['net_r'] for t in short_t)/len(short_t):+.3f} R "
                 f"(n={len(short_t)}, win {sum(1 for t in short_t if t['net_r']>0)/len(short_t):.1%})")
    L.append("")
    L.append("## Portfolio simulation vs SPY buy-and-hold")
    L.append(f"- Trades actually taken (concurrency cap): {len(taken)}/{n}")
    L.append(f"- Final equity: ${final_eq:,.0f}  ({port_ret:+.2%})")
    if bh:
        bh_ret, bs, be = bh
        L.append(f"- SPY buy-and-hold {bs}..{be}: {bh_ret:+.2%}")
        L.append(f"- **Strategy minus SPY: {port_ret - bh_ret:+.2%}**")
    L.append("")
    # ---- Selection-policy comparison (competing signals at the position cap) ----
    # The live bot enforces the position cap first-come-first-served (arrival order),
    # with NO cross-signal ranking. These policies use only signal-time info.
    by_entry = lambda t: (t["entry_dt"],)
    by_rrs = lambda t: (t["entry_dt"], -abs(t.get("rrs") or 0.0))
    longs = [t for t in trades if t["direction"] == "long"]
    pol = [
        ("Arbitrary arrival order (current bot behavior)", trades, by_entry),
        ("RRS-priority (strongest RRS gets the slot)", trades, by_rrs),
        ("RRS-priority + long-only", longs, by_rrs),
    ]
    L.append("## Selection policy at the position cap (why arrival order matters)")
    L.append(f"On {max((sum(1 for t in trades if t['entry_dt']==e) for e in set(t['entry_dt'] for t in trades)))} "
             f"same-day signals vs a {MAX_CONCURRENT}-slot cap, *which* signals get taken dominates the result:")
    for name, ts, key in pol:
        eq, _ = portfolio_sim([dict(t) for t in ts], order_key=key)
        L.append(f"- {name}: {(eq-START_EQUITY)/START_EQUITY:+.2%}")
    if bh:
        L.append(f"- (SPY buy-and-hold same window: {bh[0]:+.2%})")
    L.append("")
    verdict = "BEATS" if (bh and port_ret > bh[0]) else "DOES NOT BEAT"
    L.append(f"## Verdict: strategy {verdict} SPY buy-and-hold over this window "
             f"(net of {COST_BPS} bps costs).")
    report = "\n".join(L)
    print(report)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(report + "\n")
        # also dump per-trade detail as JSON next to it
        detail = [
            {k: (v.isoformat() if isinstance(v, date) else v) for k, v in t.items()}
            for t in trades
        ]
        Path(sys.argv[2]).with_suffix(".trades.json").write_text(json.dumps(detail, indent=2))


if __name__ == "__main__":
    main()
