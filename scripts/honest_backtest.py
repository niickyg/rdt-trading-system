#!/usr/bin/env python3
"""
Honest daily backtest for the RRS strategy.

The committed backtest (backtesting/engine_enhanced.py) manufactures an edge via
three well-known artifacts, verified in code:
  1. SAME-BAR LOOK-AHEAD: signals are computed from the daily close, then the
     position is entered AT that same close (engine_enhanced.py:462,494). You can
     only know the close after the bar is over — this is not tradeable.
  2. IDEALIZED EXIT FILLS: stops fill exactly at stop_price, targets exactly at
     target_price (engine_enhanced.py:313,374) — no gap-through slippage.
  3. ZERO COSTS: no commission, spread, or slippage anywhere in backtesting/.

This harness runs the SAME core RRS signal but lets you toggle honesty, so the
manufactured edge is isolated as the delta between two modes on identical signals:

  --mode optimistic : reproduce their assumptions (enter at signal-bar close,
                      perfect stop/target fills, zero cost).
  --mode honest     : enter at the NEXT bar's open, cross the spread, pay
                      commission + slippage, and let stops/targets gap through.

It also prints SPY buy-and-hold over the identical window as the benchmark that
actually matters (the operator mandate's definition of "profitable").

Data: uses scripts/_yahoo_fetch.fetch_daily (proxy-aware, cached). If the market
data source is unreachable (e.g. Yahoo rate-limits the datacenter IP), the run
fails loudly with instructions rather than fabricating numbers.

Quick mechanics check with NO network:
    python scripts/honest_backtest.py --selftest

Real run (needs reachable data):
    python scripts/honest_backtest.py --range 2y --rrs 2.0
"""
from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass, field

# Same fixed universe the committed walk-forward uses, so an honest run is
# directly comparable to the quoted numbers (same survivor-biased universe,
# but honest fills + costs). Universe bias is a SEPARATE problem, noted below.
UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD", "NFLX",
    "CRM", "AVGO", "ADBE", "COST", "PEP", "LIN", "TMO", "ACN", "MCD", "ABT",
    "QCOM", "TXN", "INTC", "AMAT", "MU", "NOW", "INTU", "ISRG", "BKNG", "PANW",
]

ATR_PERIOD = 14


# --------------------------------------------------------------------------- #
# Indicators (pure python; mirrors shared/indicators/rrs.py semantics)
# --------------------------------------------------------------------------- #
def atr(bars: list[dict], period: int = ATR_PERIOD) -> list[float | None]:
    trs: list[float] = []
    out: list[float | None] = []
    for i, b in enumerate(bars):
        if i == 0:
            tr = b["high"] - b["low"]
        else:
            pc = bars[i - 1]["close"]
            tr = max(b["high"] - b["low"], abs(b["high"] - pc), abs(b["low"] - pc))
        trs.append(tr)
        out.append(statistics.fmean(trs[-period:]) if i + 1 >= period else None)
    return out


def ema(vals: list[float], span: int) -> list[float]:
    k = 2 / (span + 1)
    out: list[float] = []
    prev = vals[0]
    for v in vals:
        prev = v * k + prev * (1 - k)
        out.append(prev)
    return out


def daily_strength_score(bars: list[dict], i: int) -> int:
    """Relaxed strength score (0-5), mirroring check_daily_strength_relaxed."""
    closes = [b["close"] for b in bars[: i + 1]]
    e3, e8, e21 = ema(closes, 3), ema(closes, 8), ema(closes, 21)
    last3 = bars[i - 2 : i + 1]
    two_plus_green = sum(b["close"] > b["open"] for b in last3) >= 2
    last5_lows = [b["low"] for b in bars[max(0, i - 4) : i + 1]]
    higher_lows = len(last5_lows) >= 3 and last5_lows[-1] > last5_lows[0]
    return sum([
        e3[i] > e8[i], e8[i] > e21[i], bars[i]["close"] > e8[i],
        higher_lows, two_plus_green,
    ])


def daily_weakness_score(bars: list[dict], i: int) -> int:
    closes = [b["close"] for b in bars[: i + 1]]
    e3, e8, e21 = ema(closes, 3), ema(closes, 8), ema(closes, 21)
    last3 = bars[i - 2 : i + 1]
    two_plus_red = sum(b["close"] < b["open"] for b in last3) >= 2
    last5_highs = [b["high"] for b in bars[max(0, i - 4) : i + 1]]
    lower_highs = len(last5_highs) >= 3 and last5_highs[-1] < last5_highs[0]
    return sum([
        e8[i] > e3[i], e21[i] > e8[i], bars[i]["close"] < e8[i],
        lower_highs, two_plus_red,
    ])


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
@dataclass
class CostModel:
    commission_per_share: float = 0.005
    commission_min: float = 1.0
    spread_bps_each_side: float = 2.0      # cross half-spread each side (bps of notional)
    slippage_bps: float = 10.0             # adverse fill on stop-outs / market entries

    def entry_cost(self, price: float, shares: int) -> float:
        comm = max(self.commission_min, shares * self.commission_per_share)
        spread = price * shares * self.spread_bps_each_side / 1e4
        return comm + spread

    def exit_cost(self, price: float, shares: int, gapped: bool) -> float:
        comm = max(self.commission_min, shares * self.commission_per_share)
        spread = price * shares * self.spread_bps_each_side / 1e4
        slip = price * shares * self.slippage_bps / 1e4 if gapped else 0.0
        return comm + spread + slip


@dataclass
class Position:
    symbol: str
    direction: str
    shares: int
    entry: float
    stop: float
    target: float
    entry_cost: float


@dataclass
class Result:
    mode: str
    start_equity: float
    end_equity: float
    trades: int
    wins: int
    gross_pnl: float
    costs: float
    equity_curve: list[float] = field(default_factory=list)

    @property
    def net_pnl(self) -> float:
        return self.end_equity - self.start_equity

    @property
    def ret_pct(self) -> float:
        return self.net_pnl / self.start_equity * 100

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades * 100 if self.trades else 0.0


def run_backtest(data: dict[str, list[dict]], spy: list[dict], *, mode: str,
                 rrs_threshold: float, capital: float, risk_pct: float,
                 max_positions: int, stop_atr: float, target_atr: float,
                 costs: CostModel) -> Result:
    honest = mode == "honest"
    # Index everything by common trading dates present in SPY.
    dates = [b["date"] for b in spy]
    spy_by_date = {b["date"]: b for b in spy}
    by_symbol = {s: {b["date"]: b for b in bars} for s, bars in data.items()}
    # Precompute ATR + rrs per symbol per date.

    cap = capital
    equity_curve = []
    positions: dict[str, Position] = {}
    trades = wins = 0
    gross = total_costs = 0.0

    # Precompute indicator series per symbol.
    series = {}
    for s, bars in data.items():
        series[s] = {"bars": bars, "atr": atr(bars),
                     "idx": {b["date"]: i for i, b in enumerate(bars)}}

    for di, d in enumerate(dates):
        if di < ATR_PERIOD + 22 or di + 1 >= len(dates):
            equity_curve.append(cap + _open_value(positions, spy_by_date, by_symbol, d))
            continue
        nd = dates[di + 1]  # next date (for honest entry at next open)
        spy_today = spy_by_date[d]
        spy_prev = spy_by_date[dates[di - 1]]
        spy_pc = (spy_today["close"] / spy_prev["close"] - 1) * 100

        # ---- manage open positions on this bar (exits) ----
        for s in list(positions.keys()):
            pos = positions[s]
            bar = by_symbol[s].get(d)
            if not bar:
                continue
            exit_price = None
            gapped = False
            if pos.direction == "long":
                if bar["low"] <= pos.stop:
                    # honest: fill at min(open, stop) -> gap-through; optimistic: exact stop
                    exit_price = min(bar["open"], pos.stop) if honest else pos.stop
                    gapped = honest and bar["open"] < pos.stop
                elif bar["high"] >= pos.target:
                    exit_price = max(bar["open"], pos.target) if honest else pos.target
            else:  # short
                if bar["high"] >= pos.stop:
                    exit_price = max(bar["open"], pos.stop) if honest else pos.stop
                    gapped = honest and bar["open"] > pos.stop
                elif bar["low"] <= pos.target:
                    exit_price = min(bar["open"], pos.target) if honest else pos.target
            if exit_price is not None:
                pnl = ((exit_price - pos.entry) if pos.direction == "long"
                       else (pos.entry - exit_price)) * pos.shares
                xc = costs.exit_cost(exit_price, pos.shares, gapped) if honest else 0.0
                cap += pos.entry * pos.shares + pnl - xc
                gross += pnl
                total_costs += xc + (pos.entry_cost if honest else 0.0)
                trades += 1
                if pnl - xc - (pos.entry_cost if honest else 0.0) > 0:
                    wins += 1
                del positions[s]

        # ---- scan for entries ----
        if len(positions) < max_positions:
            for s, sd in series.items():
                if s in positions or len(positions) >= max_positions:
                    continue
                i = sd["idx"].get(d)
                if i is None or i < ATR_PERIOD + 22 or sd["atr"][i] is None:
                    continue
                bars = sd["bars"]
                a = sd["atr"][i]
                close = bars[i]["close"]
                prev_close = bars[i - 1]["close"]
                stock_pc = (close / prev_close - 1) * 100
                atr_pct = a / close * 100
                if atr_pct <= 0:
                    continue
                rrs = (stock_pc - spy_pc) / atr_pct
                if abs(rrs) < rrs_threshold:
                    continue
                direction = None
                if rrs > rrs_threshold and daily_strength_score(bars, i) >= 3:
                    direction = "long"
                elif rrs < -rrs_threshold and daily_weakness_score(bars, i) >= 3:
                    direction = "short"
                if not direction:
                    continue

                # Entry price: honest -> NEXT bar open; optimistic -> this close.
                if honest:
                    nb = by_symbol[s].get(nd)
                    if not nb:
                        continue
                    entry = nb["open"]
                else:
                    entry = close

                stop_dist = a * stop_atr
                if stop_dist <= 0:
                    continue
                shares = int((cap * risk_pct) / stop_dist)
                if shares <= 0 or shares * entry > cap:
                    continue
                if direction == "long":
                    stop = entry - stop_dist
                    target = entry + a * target_atr
                else:
                    stop = entry + stop_dist
                    target = entry - a * target_atr
                ec = costs.entry_cost(entry, shares) if honest else 0.0
                cap -= entry * shares + ec
                positions[s] = Position(s, direction, shares, entry, stop, target, ec)

        equity_curve.append(cap + _open_value(positions, spy_by_date, by_symbol, d))

    # liquidate remaining at last close
    last = dates[-1]
    for s, pos in positions.items():
        bar = by_symbol[s].get(last)
        if not bar:
            continue
        px = bar["close"]
        pnl = ((px - pos.entry) if pos.direction == "long"
               else (pos.entry - px)) * pos.shares
        xc = costs.exit_cost(px, pos.shares, False) if honest else 0.0
        cap += pos.entry * pos.shares + pnl - xc
        gross += pnl
        total_costs += xc + (pos.entry_cost if honest else 0.0)
        trades += 1
        if pnl - xc > 0:
            wins += 1

    return Result(mode, capital, cap, trades, wins, gross, total_costs, equity_curve)


def _open_value(positions, spy_by_date, by_symbol, d) -> float:
    """Mark open positions to current close for the equity curve."""
    v = 0.0
    for s, pos in positions.items():
        bar = by_symbol[s].get(d)
        px = bar["close"] if bar else pos.entry
        if pos.direction == "long":
            v += pos.shares * px
        else:  # short: value = entry notional + unrealized
            v += pos.shares * pos.entry + (pos.entry - px) * pos.shares
    return v


def spy_buy_hold(spy: list[dict], capital: float) -> tuple[float, float]:
    first, last = spy[0]["close"], spy[-1]["close"]
    end = capital * last / first
    return end, (end - capital) / capital * 100


# --------------------------------------------------------------------------- #
# Self-test (no network): proves fill/cost mechanics on crafted data.
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    # Build a stock that on day D jumps +5% while SPY is flat (strong RRS),
    # then the next day gaps DOWN through the stop at the open. The honest run
    # must fill worse than the stop; the optimistic run fills exactly at stop.
    n = 60
    spy = [{"date": f"2025-01-{i+1:02d}"[:10].replace("2025-01-00", "2025-01-01"),
            "open": 100, "high": 100.5, "low": 99.5, "close": 100, "volume": 1e6}
           for i in range(n)]
    # give SPY real distinct dates
    for i in range(n):
        spy[i]["date"] = f"D{i:03d}"
    stock = []
    price = 50.0
    for i in range(n):
        o = price
        c = price * (1.001)  # gentle uptrend to satisfy EMA/strength
        stock.append({"date": f"D{i:03d}", "open": o, "high": max(o, c) * 1.005,
                      "low": min(o, c) * 0.995, "close": c, "volume": 1e6})
        price = c
    # Force a strong-RRS long trigger on day 45 (+5% pop).
    stock[45]["close"] = stock[45]["open"] * 1.05
    stock[45]["high"] = stock[45]["close"] * 1.001
    # Next bar gaps down hard through any stop.
    stock[46]["open"] = stock[45]["close"] * 0.90
    stock[46]["high"] = stock[46]["open"] * 1.001
    stock[46]["low"] = stock[46]["open"] * 0.98
    stock[46]["close"] = stock[46]["open"] * 0.99

    data = {"TEST": stock}
    costs = CostModel()
    opt = run_backtest(data, spy, mode="optimistic", rrs_threshold=2.0,
                       capital=25000, risk_pct=0.01, max_positions=5,
                       stop_atr=1.0, target_atr=2.0, costs=costs)
    hon = run_backtest(data, spy, mode="honest", rrs_threshold=2.0,
                       capital=25000, risk_pct=0.01, max_positions=5,
                       stop_atr=1.0, target_atr=2.0, costs=costs)
    print("SELFTEST — identical signal, different fill honesty")
    print(f"  optimistic: trades={opt.trades} net=${opt.net_pnl:,.2f} "
          f"costs=${opt.costs:,.2f}")
    print(f"  honest    : trades={hon.trades} net=${hon.net_pnl:,.2f} "
          f"costs=${hon.costs:,.2f}")
    ok = True
    if opt.trades < 1 or hon.trades < 1:
        print("  FAIL: expected at least one trade in each mode"); ok = False
    if not (hon.costs > opt.costs):
        print("  FAIL: honest costs should exceed optimistic (zero) costs"); ok = False
    if not (hon.net_pnl < opt.net_pnl):
        print("  FAIL: honest P&L should be worse (gap-through + costs)"); ok = False
    print("  RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="run no-network mechanics check and exit")
    ap.add_argument("--range", default="2y", help="Yahoo range (e.g. 2y, 5y)")
    ap.add_argument("--rrs", type=float, default=2.0, help="RRS threshold")
    ap.add_argument("--capital", type=float, default=25000)
    ap.add_argument("--risk", type=float, default=0.01)
    ap.add_argument("--max-positions", type=int, default=5)
    ap.add_argument("--stop-atr", type=float, default=1.5)
    ap.add_argument("--target-atr", type=float, default=3.0)
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

    try:
        from _yahoo_fetch import fetch_daily
    except ImportError:
        sys.path.insert(0, __file__.rsplit("/", 1)[0])
        from _yahoo_fetch import fetch_daily

    import time
    print(f"Fetching SPY + {len(UNIVERSE)} symbols ({args.range})...")
    try:
        spy = fetch_daily("SPY", args.range)
    except Exception as e:  # noqa: BLE001
        print(f"\nERROR: could not fetch SPY data: {e}")
        print("The operator datacenter IP is likely rate-limited by Yahoo. Run this")
        print("from an environment with market-data access, or point _yahoo_fetch at")
        print("another source. NOT fabricating results.")
        return 2
    data = {}
    for i, s in enumerate(UNIVERSE):
        time.sleep(2.0)
        try:
            data[s] = fetch_daily(s, args.range)
        except Exception as e:  # noqa: BLE001
            print(f"  {s}: skip ({e})")
    if len(data) < 5:
        print(f"\nERROR: only {len(data)} symbols fetched; too few to be meaningful.")
        return 2

    costs = CostModel()
    common = dict(rrs_threshold=args.rrs, capital=args.capital, risk_pct=args.risk,
                  max_positions=args.max_positions, stop_atr=args.stop_atr,
                  target_atr=args.target_atr, costs=costs)
    opt = run_backtest(data, spy, mode="optimistic", **common)
    hon = run_backtest(data, spy, mode="honest", **common)
    spy_end, spy_ret = spy_buy_hold(spy, args.capital)

    print(f"\nWindow: {spy[0]['date']} -> {spy[-1]['date']}  "
          f"({len(data)} symbols, RRS>={args.rrs})")
    print("=" * 68)
    for r in (opt, hon):
        print(f"  {r.mode:>10}: net ${r.net_pnl:>9,.0f} ({r.ret_pct:>6.1f}%)  "
              f"trades={r.trades:>3}  WR={r.win_rate:4.1f}%  costs=${r.costs:,.0f}")
    print(f"  {'SPY B&H':>10}: net ${spy_end-args.capital:>9,.0f} ({spy_ret:>6.1f}%)")
    print("=" * 68)
    print("  Manufactured edge (optimistic - honest): "
          f"${opt.net_pnl - hon.net_pnl:,.0f}")
    beat = "BEATS" if hon.ret_pct > spy_ret else "TRAILS"
    print(f"  Honest strategy {beat} SPY buy-and-hold "
          f"({hon.ret_pct:.1f}% vs {spy_ret:.1f}%).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
