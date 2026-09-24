# POST-MORTEM: The RRS Momentum Strategy

> Bootstrapped 2026-09-24 (operator run #001). The scheduled operator task
> referenced this file as "the history of why the bot is in its current state,"
> but no such file existed in the repository. This is an honest reconstruction
> from the code, the committed strategy docs, and a fresh backtest run. Future
> operators: append, correct, and date your additions.

## What the bot is

An autonomous system implementing the r/RealDayTrading (RDT) methodology:
- **Real Relative Strength (RRS):** `RRS = (stock %chg − SPY %chg) / ATR%`.
  Long strong stocks (RRS > 2), short weak stocks (RRS < −2), only when the
  broad market ("market first") agrees.
- A stack of filters (SPY gate, 50/200 SMA, VWAP, lightweight MTF, VIX regime,
  sector RS, intermarket) that remove ~98% of raw signals.
- Multi-broker execution (paper / IBKR / Schwab), an options module, an ML
  layer (advisory only — the docs themselves say ML adds no measurable edge),
  and a large Flask/SaaS dashboard surface.

## The core problem (measured, not asserted)

RDT is an **intraday day-trading** methodology. RRS, VWAP, and MTF are intraday
constructs. But the only backtests in this repo run on **daily bars**
(`scripts/run_walkforward_v2.py`, `backtesting/engine*.py`). yfinance cannot
supply 2 years of 5-minute bars, so the intraday strategy has **never been
validated on the timeframe it actually trades.**

What the daily-bar proxy *does* show, re-run fresh on 2026-09-24 over a trailing
2-year walk-forward (30-stock core watchlist, $25k, Config C exits):

| Config | Total return (2yr) | Annualized | Trades |
|--------|-------------------:|-----------:|-------:|
| A) Baseline (no filters)   | +5.76% | +2.8%/yr | 237 |
| B) Old filters             | +6.17% | +3.0%/yr | 427 |
| **C) RDT filters (best)**  | **+8.69%** | **+4.3%/yr** | 277 |

**SPY buy-and-hold over the same 2 years: +37.50% (+17.29%/yr).**

So the *best* configuration earned **+$2,172** on $25k while simply holding SPY
earned **+$9,375** — over 4x more, with zero trades, zero overnight risk, and
zero monitoring. The strategy trails buy-and-hold by **~13 percentage points per
year.** (Reproduce: `python scripts/benchmark_vs_spy.py --days 730
--strategy-return-pct 8.69`.)

## Why it's in this state

1. **Long-biased momentum vs. a market that went straight up.** 2024–2026 was a
   strong bull run. A strategy that sits in cash 98% of the time and takes small,
   tightly-stopped positions structurally cannot keep up with beta in that
   regime — and it didn't.
2. **The filters optimize the wrong thing.** Removing 98% of signals improved
   risk-adjusted numbers vs. baseline, but the whole envelope is far below the
   do-nothing alternative. Better than a worse active strategy ≠ good.
3. **Curve-fit params.** "Config C" (1.5x stop, 2.0x target, etc.) was selected
   by optimization over this same history; the forward-looking edge is likely
   smaller than shown.
4. **Costs are optimistic.** ~277 daily-bar round trips ignore the real
   intraday spread/slippage the live strategy would pay.
5. **Scope drift.** Recent commits and top-level docs (`ACTIONABLE_100X_STRATEGY.md`,
   `WEALTH_STRATEGY_100X.md`, `QUICK_START_100X.md`, a "SaaS product overhaul")
   pushed effort toward product/marketing framing and "100x" language, not
   toward answering the one question that matters: does it beat buy-and-hold?

## The honest bottom line

There is **no valid evidence the bot is profitable**, and clear evidence its
daily-bar proxy underperforms buy-and-hold by a wide margin. The intraday
strategy it actually runs has never been tested on intraday data, and no live
paper-trading track record is reachable from the remote operator environment.

Before any further optimization is justified, the bot needs ONE of:
- an intraday backtest with realistic costs on real 5-minute data, or
- a real, dated paper-trading track record measured against SPY buy-and-hold.

Until then, "make it profitable" has no measurable target and the default
recommendation is: **do not scale, do not go live, and treat wind-down (park the
capital in SPY) as the leading option** unless intraday/live evidence overturns
the daily-bar picture.
