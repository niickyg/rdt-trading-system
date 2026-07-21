# POST-MORTEM: The RRS Strategy vs. Reality

> Bootstrapped 2026-07-21 by operator Run 001. This file did not previously
> exist; the scheduled operator prompt referenced it as "the history of why the
> bot is in its current state," so Run 001 created it from the evidence on hand
> (CLAUDE.md, the persisted signal metrics, and a fresh reproducible backtest).
> Future operators: append, correct, and extend — do not delete history.

## What this bot is

An autonomous system implementing the r/RealDayTrading (RDT) methodology:
Real Relative Strength (RRS) scanning + a stack of "market-first" filter gates
(SPY hard gate, 50/200 SMA, VWAP, multi-timeframe alignment), plus VIX / sector
/ regime / intermarket overlays, and an advisory-only ML layer. It trades
stocks and (optionally) options, on paper, against an IBKR paper account.

## The central problem: it has never been shown to beat buy-and-hold

The project's documentation (CLAUDE.md) presents the RDT filters as a success,
citing a 2-year walk-forward: **"C) RDT Filters: +6.9%, best of three."** That
framing is the root misconception. The three things compared were all *variants
of the strategy* (baseline vs. old filters vs. RDT filters). **The strategy was
never compared against the one benchmark that matters — SPY buy-and-hold.**

Run 001 reproduced the project's own walk-forward (`run_walkforward_v2.py`,
unmodified, on fresh 2024–2026 data) and added the missing benchmark:

| Config — identical window 2024-03-27 → 2026-04-09 (~2yr) | Total | Annualized |
|----------------------------------------------------------|-------|-----------|
| A) Baseline (no filters)                                 | +4.4% | 2.2% |
| B) Old filters                                           | +4.5% | 2.2% |
| C) RDT filters (best config)                             | +5.4% | 2.7% |
| **SPY buy-and-hold**                                     | **+33.5%** | **15.2%** |

The best configuration captured **~1/6 of the benchmark's return** while taking
active-trading risk (worst single day −$226, max drawdown −$818 on $25k).

### It is worse than it looks

- **The +5.4% is GROSS.** `backtesting/engine.py` / `engine_enhanced.py` model
  **no commissions and no slippage** (grep confirms: zero cost terms). Config C
  took **266 trades** over the window. Realistic round-trip friction on a $25k
  book plausibly consumes a large fraction — possibly all — of the 5.4% gross.
- **98%+ of raw signals are filtered out.** The filter stack is doing enormous
  work to convert a large signal stream into a handful of trades that, in
  aggregate, still lose to doing nothing.

### The live signal record is empty

`data/signals/signal_metrics.json` (last live scan 2026-03-05): 880 scans,
120 emitted signals, **only 2 tracked outcomes (1 win, 1 loss).** There is no
live P&L track record — the bot has never demonstrably made or lost real money
in a measurable way. Emitted signals were **119 short : 1 long**, because the
SPY hard gate blocks *all* longs whenever SPY is below its 50 & 200 EMA (a
bearish Feb–Mar 2026 tape). That gate is working as coded — but note it
discards RDT's core edge (buying *relative strength*) precisely when the market
is weak, which is when relative-strength longs are most distinctive.

## Why the daily backtest may understate — and why that doesn't rescue it

The live strategy is **intraday** (5-minute RRS, VWAP, first-hour timing). The
walk-forward is a **daily-bar proxy** that cannot simulate VWAP or the
first-hour filter. So it is *possible* an intraday edge exists that daily bars
can't see. But:

1. This daily backtest is the **project's own primary evidence** for the
   strategy. There is no intraday backtest, no results file, no live record
   that shows an edge. The intraday edge is **asserted, not demonstrated.**
2. A long/short system returning +5% gross while the market returns +33% is
   capturing neither beta nor meaningful alpha. Intraday microstructure would
   have to add ~28 points over 2 years to merely tie buy-and-hold — an
   extraordinary claim with zero supporting evidence in this repo.

## The honest bottom line

As of Run 001, **no tested configuration of this bot beats SPY buy-and-hold**,
and net of realistic costs the best config is plausibly break-even to negative.
The burden of proof is now inverted: the next operator's job is not to add
another filter, but to find **any** rigorous, cost-aware, out-of-sample
evidence that a config beats buy-and-hold. If successive sessions cannot, the
correct recommendation is **wind-down** — the user is better served holding SPY.

## Open threads for future operators

1. Build a **cost-aware** backtest (commissions + slippage + spread) and re-run
   the 3-way + SPY comparison. Quantify how much of the 5.4% survives costs.
2. Obtain/construct an **intraday** backtest (Yahoo gives ~60d of 5m bars) to
   test whether the *actual* live strategy has any edge the daily proxy misses.
3. Test the strategy in a **flat/bear** regime specifically — its only defense
   is "it protects capital in downturns," which is measurable, not assumed.
4. If 1–3 keep failing: draft the wind-down / escalation recommendation.
