# Post-Mortem: The RRS Strategy and How the Bot Got Here

> History file for the autonomous operator. Written at genesis (2026-09-02) from
> the evidence committed in the repo. Future instances: update this as the
> understanding evolves, but never delete the record of what was believed and
> why.

## What the bot is

An autonomous trading system built around **Real Relative Strength (RRS)**:

```
RRS = (Stock % Change − SPY % Change) / ATR%
```

Long candidates when RRS is strongly positive (stock outrunning SPY), short
candidates when strongly negative. On top of this core signal sits a large
apparatus: RDT "market-first" filter gates (SPY gate, 50/200 SMA, VWAP, MTF), a
VIX regime filter, sector RS, intermarket analysis, a stacked ML ensemble
(XGBoost + RF + meta-learner), a regime detector (HMM/heuristic), and an options
overlay. The methodology is borrowed from the r/RealDayTrading community.

## The uncomfortable truth (as of genesis)

**There is no evidence the bot beats SPY buy-and-hold, and substantial evidence
it does not.** This is the central fact every future instance must sit with
before adding anything.

### 1. Even the best *self-reported* backtest loses to SPY

`CLAUDE.md` reports the strongest walk-forward result (2yr, Config C "RDT
Filters", $25K): **+$1,716 total return = 6.9% over ~21 months (~3.4%
annualized).** Over that same Feb-2024→Nov-2025 window, SPY buy-and-hold
returned far more (the S&P 500's 2024 total return alone was ~+25%). So even
taking the bot's own best number at face value, **it dramatically underperforms
the default of just holding SPY** — which is the only benchmark the mandate
cares about. No existing backtest script even computes the SPY comparison; they
compare filter configs against each other, never against buy-and-hold.

### 2. The strategy documents contradict themselves

`ACTIONABLE_100X_STRATEGY.md` states the strategy has "profit factor 1.29" and
"38% win rate," and computes a **negative Kelly (~−0.02)**. But at 38% win rate
with the stated avg win $70 / avg loss $45, the profit factor is
`(0.38·70)/(0.62·45) ≈ 0.95` — i.e. **below 1.0, a losing system**, not 1.29.
The numbers are mutually inconsistent. When a source's headline stats don't
reconcile with its own inputs, none of its numbers can be trusted. This casts
doubt on the 6.9% figure too.

### 3. The edge is marginal-to-negative by the strategy's own math

A negative Kelly criterion means the per-trade edge is essentially zero. That is
why the "100X" documents quietly pivot away from trading: their real plan to hit
the return target is **selling signal subscriptions and API access**
($10-20K/yr of the $25K goal), not trading profit. That pivot is the tell — if
the trading edge were real, you wouldn't need to sell shovels. Per the mandate,
selling the system is *not* the bot being profitable.

### 4. There is almost no real track record

`data/signals/signal_metrics.json`: **880 scans, 120 emitted signals, and only
2 tracked outcomes** (1 target hit, 1 stop-out). `data/signals/signal_history.json`
covers a **single month** (2026-02-03 → 2026-03-05), 1986 raw signals. The bot
has effectively never accumulated a statistically meaningful set of *closed*
trades. Every performance claim rests on backtests, not lived results.

### 5. The metrics files disagree with each other

`signal_metrics.json` summarizes 120 signals as **1 long / 119 short**, but the
actual `signal_history.json` for the same period is **1687 long / 299 short**.
The counters are out of sync with the underlying data — a data-integrity problem
that further undermines confidence in the reported numbers. (Note: the RRS sign
convention itself is correct — longs carry positive RRS, shorts negative — so
this is a bookkeeping inconsistency, not a signal-direction bug.)

### 6. The "sophistication" is largely non-functional

- **ML ensemble** (`models/ensemble/metrics.json`): cross-validated AUC **0.543**
  (barely above the 0.5 coin-flip) while train AUC is **0.9925** — a textbook
  overfit. Train precision/recall/F1 are all **0.0**. `CLAUDE.md` itself
  concedes "ML is advisory-only" and the exit predictor (43.3% accuracy) is
  "SKIP."
- **Regime detector** (`models/training_metrics.json`): silhouette score
  **−0.087** (negative = clusters worse than random), and it labels 1030 of 1056
  samples "low_volatility." It is not meaningfully detecting regimes.

The measurable value, per `CLAUDE.md`, comes from simple rule-based filters — the
ML and regime layers are decoration that add parameters (and overfitting risk)
without demonstrated P&L.

## What this means for the operator

The honest prior, entering any run, is: **the RRS system has not been shown to
beat SPY buy-and-hold net of costs, and its own math suggests the edge is at or
below zero.** The burden of proof is on any change to demonstrate — with a
reproducible, cost-inclusive, out-of-sample, walk-forward test against SPY — that
it clears that bar. Adding leverage (the "100X" TQQQ/SOXL path) to a
zero-edge strategy multiplies losses, not gains, and must be treated with extreme
skepticism.

If successive runs cannot produce that evidence, the mandated outcome is not
"try harder" — it is to **document the null result and recommend winding down
active trading in favor of SPY buy-and-hold.**

## Open questions for future instances

1. Run `scripts/honest_benchmark.py` on real data (human infra or IBKR MCP): does
   *any* configuration beat SPY buy-and-hold net of costs, out-of-sample?
2. If shorts dominate emitted signals in bull markets, are the RDT gates
   systematically fighting the tape? Quantify emitted-signal direction vs SPY
   trend.
3. Is there a genuinely low-parameter version of the core signal that holds up
   out-of-sample, stripped of the ML/regime decoration?
4. What are honest per-trade costs for this account/broker, and how much of the
   thin backtested edge do they consume?
