# POST-MORTEM: The RRS Strategy and Why This Bot Is Where It Is

> Bootstrapped 2026-09-23 (operator run 001). This file did not previously exist; it is a
> reconstruction of the system's history and current honest state from the repository's own
> code, documentation, and a fresh independent backtest. Future operator instances should
> extend it, not overwrite it.

## The premise

The RDT Trading System implements the r/RealDayTrading (RDT) methodology: trade **Real
Relative Strength (RRS)** — stocks moving more than SPY-implied, normalized by ATR —
"market first," with layered filters (SPY gate, 50/200 SMA, VWAP, multi-timeframe, VIX,
sector RS, regime adaptation). The thesis: relative strength reveals institutional
accumulation, and stacking confirmation filters yields fewer but higher-quality trades.

An enormous amount of engineering has gone into this thesis: an agent architecture, ML
ensembles (XGBoost/RF/LSTM), an options module, intermarket analysis, 87 engineered
features, multi-broker execution, a full web dashboard, and a SaaS product overhaul. The
codebase is large and, by its own audit logs (Feb/March 2026), heavily hardened.

## The problem: the edge was never demonstrated

Beneath the engineering, the core question — *does this make money net of costs, better than
just holding SPY?* — has a discouraging answer in the repo's own artifacts:

1. **Documented walk-forward (2yr, `CLAUDE.md`):** the best configuration ("RDT Filters")
   returns **~6.9% total / ~3.4% annualized** over ~Feb 2024–Nov 2025.

2. **That number is gross.** Independent inspection (run 001) confirms the backtest engine in
   `backtesting/` contains **no commission and no slippage modeling at all.** Every reported
   backtest return is frictionless. With ~280 trades over two years, realistic costs
   (spread + slippage + commission on round trips) materially erode an already-thin result.

3. **SPY buy-and-hold over the same window returned ~+42.7%** (total return, dividends
   included). The bot's own best config underperforms simply owning SPY by roughly **6x**,
   before costs. Trailing 2yr and 1yr SPY (as of 2026-09): ~+38.9% and ~+17.9%.

4. **The strategy's own math is marginal-to-negative.** `ACTIONABLE_100X_STRATEGY.md` states
   the strategy runs at ~38% win rate, profit factor ~1.29, and computes a **slightly
   negative Kelly criterion** — i.e. the documented edge is at or below zero. Its proposed
   "path to 100% annual returns" is not a better strategy but a **pivot to selling signal
   subscriptions** ($49–$499/mo tiers). Monetizing signals you can't trade profitably
   yourself is a business pivot, not a trading edge — and a warning sign.

5. **Fresh reproduction (run 001, data through 2026-09-22)** shows the early walk-forward
   windows producing **negative** returns across all three configs (Baseline / Old / RDT),
   with the filters not reliably helping. See `data/operator_journal/entries/` for the full
   run. The "RDT Filters improve results" conclusion is not robust to a shifted data window.

## Why RRS-momentum struggles here

- **Frictionless backtests flatter high-turnover strategies.** ~280 trades/2yr is a lot of
  round trips for a ~$25k account; costs scale with turnover while the per-trade edge is tiny.
- **Filter stacking reduces sample size, not necessarily variance of edge.** "98% of signals
  filtered out" (per `CLAUDE.md`) leaves a small, possibly over-fit residual. Six quarterly
  windows is thin evidence for a multi-filter system with many tunable parameters.
- **A near-zero-Kelly edge cannot be levered into returns.** Raising risk-per-trade multiplies
  a ~0 expectancy by a larger number — still ~0, with higher ruin risk. The `ACTIONABLE`
  doc's own Kelly calc says this explicitly.
- **The benchmark is brutal.** 2024–2026 was a strong bull market for SPY. A long-biased
  stock-picking system that trades in and out captured a small fraction of a move it would
  have fully captured by holding.

## What this means for the operator

The honest prior is that **no edge over SPY buy-and-hold has been demonstrated, and the most
credible internal evidence suggests the edge is ≤ 0 once costs are included.** This does not
prove no edge can exist — but it shifts the burden of proof heavily. The operator's job is
not to keep adding filters and features; it is to either (a) find, with cost-inclusive
out-of-sample evidence, a variant that genuinely beats buy-and-hold, or (b) confirm the
negative result rigorously and recommend wind-down or a fundamental pivot to the human.

The single most important discipline going forward: **every performance claim must be net of
honest costs and benchmarked against SPY buy-and-hold over the identical window.** The
repo's history is a case study in what happens when that discipline is absent — years of
engineering on top of an unproven, likely-negative edge.
