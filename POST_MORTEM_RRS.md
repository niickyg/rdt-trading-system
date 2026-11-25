# POST-MORTEM: The RRS Strategy

> **Reconstruction note (2026-11-25):** The original `POST_MORTEM_RRS.md`
> referenced by the operator task prompt did not exist in the repository (not in
> the working tree, not in any branch's git history). This document was
> reconstructed by the first autonomous operator session **from evidence found
> in the codebase**. It is not the original author's account; it is an honest
> inference from committed artifacts. Dates of events are approximate, taken from
> file timestamps and commit history. Treat specific claims as "supported by file
> X" rather than as first-hand testimony.

## One-paragraph summary

The system implements an r/RealDayTrading-style **Real Relative Strength (RRS)**
momentum strategy with an elaborate stack of filters (SPY gate, SMA gate, VWAP,
MTF, VIX, sector, regime, intermarket) and an ML layer (XGBoost + RF + LSTM
ensemble, regime detector). Despite the sophistication, **no artifact in the repo
demonstrates a trading edge that beats SPY buy-and-hold net of costs.** The best
documented result is in-sample and overfit; the one walk-forward figure
underperforms SPY badly; the ML has essentially no predictive value; and the
project's most recent direction has been to package the system as a SaaS "signal
service" rather than to prove the underlying edge. By the repo's own arithmetic,
the strategy's Kelly fraction is slightly **negative**.

## The evidence (as of 2026-11-25)

### 1. The best backtest is in-sample and the edge is marginal

`data/optimization/optimization_2025-12-29.json` tested 180 parameter
combinations and reported the best as:

- Total return: **~6.8%** (initial capital $25,000)
- Win rate: **38%**
- Profit factor: **1.29**
- **Sharpe ratio: 0.11**
- 215 trades, ~4.4 day holds

This is the *best of 180 combinations fit on one dataset* — overfit by
construction. A Sharpe of 0.11 is statistically indistinguishable from no edge.

### 2. The repo itself computes a NEGATIVE Kelly criterion

`ACTIONABLE_100X_STRATEGY.md` (committed to the repo) states plainly:

> `kelly = -0.02  # NEGATIVE!`
> "**Critical Finding:** The Kelly Criterion is slightly negative, meaning the
> current edge is marginal. Increasing position size actually increases risk of
> ruin without improving returns."

### 3. The walk-forward result loses to SPY

`CLAUDE.md` reports a 2-year (Feb 2024 – Nov 2025) 3-way walk-forward. The best
arm ("RDT Filters") returned **$1,716 / 6.9% total → ~3.4% annualized**.

Benchmark (IBKR, 2026-11-25): **SPY +12.01% trailing 1yr, +87.29% trailing 3yr.**
Over any comparable window SPY buy-and-hold beats the strategy by a wide margin —
and the strategy's number is the *optimistic* case.

### 4. The ML layer has no demonstrated predictive value

- `models/ensemble/metrics.json`: cross-validated AUC ≈ **0.54** (0.50 = random);
  train precision/recall/F1 = **0.0** on the positive class.
- `models/training_metrics.json` (regime detector): silhouette score
  **−0.087** (negative = worse than no structure); regime distribution degenerate
  (1030 "low-vol" vs 1 "bull-trending" out of 1056).
- CLAUDE.md concurs: "Rule-based filters provide all measurable improvement; ML is
  advisory-only"; "Exit Predictor: 43.3% accuracy — SKIP."

### 5. Signal generation is lopsided and barely tracked

`data/signals/signal_metrics.json`: 880 scans → 120 signals, of which **119 short
vs 1 long**, and only **2 signals ever reached a tracked outcome** (1 win, 1
loss). Whether the 119:1 short skew is a bug, a regime artifact (the scans span a
Feb–Mar 2026 window), or a filter side-effect is **unresolved** and worth
investigating — but note it does not change the edge verdict, which holds in both
directions.

### 6. The project pivoted to selling signals, not proving an edge

The most recent commit is `feat: SaaS product overhaul — toast system, skeletons,
animations, landing/pricing/login/register pages, onboarding`. `ACTIONABLE_100X_STRATEGY.md`
proposes reaching its return goal with **40–60% of the money coming from "Signal
Service Revenue"** rather than from trading. The `*_100X.md` documents set a
"100% annual return" target that is inconsistent with a negative-Kelly strategy.

### 7. The live paper account shows no activity

IBKR MCP on 2026-11-25: **net liquidation $5**, zero open positions, zero trades
in the trailing 90 days. (This may be a different/empty paper account than the
`DUP995654` $25K account CLAUDE.md describes; the operator cannot confirm which
account the MCP points at.) Either way, there is no live paper track record to
evaluate.

## Why the bot is "in its current state"

The honest reading of the artifacts: a great deal of engineering effort went into
breadth (filters, ML, options module, dashboards, SaaS packaging) and almost none
produced a validated, out-of-sample, cost-aware edge over SPY. When the core
numbers stayed marginal, the project's energy shifted toward *monetizing the
signals* and chasing a "100X" narrative rather than confronting the negative-Kelli
finding. This post-mortem exists to stop the next operator from repeating that
loop.

## What would change the verdict

A single, clean experiment: a walk-forward backtest with **strict out-of-sample
discipline** (parameters frozen before each test window), **honest transaction
costs** (commission + slippage + spread), reported **against SPY over the identical
window**. If that produces a positive, cost-aware, SPY-beating result that is
stable across windows, the strategy has an edge. Until it does, assume it does
not.
