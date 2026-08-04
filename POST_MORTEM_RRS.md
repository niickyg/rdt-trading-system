# Post-Mortem: The RRS Trading System

> **Provenance.** This file was expected by the operator scheduled-task prompt but did not
> exist in the repo. It was reconstructed on **2026-08-04** by the first operator instance from
> the only trustworthy sources available: the live IBKR paper account (via MCP) and the repo's
> own data/config. Where a claim comes from the codebase's own documentation rather than
> measured data, it is labeled `[REPO-CLAIM]`. Measured facts are labeled `[LIVE]`.

---

## The one-sentence story

An elaborately engineered r/RealDayTrading momentum bot — RRS scanning, an 8-layer filter
stack, an ML ensemble, options, multi-broker plumbing — **lost 61.5% of its paper account in
its first month of live paper trading, then went dormant**, while the market it was trying to
beat (SPY) rose ~10% over the same window. The sophistication is real; the profitability is not.

---

## The scoreboard (measured 2026-08-04)

| | Value | Source |
|---|---|---|
| Bot paper account, TWR since inception (2026-02-26) | **−61.5%** | `[LIVE]` get_pa_performance_all_periods |
| SPY buy-and-hold, same window (2026-02-26 → 2026-08-04) | **+10.3%** | `[LIVE]` 756733 ($689.30 → $760.12) |
| **Underperformance vs. the mission benchmark** | **~72 points** | derived |
| Current net liquidation value | **$5** | `[LIVE]` get_account_summary |
| Open positions | 0 | `[LIVE]` get_account_positions |
| Last scanner activity | **2026-03-05** (≈5 months stale) | `data/signals/signal_metrics.json` |

TWR is time-weighted and **cash-flow-insensitive** — the −61.5% is trading performance, not the
artifact of a deposit or withdrawal. The NAV path tells the story bluntly: **$50 → $21 in the
first month** (a −58% hole before any capital was added), a top-up to ~$521, a slow bleed to
~$477 through spring, then a collapse to **$5** in early July, flat since.

---

## Why it failed (root causes, most important first)

### 1. It was flown without instruments. The measurement loop never worked.
Across the entire signal history (1,986 signals logged, 880 scans) exactly **2** outcomes were
ever recorded — 1 target hit, 1 stop out. `[LIVE from signal_metrics.json]` The system emitted
thousands of signals and almost never checked whether they were right. With no feedback loop,
every "improvement" ever made to this bot was unfalsifiable. **You cannot make a strategy
profitable that you are not measuring.** This is the deepest problem and the precondition for
fixing any of the others.

### 2. The backtest lied, and nobody reconciled it.
The repo's own docs headline a walk-forward result of **+6.9% over 2 years** for the "RDT
filters" configuration `[REPO-CLAIM, CLAUDE.md]`. The live paper account did **−61.5%**. Two
possibilities, both damning: either the backtest is materially wrong (look-ahead bias,
no/underestimated costs, survivorship, optimistic fills), or the live execution diverges wildly
from what was backtested. Nobody caught the gap because of root cause #1. **Distrust every
backtest in this repo until it is reconciled against realized results.**

Note too: even if the +6.9%/2yr backtest were *true*, it would **still fail the mission** —
SPY buy-and-hold over that window returned far more. The system was optimizing to beat a bar
(some notion of "good trades") that was never the actual bar (beat SPY).

### 3. Complexity outran validation.
Signals pass through ~8 stacked, independently-toggleable filters — SPY hard gate, 50/200 SMA
gate, VWAP gate, lightweight MTF, VIX regime, sector RS, regime-adaptive thresholds,
intermarket — that together **reject ~98% of raw signals** `[REPO-CLAIM]`. There is no evidence
in the repo that any individual layer contributes positive realized edge; they were added on
methodology-plausibility, not measured contribution. An 8-parameter filter tuned on one
2-year window is a machine for overfitting. The ML layer is, per the repo's own notes,
"advisory-only" with an exit model at 43% accuracy (barely above random).

### 4. Direction/regime bet went the wrong way at the worst time.
The recorded signal metrics are lopsided: in the tracked window **119 of 120** emitted signals
were **shorts** `[LIVE from signal_metrics.json]`, generated into a market that then rose ~10%.
A momentum/short-heavy book fighting an uptrend is a straightforward way to lose 58% in a month.

### 5. It's dead. A dormant bot's first problem is dormancy.
No scan since 2026-03-05. Whatever its merits, a system that isn't running has an expected
return of exactly zero (minus maintenance) — strictly worse than the SPY bar it's measured
against.

---

## What is NOT the problem
The engineering quality is genuinely high (security hardening, thread-safety, clean agent
architecture). **That is precisely the trap:** effort and polish were spent on plumbing and
features while the core question — *does this signal make money net of costs, out of sample,
better than doing nothing?* — went unanswered. Do not add more of what already exists in
abundance. Answer the core question.

---

## The mandate for whoever works on this next
1. **Fix measurement first.** An honest, cost-aware, out-of-sample edge test is worth more than
   any new feature. The IBKR `get_price_history` MCP tool makes this possible in the remote env
   even though yfinance is blocked (see MANDATE §3).
2. **Reconcile the backtest to reality** or stop trusting it.
3. **Subtract before adding.** Strip unvalidated filters; measure the naked signal; only keep a
   layer that demonstrably pays for itself out of sample.
4. **Hold the SPY bar honestly.** If, after real measurement, nothing beats SPY buy-and-hold,
   the correct recommendation is wind-down, and saying so is a success.

*— First operator instance, 2026-08-04*
