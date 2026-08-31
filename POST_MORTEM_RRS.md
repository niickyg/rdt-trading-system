# POST-MORTEM: RRS Strategy & System State

> **Provenance.** This file was created 2026-08-31 by the first operator
> instance. No prior post-mortem existed in the checkout. It reconstructs the
> honest state of the system from the code and the persisted signal data that
> *are* in the repo. Where a claim could not be independently verified in this
> environment, it is labelled **[unverified]**. Future instances should append,
> correct, and date their additions rather than overwrite.

---

## TL;DR

- The system is an elaborate, feature-rich RDT/RRS momentum bot (scanner, ML
  ensemble, options module, multi-broker, dashboards). The *engineering* is
  extensive.
- **The bot cannot currently measure whether its own trades win or lose.** The
  emitted-signal outcome recorder (`SignalMetricsTracker.record_outcome`) is
  **never called anywhere in the codebase**, and the 1,986 logged signals carry
  **zero outcome labels**. Outcome coverage in `signal_metrics.json` is **2 of
  120 (1.7%)**. There is no closed feedback loop.
- The **only quantitative profitability claim** that exists (the walk-forward
  table in `CLAUDE.md`) reports the best configuration at **+6.9% over ~1.85
  years ≈ 3.4% annualized**. Over that same window (Feb 2024 – Nov 2025) SPY
  buy-and-hold returned *far* more (order of ~10–25%/yr). **By its own best
  documented number, the active strategy loses badly to SPY buy-and-hold.**
  [The walk-forward number itself is **[unverified]** here — see "Cannot verify".]
- Net: on all evidence available in this environment, **the bot does not beat
  SPY buy-and-hold, and it lacks the instrumentation to prove otherwise.**

## What the RRS strategy is

`RRS = (Stock %Δ − SPY %Δ) / ATR%`. Longs on strong positive RRS with the market
trend, shorts on strong negative RRS against weak stocks, filtered through
sequential gates (SPY regime → 50/200 SMA → VWAP → multi-timeframe) plus VIX,
sector, intermarket, and regime overlays. Planned trades use a clean **2.0
reward:risk** (median and mean R:R across all 1,986 logged signals is exactly
2.00), implying a **33.3% breakeven win rate before costs.** After realistic
costs (commission + slippage + spread on a ~$25k account trading intraday), the
required win rate is meaningfully higher — and it is *unmeasured*.

## The evidence that exists in the repo (as of 2026-08-31)

Reproduce with: `python3 scripts/analyze_signal_history.py`

| Fact | Value | Source |
|---|---|---|
| Signals logged | 1,986 | `data/signals/signal_history.json` |
| Window covered | 2026-02-03 → 2026-03-05 (~1 month) | same |
| Direction split | 1,687 long / 299 short | same |
| Unique symbols | 48 | same |
| Planned R:R (median/mean) | 2.00 / 2.00 | same |
| **Outcome labels on signals** | **0** | same |
| total_signals / total_outcomes | 120 / 2 | `signal_metrics.json` |
| target_hits / stop_outs | 1 / 1 | same |
| **Outcome coverage** | **1.7%** | same |

The signal data is ~6 months stale (ends 2026-03-05; today is 2026-08-31),
consistent with it being a snapshot from a paper run on the human's local infra,
which this remote agent cannot reach.

## Root cause: the missing feedback loop

1. `scanner/realtime_scanner.py` calls `get_metrics_tracker().record_scan(...)`
   on every scan — it counts signals produced.
2. Nothing ever calls `record_outcome(signal_id, hit=...)`. Grep confirms zero
   call sites for the file-based tracker's `record_outcome` in
   `scanner/agents/web/api`.
3. `agents/outcome_tracker.py` only tracks *rejected* signals (for threshold
   tuning), depends on a DB historical cache that was not populated, and does not
   label emitted signals as win/loss either.

**A bot that never records trade outcomes cannot be made profitable by
iteration** — every "improvement" is unfalsifiable. Closing this loop is
prerequisite #1 for the entire mission.

## What this environment cannot verify

- The `CLAUDE.md` walk-forward results (`+$1,716 / 6.9%` etc.). No market-data
  access (Yahoo/yfinance blocked by egress policy), no cached price data, so the
  backtest cannot be re-run here. Treat those numbers as **[unverified]** until a
  session with data access reproduces them.
- Any live/paper P&L. No access to the human's running container or database.

## Honest verdict

Beating SPY is the bar. The single documented estimate (3.4%/yr) is a fraction of
SPY buy-and-hold for the same period, and even that estimate is unverified here.
The system also cannot currently produce the evidence that would change this
verdict. Until (a) the outcome feedback loop is closed and (b) a costed
walk-forward is reproduced with real data, the responsible position is:
**not demonstrated to be profitable; on documented evidence, underperforms SPY.**

## Recommended path (in priority order)

1. **Close the feedback loop.** Wire emitted-signal outcomes (target hit / stop /
   time-exit + realised PnL) into a persisted, labelled store. Without this,
   nothing else is measurable. (Design it so it works in paper mode against the
   broker's fills, not just backtests.)
2. **Reproduce a costed walk-forward** in an environment with market-data access;
   compare net-of-cost return *and* risk-adjusted return against SPY
   buy-and-hold over the same window. Publish the number with its provenance.
3. **Decide on the evidence.** If the honest, costed, outcome-labelled results
   still trail SPY, escalate to the human with the wind-down recommendation per
   `MANDATE.md` §7 rather than continuing to tune parameters.
