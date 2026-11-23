# POST-MORTEM: RRS Strategy & Bot State

> **PROVENANCE NOTE (2026-11-23):** The scheduled operator routine instructed
> each run to read a `POST_MORTEM_RRS.md` describing "the history of why the
> bot is in its current state." **That file was never committed to this
> repository.** This document is a *forensic reconstruction* assembled by the
> operator agent from the only evidence available in the repo — it is **not**
> the original post-mortem and does not claim to recover decisions that were
> never recorded. If the real post-mortem exists elsewhere (e.g. the user's
> local machine), it should replace this file.

## What the committed evidence actually shows

Source files inspected: `data/signals/signal_metrics.json`,
`data/signals/active_signals.json`, `data/signals/signal_history.json`,
`CLAUDE.md`, git history (8 commits total).

### 1. Outcome tracking has effectively never run
`signal_metrics.json` (as of last scan `2026-03-05T16:50:50Z`):
- `total_scans`: 880
- `total_signals`: 120  (119 short, **1 long**)
- `scans_with_no_signals`: 802
- **`total_outcomes`: 2** → `target_hits`: 1, `stop_outs`: 1

So across 880 scans the system recorded **exactly two resolved trade
outcomes** (one win, one loss). That is statistically meaningless — there is
no basis in this data to claim the strategy is or is not profitable.

### 2. Signal data is stale
The newest timestamp anywhere in the signal files is **2026-03-05**.
`signal_history.json` holds 1,986 generated *signals* (not trades with
outcomes), spanning roughly Feb–Mar 2026. As of this writing (2026-11-23) the
recorded activity is **~8.5 months stale.**

### 3. Headline backtest claims are unverifiable as committed
`CLAUDE.md` presents a "Walk-Forward Backtest Results (2 Years)" table
(e.g. RDT filters: +$1,716 / 6.9% / 49.5% WR / PF 1.24). **No committed
artifact reproduces these numbers** — no result JSON/CSV, no run log. The
backtest dependencies (`pandas`, `numpy`, `scipy`, `yfinance`) are **not
installed** in this environment, so the table could not be regenerated here.

### 4. Documentation drift undermines the docs' credibility
`CLAUDE.md` documents modules that **do not exist** in the repo, including the
entire `research/` package (factors, factor_tester, signal_combiner,
backtest_harness) and `scripts/run_signal_research.py`. Claims in a doc that
describes absent code should be treated as aspirational until verified.

### 5. Hype-document red flags
The repo root carries `ACTIONABLE_100X_STRATEGY.md`, `QUICK_START_100X.md`,
`WEALTH_STRATEGY_100X.md`, `WEALTH_OPTIMIZATION.md`. "100x" framing is a
classic warning sign for an unproven system and is inconsistent with the
mandate's honest-profitability goal. They should be reconciled or removed.

## Honest conclusion
There is **insufficient evidence to conclude the bot is profitable**, and the
strongest available claims (the backtest table) are unverifiable as committed.
The heavy short skew (119/120) over a single ~1-month window is a curiosity,
not a track record. Before any strategy change, the system needs: (a) a
ratified MANDATE, (b) a reproducible, committed backtest, and (c) real tracked
P&L vs SPY over a meaningful sample.

## Recommended next steps (for a human or a governance-ratified run)
1. Ratify `data/operator_journal/MANDATE.md` (or replace with the real one).
2. Supply the genuine post-mortem if one exists; otherwise adopt this.
3. Install backtest deps and commit a reproducible walk-forward result so the
   CLAUDE.md table can be verified or corrected.
4. Reconcile documentation drift (remove/implement `research/`, etc.).
5. Decide whether the "100x" wealth docs stay or go.
