# POST-MORTEM: The RRS Strategy and How the Bot Got Here

> Reconstructed 2026-10-09 by the genesis operator run. The scheduling prompt
> referenced this file as pre-existing history; it did not exist in the repo or
> git history. This version is assembled from the repository's own documents
> (`WEALTH_OPTIMIZATION.md`, `ACTIONABLE_100X_STRATEGY.md`, `CLAUDE.md`), the
> persisted signal data, and three independently-run backtests. Treat earlier
> sections as reconstruction; append future post-mortems below.

## The one-paragraph version

This bot implements the r/RealDayTrading "Real Relative Strength" (RRS)
methodology with an elaborate stack of filters, agents, ML models, and an
options module. It has never been shown to make money. Every honest backtest
puts its best configuration at roughly **3–12% total over two years (~4–6%
annualized)**, while SPY buy-and-hold returned **~40%+ over the same window
(~20%/yr)**. The system also never measured its own live results: of 1,986
signals logged in production, **zero** had outcomes recorded. The surrounding
documents escalate from a 2.85% baseline to "$10M wealth" and "100% annual
return" targets whose concrete plan is not trading edge but **selling the
signals to other people**. The engineering is extensive and mostly sound; the
trading edge is absent.

## Timeline (from the repo's own docs)

1. **Baseline RRS strategy.** `RRS = (stock %chg − SPY %chg) / ATR%`. Longs on
   RRS > 2, shorts on RRS < −2. Documented baseline: **2.85%/yr, 33.7% win
   rate, profit factor 1.23** (`WEALTH_OPTIMIZATION.md`).
2. **Filter stack added** ("market first"): SPY hard gate, 50/200 SMA gate,
   VWAP gate, lightweight MTF, VIX regime, sector RS, regime-adaptive
   thresholds, intermarket (Murphy) analysis, news sentiment. These filter out
   **97–98% of raw signals**. Net effect in the 2-year walk-forward: profit
   factor rises ~1.35 → ~1.44, annualized return ~4.3% → ~6.1%. A real but
   small improvement that leaves the strategy far below buy-and-hold.
3. **ML layer added** (ensemble, regime detector, drift, dynamic sizer, exit
   predictor). Per `CLAUDE.md`'s own notes: exit predictor 43% accuracy
   (barely above random, "SKIP"); regime detector is statistically degenerate
   (silhouette −0.09, ~97% of samples in one cluster). ML is advisory-only and
   contributes no measured edge.
4. **Options module added** (chains, Greeks, IV-regime strategy selection,
   combo execution). Large surface area; no evidence it is net profitable.
5. **"Wealth" / "100X" documents.** Targets of 15–25%/yr then 100%/yr, with the
   gap to be filled primarily by **signal subscriptions, software licenses, and
   coaching** (~$225K/yr projected) — i.e. monetizing the appearance of an edge
   rather than an edge. This is the central red flag.

## What three independent 2026-10-09 measurements found

| Measurement | Method | Result | vs SPY |
|---|---|---|---|
| Walk-forward V2 (best config C) | project's own engine, 2yr daily, 6 windows, costs | **+12.25% total / 6.1%/yr**, PF 1.44, 50% WR | SPY ~+41% (window) / ~20%/yr → **trails ~3x** |
| Signal backfill, realistic entry | 601 distinct logged signals, enter next open, 5d hold, costs | +0.23R/trade, PF 1.69 (one flat month) | too-small sample; not conclusive |
| Signal backfill, logged entry | same but filling at stale `entry_price` | PF 3.4, +505R (ABSURD) | **artifact** — look-ahead bias |

The third row is a cautionary tale, documented so no future instance repeats
it: filling at the logged `entry_price` is look-ahead bias (for 73% of long
signals the next open is already above it, mean +9%). It manufactures a fake
+505% month. Correcting to next-open entry collapses it to a thin, one-month,
single-regime positive expectancy that the 2-year walk-forward does not sustain.

## Root causes

- **Signal-limited, not capital-limited.** The filters are so aggressive that
  there are too few high-quality trades to compound meaningfully, and the
  per-trade edge is too thin (PF ~1.4) to overcome costs and beat the index.
- **No feedback loop.** `SignalMetrics.record_outcome()` exists but is never
  called (zero callers). The system could not learn from its own results
  because it never recorded them. 1,986 signals, 2 outcomes ever.
- **Degenerate models presented as capability.** The regime detector and exit
  predictor are near-random but are wired into the pipeline and the dashboards.
- **Goal substitution.** The documented plan to "make money" quietly shifted
  from trading profits to selling signals. That is the tell that the edge was
  never there.

## Bottom line for the operator

The mission is positive P&L net of costs, beating SPY buy-and-hold. On all
available evidence the bot does not meet that bar and never has. Buy-and-hold
of SPY would have roughly tripled the best configuration's return over the same
period with far less complexity and risk. Any future work must (a) first make
the system measure its own outcomes honestly, and (b) demonstrate an
out-of-sample edge over SPY *before* adding any further complexity. If
successive honest tests keep failing, the correct recommendation is to wind the
active-trading approach down in favor of buy-and-hold.
