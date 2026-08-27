# POST-MORTEM: The RRS Strategy and Why the Bot Is Where It Is

> Reconstructed 2026-08-27 by the first operator instance from repository
> evidence (code, committed signal data, and the project's own strategy docs).
> Earlier history before this checkout is inferred, not witnessed; where that is
> the case it is marked "(inferred)". Correct this file as better evidence
> appears.

## The one-paragraph version

This bot implements the r/RealDayTrading "Real Relative Strength" (RRS)
momentum strategy with an elaborate stack of filters (SPY gate, SMA gate, VWAP,
multi-timeframe alignment, VIX regime, sector RS, intermarket analysis) and an
ML advisory layer. Despite the machinery, **there is no evidence it has a
tradable edge.** Its own strategy documents concede a *negative Kelly
criterion*. Its headline backtest returns (~3–7% annualized) were computed
**gross of transaction costs and were never compared to SPY buy-and-hold** —
and over the backtested period SPY buy-and-hold returned far more. Live, the
system has recorded only **2 trade outcomes ever** (1 win, 1 loss) and appears
to have **stopped running around 2026-03-05**.

## The evidence (as of 2026-08-27)

### 1. The strategy's own math says the edge is ~zero
From `ACTIONABLE_100X_STRATEGY.md` and `WEALTH_STRATEGY_100X.md` (committed to
the repo):
- Win rate: **37–38%**
- Profit factor: **~1.29**
- Kelly criterion: **-0.02 (NEGATIVE)** — by the docs' own calculation.
  A negative Kelly means increasing size increases risk of ruin without
  improving expectancy; the edge is marginal at best.
- "Best backtest": **6.8% annual** (~$1,700 on $25K).

A negative-Kelly, sub-40%-win-rate, 1.29-profit-factor system is
indistinguishable from "no durable edge after costs."

### 2. The headline backtests were not honest about costs or the benchmark
- `scripts/run_walkforward_v2.py` — the script `CLAUDE.md` cites for its
  flagship "$1,716 / 6.9% over 2 years" result — had **zero** commission or
  slippage modeling. Those numbers were **gross**.
- `scripts/run_backtest.py` defines a 0.1% slippage constant but, by its own
  comment, **does not apply it to fills** — it only mentions the impact in a
  footnote.
- **No backtest computed SPY buy-and-hold.** The mandate's actual success
  criterion had never been measured in code until 2026-08-27.
- Context: Feb 2024 → Nov 2025 (the walk-forward window) was a strong bull
  market; SPY buy-and-hold materially outperformed the strategy's own gross
  numbers. So the strategy very likely **underperformed a do-nothing SPY hold,
  before costs, and worse after.**

### 3. There is essentially no live track record
From `data/signals/signal_metrics.json` (committed):
- `total_scans`: 880
- `total_signals`: 120  (`total_long_signals`: 1, `total_short_signals`: 119 —
  a bizarre near-total short skew inconsistent with the long-heavy
  `signal_history.json`; the metrics accounting is itself suspect)
- `total_outcomes`: **2**  → `target_hits`: 1, `stop_outs`: 1
- `last_scan_at`: **2026-03-05** (system appears dormant since)

Two outcomes is not a track record. It is noise. No profitability claim —
positive or negative — can rest on it.

### 4. The "100X" / "WEALTH" documents are a warning sign
`ACTIONABLE_100X_STRATEGY.md`, `WEALTH_STRATEGY_100X.md`,
`WEALTH_OPTIMIZATION.md`, `QUICK_START_100X.md` lay out a plan to turn ~7% into
100% annual returns. Their own honest half admits the trading edge cannot do
this, so the plan leans on **leverage** and **selling signal-service
subscriptions** — i.e. revenue that is not trading edge. Chasing a 14.6x
improvement on a negative-Kelly strategy is the trap the operator must not fall
into.

## Why the machinery didn't help
The filter stack (SPY/SMA/VWAP/MTF/VIX/sector/intermarket) reduces trade count
by ~98% and modestly improves gross win rate, but "fewer, slightly better
trades" on a strategy with no real edge still nets to approximately nothing
after costs — and to a loss against a rising SPY. Adding more advisory layers
(the Murphy intermarket work, the 87-feature ML) has not changed this; the
project's own notes concede the ML is "advisory-only" and the exit predictor is
"barely above random."

## What "fixed" would actually look like
A variant of this strategy (or a different strategy entirely) that, on an
out-of-sample walk-forward **net of `TransactionCostModel` costs**, produces a
positive return that **exceeds SPY buy-and-hold over the identical window** —
and does so across multiple windows, not one cherry-picked quarter. Until a
change clears that bar, it is not an improvement, however good its in-sample
metric looks.

## Standing recommendation
Treat the null hypothesis — "this strategy has no edge that beats holding SPY" —
as the default. Every session should either (a) produce evidence that clears
the bar in Section "What fixed would look like," or (b) add to the pile of
evidence that it cannot, moving toward the wind-down recommendation the mandate
authorizes.
