# POST-MORTEM: RRS Strategy & the State of the Bot

> **Provenance:** Bootstrapped by the autonomous operator, Run 001 (2026-08-07).
> The scheduled operator prompt referenced this file as existing history; it did not
> exist in the repo, so this is the first honest reconstruction from the artifacts that
> *do* exist. Future runs should append, not rewrite, and correct anything shown wrong
> by new evidence.

## What this bot is

An autonomous trading system built around **Real Relative Strength (RRS)** momentum, per
the r/RealDayTrading "market first" methodology: only take momentum trades that are strong
relative to SPY, gated by a stack of filters (SPY regime, 50/200 SMA, VWAP, multi-timeframe
alignment, VIX, sector RS, intermarket). Extensive infrastructure: agents, options module,
IBKR integration, ML ensemble, web dashboard, backtest engines.

## The uncomfortable truth (as of 2026-08-07)

The engineering surface is large and polished. The **evidence of a profitable edge is
almost nonexistent.** Concretely, from the artifacts in the repo:

### 1. One real backtest, and it is mediocre
- Sole committed result: `data/optimization/optimization_2025-12-29.json`, a 180-combo grid
  search on $25k over a trailing ~365 days.
- Best config: **~6.8% total return, 38.1% win rate, profit factor 1.29, Sharpe 0.11,
  max drawdown 2.4%, 215 trades.**
- Sharpe 0.11 is, for practical purposes, **no risk-adjusted edge.**

### 2. No honest cost accounting
- No commissions modeled anywhere in the backtest engines.
- `run_backtest.py` defined `SLIPPAGE_PCT = 0.001` but **never applied it to prices** — it
  only subtracted an estimate in the printed report. With PF 1.29 on 215 trades, realistic
  round-trip costs plausibly erase the entire 6.8%.

### 3. No benchmark — the central omission
- The whole premise is "beat the market," yet **SPY buy-and-hold was never computed.** SPY
  data is loaded only to calculate relative strength. There was no place in the code that
  asked "did we beat just holding SPY?" (Run 001 added this to `run_backtest.py`.)
- For reference, SPY buy-and-hold has historically returned ~10%/yr — i.e. the best
  backtested config (6.8%, before costs) likely **underperformed passive SPY.**

### 4. Essentially no live track record
- `data/signals/signal_history.json`: ~1,986 generated signals (2026-02-03 → 2026-03-05,
  ~1 month) with **`outcome = None` on every single one.**
- `data/signals/signal_metrics.json`: 880 scans, 120 signals, but only **2 outcomes ever
  recorded (1 win, 1 loss).** That is the entire real-world result set: two trades.
- The two signal files even disagree (metrics says 119/120 short; history is 85% long),
  pointing to an outcome-tracking pipeline that isn't reliably running.

### 5. ML adds no demonstrated edge
- `models/ensemble/metrics.json`: cross-validated AUC **0.54** (coin-flip is 0.50), with
  train AUC 0.99 — textbook overfitting. Per CLAUDE.md the ML layer is already treated as
  advisory-only; the data supports that demotion.

### 6. The "100X" documents are aspiration, not results
- `DEPLOYMENT_SUMMARY.md` claims **50.2% win rate / PF 1.35**; the actual data file shows
  **38.1% / 1.29**. That discrepancy is unsupported by any artifact and should be treated as
  marketing, not measurement.
- To their partial credit, `ACTIONABLE_100X_STRATEGY.md` and `WEALTH_STRATEGY_100X.md`
  **admit** trading alone caps near ~7% and that Kelly sizing comes out **negative** — then
  bridge to "100X" entirely via revenue streams (signal subscriptions, options income,
  crypto, futures) that **do not exist** in code or customers.

## Why it's in this state (root causes)

1. **Build-first, measure-later.** Enormous feature breadth (options, agents, dashboards)
   was built before establishing whether the core signal beats a passive benchmark.
2. **Metric theater.** Win rate / profit factor were reported without the two things that
   determine whether a strategy is worth running: **transaction costs** and a **benchmark.**
3. **Broken feedback loop.** Signal outcomes weren't being tracked, so the system never
   learned from its own (paper) results.
4. **Narrative pull.** "100X" framing set a target that the honest numbers can't support,
   creating pressure toward flattering, inconsistent figures.

## The bar going forward

A change is only progress if it moves the **cost-adjusted, SPY-relative** number in a way
that survives out-of-sample testing. Everything else is motion, not progress.

## Open questions for future runs
- Does *any* configuration beat SPY buy-and-hold net of realistic costs, out-of-sample?
  (Run the updated `run_backtest.py` / a walk-forward with a benchmark and a real cost model.)
- Why is outcome tracking not persisting results? Fixing the feedback loop may matter more
  than any parameter change.
- If, after honest benchmarking across regimes, no config beats SPY: recommend wind-down of
  active trading and repurpose the infra (e.g. research-only), per the mandate.
