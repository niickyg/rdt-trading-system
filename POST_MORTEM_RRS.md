# POST-MORTEM: The RRS Strategy — Why The Bot Is Where It Is

> **Provenance note.** The scheduled operator prompt referenced this file as
> pre-existing "history of why the bot is in its current state," but no such file
> existed in the repo (working tree, history, or any branch) as of 2026-09-29.
> This document is therefore an **evidence-based post-mortem written from scratch**
> on 2026-09-29 using only artifacts actually present in the repository. Every
> claim below is sourced to a file. Nothing here is invented. Future instances:
> extend this with real tracked-outcome data as it accumulates.

---

## TL;DR

The core Real Relative Strength (RRS) momentum strategy, as implemented and as
documented **by its own strategy files**, has at best a *marginal* edge and, on
the honest numbers, **underperforms simply buying and holding SPY**. There is
almost no real tracked-outcome data to contradict or confirm this — the system
has recorded only **2 trade outcomes ever**. The repository contains aggressive
"100X" growth documents whose own math admits the edge is near zero.

## The evidence in the repo

### 1. The strategy's own math says the edge is marginal-to-negative
From `ACTIONABLE_100X_STRATEGY.md`:
- Best backtest: **6.8% annual (~$1,700 profit)** on $25K.
- Profit factor **1.29**, win rate **38%**.
- Kelly Criterion computed in that same doc: **≈ −0.02 (negative)** — i.e. the
  edge is so thin that increasing size raises risk of ruin without raising return.
- The doc's own "solution" to hitting 100% returns is **not better trading** — it
  is to make ~half the target from a **signal-service subscription business**.
  That is a tell: when the plan to be profitable relies on selling the signals
  rather than trading them, the trading edge is not there.

### 2. The walk-forward backtest underperforms buy-and-hold
From `CLAUDE.md` (2-year walk-forward, $25K, Feb 2024 – Nov 2025):

| Config | Total Return | Annualized |
|--------|-------------|-----------|
| Baseline | $815 (3.3%) | 1.6% |
| Old Filters | $1,267 (5.1%) | 2.5% |
| **RDT Filters (best)** | **$1,716 (6.9%)** | **3.4%** |

Over that *same* window, SPY buy-and-hold returned **+34.5%** (price return;
~+37% with dividends). This is now verified via the sanctioned IBKR data source
(connector `get_price_history`, SPY US STK contract_id `756733`, monthly bars):
SPY closed **$508.08 at end of Feb 2024** and **$683.39 at end of Nov 2025**,
i.e. **+34.5%**. The best configuration of the bot returned **6.9% cumulative
over the same ~21 months** — buy-and-hold beat the bot's *best* configuration by
roughly **5x**. "RDT filters take fewer but higher-quality trades" and "98% of
raw signals are filtered out" — the filters improve the bot's *relative* ranking
(baseline 3.3% → RDT 6.9%) but do not come close to the only bar that matters.

| | Bot best (RDT filters) | SPY buy-and-hold |
|---|---|---|
| Feb 2024 → Nov 2025 | +6.9% | **+34.5%** |
| Annualized | ~3.4% | ~18% |

### 3. There is essentially no real outcome data
From `data/signals/signal_metrics.json`:
- `total_scans: 880`, `total_signals: 120`, but **`total_outcomes: 2`**
  (`target_hits: 1`, `stop_outs: 1`).
- So every profitability claim rests on **backtests**, not on realized fills.
  Backtests here assume fills; real net-of-cost trading has not been demonstrated.

### 4. A structural red flag: near-total short skew
Also from `signal_metrics.json`:
- `total_long_signals: 1` vs `total_short_signals: 119`.
- The scanner produced almost exclusively **short** signals across 880 scans in a
  period when the market rose. A momentum system that is systematically short in
  an uptrend is positioned to bleed. Whether this is a scanner bias, a data
  window artifact, or a genuine regime call, it needs root-causing before any
  size is put behind it.

## Why the bot is in its current state

1. **The edge was never robustly positive net of costs.** The strongest honest
   number in the repo (6.9% over 2y) loses to SPY, and the strategy's own Kelly
   figure is negative.
2. **Effort went into breadth, not proof of edge.** The codebase is large and
   feature-rich (options module, ML ensemble, 87 features, dashboards, multi-
   broker failover, SaaS overhaul). Very little of it is validated against the
   one metric that matters: net-of-cost P&L vs SPY.
3. **Real trading was never run long enough to generate outcome data** (2 total).
4. **Growth documents set a 100%/yr target** that the underlying edge cannot
   support, pushing toward leverage and a subscription-revenue workaround rather
   than toward an honest reassessment.

## What would change the conclusion (the bar for "it works")

- A walk-forward or paper run with **real fills and honest costs** (commission +
  spread + slippage) that shows **positive net P&L AND beats SPY over the same
  window**, out-of-sample, across more than one regime.
- Until that exists, the honest status is: **no demonstrated profitable,
  SPY-beating edge.**

## Recommended posture

Treat the current strategy as **unproven / likely negative-edge**. Do not add
features. Spend operator sessions **trying to falsify or salvage the edge cheaply**
(e.g., re-run walk-forward with realistic costs, root-cause the short skew, test
whether *any* simple variant beats SPY out-of-sample). If several honest attempts
still lose to buy-and-hold, **recommend wind-down** of active trading in favor of
holding SPY, per the mandate.
