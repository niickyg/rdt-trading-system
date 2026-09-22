# Post-Mortem: The RRS Strategy

> Reconstructed 2026-09-22 by the first operator instance. `POST_MORTEM_RRS.md`
> was referenced by the mandate but did not exist in the repo; this is built
> from the repo's own documents plus a fresh, reproducible backtest. Future
> operators should append, not rewrite.

## The one-paragraph version

The bot's core strategy is Real Relative Strength (RRS) momentum:
`RRS = (stock %chg − SPY %chg) / ATR`, go long strong / short weak, exit on
ATR-based stops and targets, gated by a stack of "RDT methodology" filters
(SPY regime, 50/200 SMA, VWAP, multi-timeframe, VIX, sector RS, regime-adaptive
thresholds, intermarket). After extensive parameter optimization the **best
documented result is ~6.8% annual (gross)**. The strategy's own author notes
its **Kelly criterion is ≈ −0.02 (negative)** — the edge is marginal at best.
A fresh out-of-sample walk-forward (below) returns **~3.7% annualized for the
production ("filtered") config vs ~20% for SPY buy-and-hold over the same
window**, and shows the RDT filters **actively hurting** returns. Net of
transaction costs (which the backtest ignores entirely) the edge is not
demonstrable.

## Timeline (from repo artifacts)

- **Initial build:** RRS scanner + agent architecture + ML ensemble + options
  module + full SaaS dashboard. Large, sophisticated codebase.
- **Optimization journey** (`DEPLOYMENT_SUMMARY.md`): 180 parameter combos →
  "optimal" RRS 1.75 / stop 0.75×ATR / target 1.5×ATR; return 2.85% → 6.80%.
  Enhanced exits (trailing/scaled/breakeven) → 6.92%. Risk-profile testing
  concluded returns **do not scale with risk** because the strategy is
  "signal-limited, not capital-limited."
- **"100X" plans** (`ACTIONABLE_100X_STRATEGY.md`, `WEALTH_STRATEGY_100X.md`):
  concede trading alone cannot hit the growth target and pivot ~half the goal
  to **selling a signal service** — i.e. revenue, not trading edge.
- **CLAUDE.md results table** claims the RDT filters *improve* a 2-year
  walk-forward from $815 (baseline) to $1,716 (filtered). **This did not
  replicate** on fresh data (see below).
- **Live/paper outcome tracking is essentially empty:** `signal_metrics.json`
  records 120 signals over 880 scans but only **2 tracked outcomes** (1 win,
  1 loss). There is no live P&L evidence of profitability. Data is stale
  (last scan 2026-03-05).

## Fresh evidence (2026-09-22, reproducible)

Command: `python scripts/run_walkforward.py` (30-stock core watchlist + SPY +
VIX + 11 sector ETFs; 3 quarterly walk-forward windows, ~201 trading days,
2025-07-24 → 2026-09-21). Full log:
`data/operator_journal/evidence/2026-09-22-walkforward-v1.txt`.

| Metric (aggregate)     | Baseline (no filters) | Filtered (RDT config) | SPY buy&hold |
|------------------------|-----------------------|-----------------------|--------------|
| Total return           | +4.75%                | **+2.97%**            | **+23.58%**  |
| Annualized (est.)      | ~6.0%                 | **~3.7%**             | **~20%**     |
| Win rate               | 54.1%                 | 45.1%                 | —            |
| Profit factor          | 1.54                  | 1.17                  | —            |
| Total trades           | 85                    | 173                   | 1            |
| Transaction costs      | **none modeled**      | **none modeled**      | negligible   |

**Two findings that matter more than the headline gap:**

1. **The RDT filters hurt.** Filtered underperformed baseline by **−$446
   (−1.78% of capital)** and cut win rate by 9 points. The central thesis of
   CLAUDE.md — that the RDT "market-first" filter stack adds edge — is
   contradicted by fresh out-of-sample data. It filtered out 96.8% of signals
   to produce a *worse* result.
2. **The backtest ignores all costs.** `backtesting/engine.py` computes
   `pnl = (exit − entry) × shares` with no commission, no slippage, and assumes
   fills exactly at the target/stop price. The ~3.7% is an optimistic ceiling;
   173 round-trips of real-world slippage would erode it materially, plausibly
   to zero or negative.

## Why the strategy structurally struggles

- A long/flat (mostly-cash-between-signals) momentum strategy taking small
  ATR-sized profits **cannot keep up with a strong bull market** — SPY did the
  compounding while the bot sat in cash 96% of the time.
- Negative-Kelly edge means position sizing can't rescue it; more size just
  adds variance.
- The filter stack adds complexity and turnover without adding demonstrable
  edge on unseen data — a classic overfit-to-the-optimization-window signature.

## Bottom line

On the mandate's bar — **positive P&L net of honest costs, beating SPY
buy-and-hold** — the RRS strategy **does not clear it** on the best evidence
available today. The honest recommendation (see the 2026-09-22 journal entry)
is to stop adding features to this strategy and either (a) find a genuinely
different, cost-aware edge with a positive out-of-sample profit factor after
costs, or (b) wind the trading ambition down to SPY buy-and-hold and treat the
platform as software, not alpha.
