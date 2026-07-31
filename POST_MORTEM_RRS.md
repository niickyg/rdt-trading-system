# POST-MORTEM: RRS Strategy & Bot Profitability

> **Provenance / honesty note.** The operator's scheduled task references this file as
> "the history of why the bot is in its current state." **It did not exist** in the
> repository (no working tree, no branch, no git history) as of 2026-07-31. This
> document was **reconstructed by operator run-001** from first-hand evidence gathered
> that day. It is NOT the original narrative (there was none). It records what can be
> *verified*, and explicitly marks what cannot. Future runs should extend it with
> evidence, not invention. See `data/operator_journal/entries/2026-07-31-run-001.md`.

## The core problem, stated plainly

The bot is built around **Real Relative Strength (RRS)** momentum signals
(`RRS = (Stock %Δ − SPY %Δ) / ATR`) with layered RDT-methodology filter gates (SPY
gate, 50/200 SMA, VWAP, multi-timeframe), VIX/sector/regime overlays, an ML ensemble
(advisory only), and options execution. Despite this machinery:

- **Live account TWR since inception (Feb 2026): −61.5%.** (IBKR MCP
  `get_pa_performance_all_periods`, run-001.)
- **SPY buy-and-hold over the same window: positive** (+10% YTD, +22% 1-year).
- **The account is currently inert:** $5 net liquidation, zero open positions, zero
  trades in every queryable period, NAV flat all of July 2026.

## Why the strategy underperforms (evidence-based hypotheses)

1. **It was never measured against the right bar.** The walk-forward backtests
   (`scripts/run_walkforward*.py`) report absolute return and win rate but **never
   compute SPY buy-and-hold** for the same period (verified by grep — no
   `benchmark`/`buy-hold` logic exists). The "best" documented result, +6.9% over 2
   years (~3.4%/yr), *looks* like success in isolation but is far below SPY's ~20%/yr.
   Optimizing a metric that ignores the alternative of just holding the index is how a
   losing system looks like a winning one on paper.
2. **Heavy filtering, thin edge.** CLAUDE.md notes "98% of raw signals are filtered
   out," yielding few trades with a profit factor of ~1.24 in-sample. Small edges over
   few trades are fragile to costs (commission + slippage + spread) and to overfitting
   across the many tunable gates/overlays.
3. **Day-trading momentum is a hard, cost-heavy game.** The r/RealDayTrading premise
   (RRS + "market first") is a discretionary human framework. Encoding it as automated
   rules has not, on this evidence, produced an edge that survives costs — and it
   competes against the extremely low-cost, low-effort baseline of holding SPY.
4. **Docs drifted from reality.** The documented `research/` factor-analysis module is
   absent; the documented "$25K funded account" does not match the live $5 account.
   Decisions made against stale docs compound the problem.

## What is NOT yet known (do not assume)

- **Account identity.** Whether the IBKR-MCP account is the bot's production account or
  a separate/drained one is unconfirmed. The Apr–Jun NAV bleed ($521→$477.5) implies
  trades the trades-API didn't return, hinting at scoping/sub-account issues. The real
  production P&L could differ. **Confirming this is run-002's first job.**
- **Whether ANY config beats buy-and-hold out-of-sample.** Never tested, because the
  backtest has no benchmark and can't run offline in the operator environment (Yahoo
  Finance 429, no cache).

## Current recommendation (run-001)

Not yet a wind-down call, but close to one. Before any more strategy work:
1. Confirm which account is real and whether the bot is actually trading.
2. Make **excess return vs SPY buy-and-hold** the primary output of every backtest.
3. Give the backtest a reliable offline data path (IBKR MCP price history).
4. Then test honestly. If no configuration shows positive risk-adjusted excess return
   vs SPY on out-of-sample data, **recommend winding the bot down** and holding the
   index — and say so without hedging. Running complex, risk-bearing machinery to
   underperform a free benchmark is not a business; it is a liability.
