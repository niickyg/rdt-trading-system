# LATEST — pointer to the most recent operator session

**Most recent entry:** [`entries/2026-07-23-instance-01.md`](entries/2026-07-23-instance-01.md)
**Date:** 2026-07-23 · **Instance:** 01 · **Branch:** `operator/2026-07-23`

## One-paragraph summary

First instance to reach committed version control. The operator-journal infrastructure
(MANDATE, LATEST, POST_MORTEM_RRS, entries/) **did not exist and was never committed** — I
bootstrapped it. Ground-truth assessment: the connected IBKR paper account holds **$5** with
0 positions (CLAUDE.md claims a $25K account — discrepancy); only **2 trade outcomes** were
ever tracked; the system has been **dormant since 2026-03-05**; and the strategy's own best
*in-sample* backtest (~3.4% annualized) **loses ~5x to SPY buy-and-hold** (~16.5% annualized,
measured from IBKR). **Verdict: no evidence of edge.** I made **no** trading/risk/config
changes and enabled no live trading.

## The single decision blocking all progress
Point the operator at a **real funded paper account**. Until then every instance is blind.
Then run a forward paper test with honest costs and auto outcome-tracking to ≥100 closed
trades before any further strategy work. If it underperforms SPY (likely), wind down.

## Do-not-repeat / standing flags
- MANDATE.md is a **reconstructed bootstrap** — verify/replace with the authoritative one.
- Repo backtest docs are **internally inconsistent** (49.5% vs 38% win rate) — do not trust
  prose; trust measured out-of-sample P&L only.
- `risk/` untouched. No live trading. Paper only.
