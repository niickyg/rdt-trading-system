# LATEST — Operator Session Pointer

**Most recent entry:** [`entries/2026-09-08-bootstrap.md`](entries/2026-09-08-bootstrap.md)
**Date:** 2026-09-08 · **Instance:** first (cold start) · **Branch:** `claude/adoring-feynman-5g66ij`

---

## TL;DR for the next instance

1. **Read `MANDATE.md` first, fully.** It did not exist before today; I bootstrapped it. It is now your constitution.
2. **Mission status: NOT beating SPY.** Bot best-case backtest ~3.4%/yr vs SPY buy-and-hold ~18.7%/yr over the same window (real IBKR data). Strategy's own Kelly is negative. See `POST_MORTEM_RRS.md`.
3. **The bot barely measures itself:** only 2 tracked outcomes across 880 scans / ~1,986 signals. **This is the #1 problem to fix.**
4. **Environment gotcha:** `yfinance`/Yahoo is **blocked by egress policy** here — the repo's backtest scripts can't fetch data in the agent env. Only IBKR MCP `get_price_history` works for real data. Don't trust a backtest that silently got empty data.

## Your next objective (handed off)

> **Close the measurement gap:** make outcome/P&L tracking record what happens to
> every signal/trade and produce a realized, cost-adjusted equity curve to compare
> against SPY. Then, and only then, run the cold experiment: does *any* config beat
> SPY out-of-sample net of costs? If repeatedly no → invoke MANDATE §7 (escalate or
> wind down).

**Do NOT:** raise risk limits to chase returns, add more filters/ML layers, or
pivot to the signal-subscription SaaS — none of those address "does the bot have
an edge over the index." See the backlog in the 2026-09-08 entry.
