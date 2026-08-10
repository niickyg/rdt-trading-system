# LATEST — Operator Journal Pointer

**Most recent entry:** [`entries/2026-08-10-instance-001.md`](entries/2026-08-10-instance-001.md)

- **Instance:** #001 (first run)
- **Date:** 2026-08-10
- **Branch:** `operator/2026-08-10`

## TL;DR for the next instance

- No mandate/journal/post-mortem existed before this run. I bootstrapped them
  (`MANDATE.md`, `POST_MORTEM_RRS.md`, this journal). Read those first.
- **Live IBKR paper account = $5, zero positions.** The bot is unfunded / not
  trading. Not the $25K account the docs describe. Reconcile this.
- **The strategy loses to SPY buy-and-hold.** SPY +18.6%/yr vs bot's best
  documented backtest 3.4%/yr — and that backtest ignores trade costs.
- I added a **SPY buy-and-hold benchmark** to `scripts/run_walkforward_v2.py`
  so future evaluations confront opportunity cost. Compiles + math verified;
  full run pending a reliable data feed (yfinance is 429-throttled — use IBKR
  MCP `get_price_history`).

## Do this next (instance #002)

1. Add explicit commission + slippage to the backtest engine.
2. Wire the walk-forward to IBKR MCP data and run it (now prints the SPY bar).
3. If no config beats buy-and-hold net of costs → **recommend wind-down**, in
   writing, plainly. Do not add another filter/feature.
