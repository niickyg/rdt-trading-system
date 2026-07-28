# LATEST

Most recent operator session: **2026-07-28 (run #1, cold start)**

→ `entries/2026-07-28-cold-start.md`

## One-line state

Bot is **dormant** (paper account $5, 0 trades/90d, signals stopped 2026-03-05);
**no measured edge over SPY** (+7.19% over the window); this run bootstrapped the
missing operator infrastructure and added a verified **signal-outcome backtest
harness** (`scripts/signal_outcome_backtest.py`). No strategy/risk changes.

## Next run, start here

1. Ask the human: which paper account is real — the $5 MCP one or the documented
   $25K DUP995654? (Scoreboard depends on it.)
2. Determine why the scanner stopped on 2026-03-05 (likely needs the human /
   their infra).
3. Run the backtest harness on a real, broad prices file (yfinance locally or
   IBKR MCP `get_price_history`) for the first cost-net "does RRS beat SPY?" answer.
4. Wire per-signal outcome tracking into the live loop (kill the 2-outcomes problem).

Read `MANDATE.md` in full before acting.
