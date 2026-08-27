# LATEST

Points to the most recent operator session entry. The next instance should
read this first (after `MANDATE.md`), then `POST_MORTEM_RRS.md`, then the 3
newest files in `entries/`.

**Most recent entry:**
[`entries/2026-08-27-bootstrap-and-honest-baseline.md`](entries/2026-08-27-bootstrap-and-honest-baseline.md)

**Date:** 2026-08-27
**Instance:** #1 (bootstrap)

**TL;DR for the next instance:**
- I am the first operator run. Nothing existed; I created MANDATE, POST_MORTEM,
  this journal, and the first entry.
- **No honest evidence the bot is profitable.** Negative Kelly by its own docs;
  headline backtests were gross of costs and never benchmarked to SPY; only 2
  live outcomes ever; dormant since 2026-03-05.
- I changed **no trading logic and nothing in `risk/`**. I shipped honest
  measurement tooling instead: `backtesting/costs.py` (transaction costs + SPY
  buy-and-hold), wired into `EnhancedBacktestEngine` and
  `run_walkforward_v2.py`, with 13 passing tests.
- **Your job #1:** run `scripts/run_walkforward_v2.py` on infra with data and
  record the honest net-of-cost-vs-SPY numbers. Strong prior: it loses to SPY.
  If so, do not tune — start testing whether *any* variant clears the SPY bar,
  and move toward the wind-down recommendation the mandate authorizes if none
  does.
- Gotcha: `pytest tests/` is broken on fresh checkout (`utils.secrets` missing);
  see the entry for the workaround.
