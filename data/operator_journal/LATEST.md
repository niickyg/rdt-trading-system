# LATEST

**Most recent entry:** [`entries/2026-08-21-run-001.md`](entries/2026-08-21-run-001.md)
**Date:** 2026-08-21 · **Run:** 001 (journal bootstrap)

## One-line state

The strategy does **not** beat SPY buy-and-hold — best documented backtest is
~3.4% annualized (gross, **no cost model**) vs SPY's ~16.6% annualized (verified
via IBKR); strategy's own math shows a **negative Kelly**; real track record is
**2 closed trades**. Verdict: honest baseline is "hold SPY." Run 001 added a
trading-cost model to the backtest engine; rough estimate says the best config is
~breakeven-or-negative net of costs.

## Next instance: check first

1. Has an authoritative `MANDATE.md` / `POST_MORTEM_RRS.md` been restored from the
   user's machine? If so it supersedes the bootstrapped `MANDATE.md`.
2. Cost model: **DONE in `engine_enhanced.py`** (see `tests/test_backtest_costs.py`,
   all passing). Remaining: replicate in `backtesting/engine.py` and
   `engine_intraday.py`; optionally fold entry slippage into the effective fill
   price. Then **re-run the walk-forward net-of-costs** and gate on beating SPY.
3. yfinance is blocked by the egress proxy here; use IBKR MCP price tools for spot
   benchmarks. Bulk multi-symbol daily history needs a different provider or cache.

## Do NOT

- Tune parameters against the frictionless backtest (overfitting; violates MANDATE
  honesty bar).
- Build out the SaaS/signal-service path — that is not "making the bot profitable."
- Enable `AUTO_TRADE`, touch broker credentials, or modify `risk/` without a flag.
