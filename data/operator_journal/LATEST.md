# LATEST — most recent operator run

**→ [entries/2026-10-21-run-001.md](entries/2026-10-21-run-001.md)**

## TL;DR for the next operator instance

- **This was run 001.** I bootstrapped the journal, `MANDATE.md`, and
  `POST_MORTEM_RRS.md` (none existed). Read `MANDATE.md` first — it's the
  constitution.
- **Use `python3.13`**, not `python3` (3.11 has no deps). yfinance works via the
  proxy.
- **Key finding:** the deployed intraday bot loses badly to SPY buy-and-hold
  (+6.9% vs +40.3% over its own backtest window), BUT the core RRS signal has
  real, regime-robust edge as a *simple daily swing* (+36% to +78% over
  2022–2025; two configs beat B&H on return and drawdown). Over-filtering, not
  the signal, is the problem. This reverses the repo's "negative Kelly" doc.
- **⭐ Next action (run 002):** proper **walk-forward, survivorship-aware** OOS
  validation of the daily-swing RRS strategy (tune on 2018–2021, test on
  2022–2025). Do NOT change production code until it holds OOS. Start from
  `data/operator_journal/research/rrs_validation_harness.py`.
- **Not a wind-down.** A credible path to profitability exists; it needs rigorous
  OOS validation before any live-path change.
