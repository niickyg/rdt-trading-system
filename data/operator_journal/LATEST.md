# LATEST — most recent operator session

**Pointer to the latest journal entry. Every session updates this.**

## → [2026-09-17 — Bootstrap + honest baseline vs SPY](entries/2026-09-17-bootstrap-and-honest-baseline.md)

**Instance #1 (first run).** Bootstrapped the operator journal (this file, MANDATE,
POST_MORTEM) which did not previously exist. Established the honest baseline with a
fresh backtest.

### TL;DR for the next instance

- **The bot does NOT meet its mandate.** Best strategy config returns +7.4% GROSS
  over ~2 years while **SPY buy-and-hold returned +47.4%** over the same window.
  Strategy loses to the benchmark by ~$10,000 even at **zero** transaction cost.
- **No backtest engine models any cost.** At ~$10/round-trip the strategy is
  outright **negative** (-3.85%). Confirmed via new tool `scripts/benchmark_vs_spy.py`.
- **The core RRS signal is weak** (unfiltered baseline only 2.9%/yr; project's own
  Kelly ≈ −0.02). Filter-tuning is not the lever; it's been exhausted.
- **Do not drift into revenue theater** (SaaS/signal-service). Selling a
  negative-edge signal sells a losing product. Prior effort drifted here.

### Next instance: test ONE structurally-different hypothesis (not more tuning)

1. SPY-trend participation vs RRS stock-picking (does dumb trend-following beat it?).
2. Does raw RRS have any forward IC? (`research/factor_tester.py`) — if ~0, wind down.
3. Long-only in uptrends (is the short side a drag in a bull market?).

Run `python scripts/benchmark_vs_spy.py` — it shows strategy-vs-SPY net of costs in
one command. Read `MANDATE.md` and `POST_MORTEM_RRS.md` first.

**Escalation:** if #2–#3 also fail, recommend wind-down. Paper-only until then.
