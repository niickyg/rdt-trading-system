# LATEST — operator journal pointer

**Most recent entry:** [`entries/2026-11-17-cold-start-baseline-assessment.md`](entries/2026-11-17-cold-start-baseline-assessment.md)
**Date:** 2026-11-17 • **Instance:** first run (cold start) • **Branch:** `operator/2026-11-17`

## One-screen summary for the next instance

- **This was a cold start.** `MANDATE.md`, `LATEST.md`, `POST_MORTEM_RRS.md`, and
  `entries/` did **not exist** in git. I reconstructed `MANDATE.md` (pending human
  ratification) and created the journal. If the human has authoritative originals,
  prefer those.

- **Honest verdict on the mission:** the RRS bot **does not beat SPY buy-and-hold**
  and is **likely unprofitable net of costs.**
  - Best backtest config: **+6.9% over ~21mo (gross, zero costs)**.
  - SPY buy-and-hold same window: **~+36% (~19%/yr)** → SPY wins ~5×, before costs.
  - Live/paper IBKR account: **$5, 0 positions, 0 trades/90d**; scanner dormant
    since **2026-03-05**. No live track record.

- **What I shipped (simulation/analysis only — no `risk/`, no live changes):**
  - `backtesting/costs.py` — `CostModel` (commission + slippage), zero-cost default.
  - `engine_enhanced.py` — costs applied at entry + every exit leg; net P&L; new
    `total_costs`.
  - `run_walkforward_v2.py` — costs on all 3 configs + **SPY buy-and-hold benchmark
    + excess-return verdict** in the aggregate output.
  - `tests/unit/test_costs.py` — 7 tests, all pass (no pandas needed).

- **NOT done / blocked:** could not run the full walk-forward here — **no pandas/numpy
  installed, no backtest cache, needs live yfinance.** Net-of-cost & vs-SPY numbers
  in the entry are **estimates**; the code to measure them is in place but **unrun**.

## Next instance should

1. Install deps and **run `python scripts/run_walkforward_v2.py`**; read the new
   "SPY BUY-AND-HOLD BENCHMARK" block — get the real net figures.
2. Treat the result as an **escalation decision** (per mandate): wind down / re-scope
   the RRS effort, or define a genuinely new, falsifiable edge hypothesis — not more
   filter-tuning on the same marginal core.
3. Clarify the $5 account, locate the real `POST_MORTEM_RRS.md`, ratify `MANDATE.md`.
4. Investigate the unexplained **119-short / 1-long** signal skew before trusting
   any live signals.
