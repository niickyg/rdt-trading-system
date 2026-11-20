# LATEST operator session

**Most recent entry:** [`entries/2026-11-20-session-01.md`](entries/2026-11-20-session-01.md)

- **Date:** 2026-11-20
- **Session:** 01
- **Branch:** `operator/2026-11-20` (based on `operator/2026-11-19`)

## One-paragraph summary for the next run

Read `MANDATE.md`, this file, `POST_MORTEM_RRS.md`, and the two entries
(`2026-11-20-session-01.md` first, then `2026-11-19-session-01.md`). Headline: the
previous run *estimated* the strategy loses to SPY; **this run MEASURED it.** I ran the
walk-forward end-to-end with the honest cost model ON, built a SPY buy-and-hold benchmark
into the tool, and tested a second regime (the 2022 bear market). Result, net of costs:
**no configuration beats SPY buy-and-hold in either a bull or a bear/flat market.** Best
case is a *tie* with SPY (raw baseline, flat market). In the bull window the best config
made +8.93% vs SPY's +37.80% (lost by 29 points). And the signature "RDT filters" are
*value-destroying* in the bear regime — they turned a +5.72% / PF-1.41 baseline into a
−1.95% / PF-0.87 loss, falsifying the thesis that they add "higher-quality" trades. Both
of the previous run's top tasks are now done and committed (`a494886`).

## Top tasks for the next run (in order)

1. **There is no more "make the backtest honest" work to do — it is honest now.** The
   measured verdict is in §3 of the entry. The next question is whether to continue at all.
2. If continuing: run **factor-level IC analysis** (`research/factor_tester.py`) to settle
   whether any raw signal in this repo has standalone predictive power. If the factors have
   no information coefficient, downstream filtering/ML/regime-tuning cannot create an edge —
   recommend wind-down without reservation.
3. Do **not** add features to the core or the SaaS layer. Do not re-tune filters against a
   bull-only window; the bear window shows that overfits.

## Open flags for the human (unchanged + new)

- **Make the call:** wind down active trading / keep only the raw baseline as a low-DD
  sleeve / treat as a pure paper research sandbox. Evidence for the decision is in the
  2026-11-20 entry §3 and §6.
- Ratify or correct `MANDATE.md` (still self-bootstrapped from the scheduled prompt).
- The operator journal lives ONLY on `operator/*` branches, never merged to `main`.
  Consider merging it so continuity isn't one lost branch away from gone.
- `CLAUDE.md`'s performance table is gross/unbenchmarked and misleading; I added an
  operator-correction note beside it. The real numbers are in the journal.
- Development has drifted toward a SaaS product layer on a core that loses to an index fund.
