# OPERATOR MANDATE

> **STATUS: RECONSTRUCTED v0 (2026-10-28).** The original `MANDATE.md` referenced by the
> scheduled operator task did **not exist** in the repository on any branch or in git history
> when operator run #001 started. This file was reconstructed by that run from (a) the safety
> reminders and mission statement embedded in the scheduled task prompt and (b) evidence found
> in the codebase. **It is not authoritative until a human reviews and ratifies it.** If you
> are a human: please replace this banner and confirm/edit the constraints below, or restore the
> real mandate if one exists elsewhere.

---

## Mission (verbatim from the scheduled task)

Make this bot **actually profitable**: positive P&L **net of honest costs**, **beating SPY
buy-and-hold**. Not "optimize metrics." Not "follow a methodology" for its own sake. If the
evidence keeps saying no strategy works, **say so in the journal and recommend escalation or
wind-down.**

Philosophy roots: r/RealDayTrading methodology (Real Relative Strength, "market first").

## Hard constraints (non-negotiable)

1. **PAPER TRADING ONLY.** Never set `AUTO_TRADE=true`. Never set `PAPER_TRADING=false`.
2. **Never modify live broker credentials** or anything that could place a real-money order.
3. **Never touch the `risk/` directory** without explicitly flagging it in the journal entry.
4. **Every session ends with a committed journal entry** in `data/operator_journal/entries/`
   and an updated `data/operator_journal/LATEST.md`.
5. **Work on a dated branch** `operator/YYYY-MM-DD`. **Never merge to `main`** — the human
   reviews and merges. (NOTE: the session harness assigned a different default branch,
   `claude/adoring-feynman-1ttmok`; see run #001 entry. Human: please reconcile.)
6. **Be honest.** Report outcomes faithfully. No metric-gaming, no cherry-picked windows, no
   claiming an edge that the data does not support. The benchmark is SPY buy-and-hold over the
   **same** period, net of costs.

## What "profitable" means here (the scorecard)

A strategy variant is only worth pursuing if, over an honest out-of-sample window and net of
commissions + slippage, it clears **all** of:

- Positive total P&L, **and**
- **Beats SPY buy-and-hold total return over the identical window**, **and**
- Does so with a risk-adjusted profile that isn't pure luck (meaningful sample size, PF
  comfortably > 1, non-negative Kelly, drawdown the account can survive).

If it does not beat SPY buy-and-hold, it is not a win — regardless of how good the internal
metrics look.

## Protocol (per run)

1. **Read** (in order, fully): this MANDATE, `LATEST.md`, `POST_MORTEM_RRS.md`, `CLAUDE.md`,
   and the 3 most recent `entries/`.
2. **Assess state.** What is the current honest P&L evidence? Is there a real (paper) track
   record, or only backtests? Is the live bot even running? What did the last run conclude and
   did it change the picture?
3. **Decide.** Pick the single highest-value, evidence-grounded action. Prefer measurement and
   falsification over adding speculative complexity. Do not ship changes that merely *look* like
   improvement without out-of-sample evidence. Chasing returns by raising risk on a strategy
   with no proven edge is forbidden (it increases risk of ruin, not return).
4. **Execute.** Research / code / test. Keep commits focused and reviewable.
5. **Verify.** Prove any claimed improvement with a reproducible, honest, out-of-sample test
   compared to SPY buy-and-hold over the same window. Byte-compile touched Python.
6. **Journal.** Write a dated entry: what you assessed, what you did, what the evidence showed,
   what you recommend next. Update `LATEST.md`. Commit and push the dated branch.

## Escalation / wind-down triggers

Recommend **escalation to the human** (or **wind-down**) when any of these hold across runs:

- Best honest backtest/paper result materially underperforms SPY buy-and-hold, and no tested
  change closes the gap.
- Kelly fraction is ~0 or negative (no edge) and remains so after honest attempts.
- The project pivots away from "make the trading strategy profitable" toward monetizing other
  people (e.g., selling signals) as the primary path to the P&L goal — this is an admission the
  strategy itself has no edge and should be named as such.
