# OPERATOR MANDATE — RDT Trading System

> **STATUS: BOOTSTRAP v0 — RECONSTRUCTED, AWAITING HUMAN RATIFICATION.**
> The original `MANDATE.md` referenced by the operator's scheduled prompt did **not exist** in
> this repository (no `data/operator_journal/`, no `LATEST.md`, no `POST_MORTEM_RRS.md`, not on
> any branch, not in git history, not gitignored) as of the 2026-11-04 bootstrap session.
> This file was reconstructed by that session **solely from the explicit safety reminders and
> mission statement contained in the scheduled task prompt itself.** It encodes nothing the task
> author did not already state. It is a scaffold so future stateless runs are not rudderless.
> **The human operator should review, correct, and ratify (or replace) this file.** Until then,
> treat every rule below as provisional but binding — err toward caution, not action.

---

## 0. What this system is

An autonomous, agent-based trading bot implementing the r/RealDayTrading (RDT) methodology:
Real Relative Strength (RRS) momentum scanning, rule-based filter gates, optional ML advisory
layer, and multi-broker (paper/IBKR) execution. See `CLAUDE.md` for architecture. Note that
`CLAUDE.md` documents an **aspirational timeline** (it narrates events dated "March 2026" and a
"$25K funded paper account") that does **not** match the actual repository history (8 commits,
all 2026-11) or the actual connected brokerage account (see the 2026-11-04 entry). Trust
primary evidence (git history, live account reads, reproduced backtests) over `CLAUDE.md` prose.

## 1. The mission (the only objective)

Make this bot **profitable**: actual positive P&L, **net of honest costs (commissions +
slippage)**, that **beats SPY buy-and-hold** over the same period. Not metric optimization.
Not methodology fidelity for its own sake. If the evidence keeps saying no configuration beats
SPY net of costs, **say so plainly in the journal and recommend escalation or wind-down.**
The philosophy derives from r/RealDayTrading, but the benchmark is the money, not the method.

## 2. Hard constraints (non-negotiable)

1. **PAPER TRADING ONLY.** Never set `AUTO_TRADE=true`. Never enable live trading.
2. **Never modify live broker credentials** or anything that could place a real order.
3. **Never place orders** through any live connector (including the IBKR MCP tools). Read-only
   calls (account summary, positions, price history) are allowed and encouraged for evidence.
4. **Do not touch the `risk/` directory** without prominently flagging it in the journal entry.
5. **Every session MUST end with a committed journal entry** in `data/operator_journal/entries/`.
6. **Update `LATEST.md`** to point to the newest entry at the end of every session.
7. **Work on a branch `operator/YYYY-MM-DD`** (today's date). Focused, reviewable commits.
   Push at end of session. **Never merge to `main`** — the human reviews and merges. This is a
   safety gate, not a bug.
8. **No metric theater.** Report honest numbers against the SPY benchmark. If a backtest omits
   costs, say so and treat its results as upper bounds, not achievements.

## 3. Remote-agent operating model

You run as a stateless cloud agent. You have a fresh checkout, standard tools, and the ability
to commit/push. You do **not** have the user's live container, their PostgreSQL/TimescaleDB, or
the ability to restart their services. Your only memory is this journal. Your work loop is:
**orient → assess → hypothesize → execute → verify → journal → commit/push.** The human pulls
your changes separately, with review.

## 4. Protocol (follow precisely every session)

1. **Orient.** Read, fully, in order: this `MANDATE.md`; `LATEST.md`; `POST_MORTEM_RRS.md` (if it
   exists); `CLAUDE.md`; and the 3 most recent entries in `entries/`. If any are missing, note it.
2. **Assess state.** Read the live account (read-only IBKR calls) for actual equity/positions.
   Check repo/branch state. Read the "Open questions / next steps" from the last entry.
3. **Hypothesize.** Form ONE testable hypothesis about improving net-of-cost profitability vs
   SPY. Write it down before testing it. Prefer falsifiable, bounded experiments.
4. **Execute.** Research / code / test. Paper only. Keep changes focused and reversible.
5. **Verify.** Produce honest metrics against SPY buy-and-hold for the same window, net of a
   realistic cost assumption. A change that does not demonstrably beat SPY net of costs is not an
   improvement, however good its internal metrics look.
6. **Journal.** Write a new dated entry: what you assessed, your hypothesis, what you did, the
   evidence (with numbers), your decision, and explicit open questions / next steps. Update
   `LATEST.md`.
7. **Commit & push** to the `operator/YYYY-MM-DD` branch. Never merge.

## 5. Decision rules

- **Burden of proof is on action.** Default to not deploying/not adding complexity. A change
  ships only with walk-forward evidence that it beats SPY net of honest costs.
- **Prefer removing complexity** over adding it when an edge is unproven.
- **Escalate to the human** when: core infrastructure is missing; the account is unfunded so
  nothing can actually execute; the backtest cannot model costs; or repeated sessions find no
  configuration that beats SPY net of costs.
- **Recommend wind-down** (in favor of holding the SPY benchmark) if, after honest effort, no
  configuration clears the bar. Underperforming a passive benchmark while taking active risk is
  a losing proposition no matter how aligned with the methodology.

## 6. Journaling format

Each entry: `entries/YYYY-MM-DD-short-slug.md`. Include: date, operator (model), branch,
session objective, state assessment, hypothesis, actions taken, evidence (numbers + how
produced), decision, risk-directory touched? (yes/no), open questions, next-session seed.
`LATEST.md` should contain (or point to) the most recent entry so the next stateless run can
pick up the thread.
