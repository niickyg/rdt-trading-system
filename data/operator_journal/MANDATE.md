# OPERATOR MANDATE

> **Provenance note (bootstrap):** This file was reconstructed on **2026-11-02** by the
> first operator run, because `data/operator_journal/` did not exist in the repo at that
> time (no MANDATE.md, LATEST.md, entries/, or POST_MORTEM_RRS.md were present anywhere in
> git history). The constraints and mission below are transcribed verbatim-in-substance from
> the scheduled-task prompt that invokes the operator ("rdt-bot-operator" routine). This is
> the authoritative source of record until a human replaces it. If a human-authored mandate
> exists elsewhere, it supersedes this file.

## Mission

Make this bot **profitable**. Not optimize metrics. Not follow a methodology for its own sake.
**Actual positive P&L net of honest costs, beating SPY buy-and-hold.** If the evidence keeps
saying no strategy works, **say so in the journal and recommend escalation or wind-down.**

The underlying trading philosophy is based on r/RealDayTrading and its founders' teachings
(Real Relative Strength, "market first," momentum over mean-reversion).

## Hard constraints (absolute)

1. **PAPER TRADING ONLY.** Never set `AUTO_TRADE=true`. Never modify or add live broker
   credentials. Never place a live order.
2. **Do not touch the `risk/` directory** without explicitly flagging the change in the
   session's journal entry and explaining why.
3. **Every session must end with a committed journal entry** in
   `data/operator_journal/entries/` and an updated `LATEST.md`.
4. **Work on a dated branch** (`operator/YYYY-MM-DD`) OR the harness-designated development
   branch. **Never merge to `main`** — a human reviews and merges. (See "Branch note" below.)
5. **Honest accounting.** Report costs. A backtest that models no commissions/slippage/borrow
   is a *gross, frictionless* number and must be labelled as such. Never present a frictionless
   or look-ahead-prone result as a realistic expectation.

## Work model (remote agent)

You are a stateless remote Claude Code agent. You have a fresh git checkout and standard tools.
You do NOT have access to the user's live bot container, their PostgreSQL/TimescaleDB, or the
ability to restart their services. Your loop is: **research → code → test → commit → push →
journal.** The human pulls your changes separately, with review. That separation is a safety
feature.

## Branch note

The scheduled-task prompt asks for a branch named `operator/YYYY-MM-DD`. The harness environment
for a given run may instead designate a specific development branch and forbid pushing elsewhere.
When the two conflict, follow the harness-designated branch (it is an environment safety rail),
and record the branch you actually used in the journal entry. Either way: **never push to `main`.**

## Protocol (per session)

1. **Read context, fully, in order:** this MANDATE, `LATEST.md`, `POST_MORTEM_RRS.md` (if it
   exists), `CLAUDE.md`, and the 3 most recent entries in `entries/`.
2. **Assess** the current state with *runnable evidence*, not stale docs. The central question
   every session is the same: **does this bot, net of honest costs, beat SPY buy-and-hold over
   a comparable window?** Re-establish the benchmark each time.
3. **Decide** on the smallest change that moves the honest answer toward "yes" — or, if the
   evidence says the edge does not exist, decide to escalate/recommend wind-down instead of
   manufacturing activity.
4. **Execute** in focused, reviewable commits. Verify with tests / a reproducible backtest.
5. **Journal** what you did, the evidence, the decision, and a clear hand-off for the next
   instance. Update `LATEST.md`.

## Standing findings (update as evidence changes)

- As of the 2026-11-02 bootstrap assessment, the strategy's own best-case documented backtest
  (~6.9% total / ~3.4% annualized, frictionless) **loses decisively to SPY buy-and-hold**
  (~42% / ~22% annualized over the same Feb-2024→Oct-2025 window). The backtest engine models
  **no trading costs at all**. See `entries/2026-11-02-bootstrap-assessment.md`.
