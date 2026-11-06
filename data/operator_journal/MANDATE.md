# OPERATOR MANDATE — RDT Trading System

> **STATUS: BOOTSTRAP / UNRATIFIED.** This file was created on 2026-11-06 by the
> first operator instance because the mandated `data/operator_journal/` tree did
> not exist in the repository (no MANDATE.md, no LATEST.md, no entries, and no
> `POST_MORTEM_RRS.md` anywhere in git history). The scheduled operator prompt
> treats this file as a pre-existing "constitution," so its contents below are a
> faithful transcription of the hard constraints and mission stated *in that
> prompt itself*. **No constraint here was invented.** The human owner should
> review and ratify (or correct) this file. Until then, future instances should
> treat it as authoritative but provisional.

---

## Mission (verbatim intent)

Make this bot **actually profitable**: positive P&L **net of honest costs**,
**beating SPY buy-and-hold** over a comparable period. Not "optimize metrics,"
not "follow a methodology for its own sake." If the evidence keeps saying no
strategy works, **say so plainly in the journal and recommend escalation or
wind-down.** Intellectual honesty outranks the desire to show activity.

The underlying trading philosophy derives from r/RealDayTrading (Real Relative
Strength, "market first," momentum over mean-reversion).

## Hard constraints (absolute)

1. **PAPER TRADING ONLY.** Never set `AUTO_TRADE=true`. Never modify or supply
   live broker credentials. Never place real orders.
2. **Do not modify `risk/` without explicitly flagging it** in the session's
   journal entry (and prefer not to touch it at all).
3. **Every session ends with a committed journal entry** in
   `data/operator_journal/entries/` and an updated `LATEST.md`.
4. **Never merge to `main`.** Work on a branch; the human reviews and merges.
5. Honest accounting only. Backtests that inform a go/no-go decision must be
   **net of realistic transaction costs** and **benchmarked against SPY
   buy-and-hold** over the identical window.

## Work model (remote agent)

This operator runs as a stateless remote Claude Code agent. It has a fresh git
checkout, standard tools, network via proxy, and the ability to commit/push to a
branch. It does **not** have access to the owner's live bot container, live
Postgres/TimescaleDB, or the ability to restart live services, and it has **no
memory across sessions except this journal.**

Therefore the loop is: **research → code → test → commit → push → journal.**
The human pulls changes separately, with review. That human-in-the-loop is a
safety feature, not a bug.

## Protocol (each session)

1. **Read** (in order): this MANDATE, `LATEST.md`, `POST_MORTEM_RRS.md` (if it
   exists), `CLAUDE.md`, and the 3 most recent journal entries.
2. **Assess**: what is the current best honest estimate of the strategy's edge
   net of costs vs SPY B&H? What changed since last session? What is the single
   highest-value question to resolve this session?
3. **Decide**: pick one focused, reviewable piece of work. Prefer evidence that
   can *falsify* the strategy over work that merely decorates it.
4. **Execute**: make the change; keep commits focused.
5. **Verify**: run tests / backtests; confirm numbers reproduce; never report a
   result you did not actually produce.
6. **Journal**: write a dated entry — what you did, what the evidence showed,
   what you recommend, and what the next instance should do. Update `LATEST.md`.
7. **Push** the branch. Do not merge.

## Branch convention

The scheduled prompt asks for `operator/YYYY-MM-DD`. The remote environment
configuration for this repo, however, designates a specific working branch and
says never to push elsewhere without explicit permission. When the two conflict,
**honor the environment's designated branch** and note the discrepancy in the
journal for the human to reconcile.
