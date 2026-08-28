# Operator Journal

This directory is the persistent memory of the **autonomous operator** — a stateless
agent whose mission is to make the RDT paper-trading bot honestly profitable (positive
P&L net of real costs, beating SPY buy-and-hold), or to say clearly when the evidence
does not support that and recommend escalation or wind-down.

## Layout
- `MANDATE.md` — **(REQUIRED, currently MISSING)** the operator's constitution: hard
  constraints, risk limits, protocol, escalation thresholds. Must be authored/ratified
  by a human. Until it exists, operators must not make autonomous strategy changes.
- `MANDATE.PROPOSED.md` — a **draft** mandate awaiting human review. **Not in force.**
- `LATEST.md` — pointer + one-paragraph summary of the most recent entry.
- `entries/YYYY-MM-DD-*.md` — one honest entry per session: what was assessed, what
  changed, why, verification, and hand-off state for the next instance.

## Non-negotiable safety rules (mirrored from the task setup)
1. **PAPER TRADING ONLY.** Never set `AUTO_TRADE=true`; never modify live broker
   credentials.
2. **Do not touch `risk/`** without explicitly flagging it in the journal entry.
3. **Every session ends with a committed journal entry** and an updated `LATEST.md`.
4. Work on a session branch; **do not merge to `main`** — a human reviews and merges.
5. Report outcomes honestly. Backtests are claims until a reproducible, cost-inclusive
   result artifact is committed.

## Current status (2026-08-28)
Bootstrapped on the first operator run. `MANDATE.md` and `POST_MORTEM_RRS.md` are
missing and must be supplied by a human before autonomous strategy work proceeds.
See `entries/2026-08-28-first-run.md`.
