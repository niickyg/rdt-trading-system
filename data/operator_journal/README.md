# Operator Journal

This directory is the persistent memory of the autonomous RDT operator, which is
otherwise stateless between scheduled runs.

## Expected contents (per the autonomous-operator prompt)

- `MANDATE.md` — the operator's constitution: hard constraints and protocol.
  **MISSING as of 2026-11-11.** Must be authored/committed by a human. The
  operator must NOT write its own mandate.
- `LATEST.md` — pointer to the most recent entry.
- `entries/YYYY-MM-DD-*.md` — one honest entry per run.

The prompt also references `POST_MORTEM_RRS.md` at the repo root (also missing).

## Current state

This directory was bootstrapped on 2026-11-11 by an operator run that found no
mandate. See `LATEST.md` and `entries/2026-11-11-mandate-missing.md`. Until a
human restores the mandate and journal history, runs cannot do more than
re-report this blocked state.
