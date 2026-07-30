# Operator Journal

Persistent memory for the autonomous operator of the RDT Trading System. The
operator is **stateless** across sessions — this directory is the only thing it
remembers. Read `MANDATE.md` first.

## Files

- **`MANDATE.md`** — the constitution: mission, hard constraints, and the protocol
  every session must follow. Read it fully, every session.
- **`LATEST.md`** — a pointer to and summary of the most recent session's entry.
  Update it at the end of every session.
- **`entries/YYYY-MM-DD-<slug>.md`** — one file per session. Immutable once
  written; the record of what was assessed, done, verified, and concluded.

## Conventions

- Every session appends exactly one entry (or more, if it does genuinely distinct
  units of work) and updates `LATEST.md`.
- Every quantitative claim cites the script/data that produced it. No unverified
  numbers.
- Reusable analysis scripts should be committed (e.g. under `research/` or
  `scripts/`) so findings are reproducible by the next instance and the human.

## Related root docs

- `POST_MORTEM_RRS.md` — running honest account of why the bot is in its current
  state and what the evidence says about its edge.
