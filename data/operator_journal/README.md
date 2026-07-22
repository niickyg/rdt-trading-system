# Operator Journal

Persistent memory for the autonomous operator of the RDT Trading System. The operator is
**stateless** across scheduled runs — this directory is its only continuity.

## Files

- **`MANDATE.md`** — the constitution. Mission, hard safety constraints, success metric,
  and the per-run protocol. Every run reads this first. (Bootstrapped 2026-07-22 because it
  was referenced by the scheduler but had never been committed.)
- **`LATEST.md`** — pointer + one-line status of the most recent run, and what the next run
  should do. Update it at the end of every run.
- **`entries/`** — one dated markdown file per run. Honest record: what was observed, what
  was done, what the evidence says, what's next. Never rewrite history; append new entries.

## The one rule

Honesty outranks motion. A correct "this doesn't work and here's the evidence" is a good
run. Manufactured activity to look busy is a failed run. See `MANDATE.md` §5.
