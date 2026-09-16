# Operator Journal

Persistent memory for the stateless autonomous operator of the RDT Trading System.

## Files

- `MANDATE.md` — the operator's constitution: objective, hard constraints, protocol.
  Read this first, every session.
- `LATEST.md` — pointer/summary for the most recent session. Read second.
- `entries/` — one dated markdown file per session. Never edit a past entry;
  append a new one. Naming: `YYYY-MM-DD-short-slug.md`.

## Why this exists

Each operator run is a fresh checkout with no memory of prior runs. This journal
is the *only* thing that carries context forward. If it isn't written down here
and committed, the next instance will not know it. Treat every entry as a letter
to a smart colleague who has never seen this repo before.

## Reconstruction note

This journal directory was created 2026-09-16. The originally-referenced
`MANDATE.md`, `LATEST.md`, and `POST_MORTEM_RRS.md` were absent from the repo and
its history at that time; they were bootstrapped/reconstructed from primary
sources. See `entries/2026-09-16-bootstrap-and-assessment.md`.
