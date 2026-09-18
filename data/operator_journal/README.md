# Operator Journal

Persistent memory for the autonomous operator of the RDT Trading System. The
operator runs **stateless** — each session starts fresh and knows only what is
written here. This directory *is* the continuity.

## Files

- `MANDATE.md` — the constitution: mission, hard constraints, protocol.
  **Read first, every session.**
- `LATEST.md` — pointer to the most recent entry (what the previous instance
  did). Update it at the end of every session.
- `entries/` — one dated markdown file per session, append-only. Never edit a
  past entry; write a new one.
- `data/` — machine-readable artifacts (result snapshots, metrics) referenced
  by entries, for provenance.

## Entry naming

`entries/YYYY-MM-DD-<slug>.md` (e.g. `2026-09-18-bootstrap.md`). If multiple
runs occur in a day, suffix `-2`, `-3`, ….

## Related repo-root files

- `POST_MORTEM_RRS.md` — narrative history of how the strategy reached its
  current state. Read after the mandate.
