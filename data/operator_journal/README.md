# Operator Journal

Persistent memory for the autonomous operator of the RDT Trading System. The
operator is **stateless** across sessions; this directory is its only continuity.

## Files
- **`MANDATE.md`** — the constitution: mission, hard constraints, protocol. Read
  first, every session.
- **`LATEST.md`** — pointer + one-paragraph summary of the most recent session.
- **`entries/`** — one immutable, dated entry per session
  (`YYYY-MM-DD-NNNN-slug.md`). Append-only; never rewrite past entries.

## Session loop (short form)
Read MANDATE → read LATEST → read POST_MORTEM_RRS.md → assess evidence → pick the
single highest-leverage *verifiable* action → execute → verify → write a new
entry + update LATEST → commit + push (never merge to `main`).

## Non-negotiables
Paper trading only. Never touch `risk/` without flagging. Never fabricate
numbers. Beating SPY buy-and-hold net of honest costs is the only scoreboard.
