# Operator Journal

This directory is the persistent memory of the **autonomous operator** — a stateless agent whose
sole objective is to make the RDT trading bot actually profitable (positive realized P&L, net of
costs, beating SPY buy-and-hold). Each run has no memory of prior runs; this journal is continuity.

## Files
- **`MANDATE.md`** — the constitution. Mission, hard safety constraints, environment reality, and
  the per-session protocol. Every operator instance reads this first and obeys it over its own
  judgment. *Provisional — drafted by the first instance on 2026-08-04; awaiting human ratification.*
- **`LATEST.md`** — pointer to the newest entry. Read after the mandate.
- **`entries/`** — one dated file per session (`YYYY-MM-DD-<slug>.md`). Append-only history.

Related, at repo root: **`POST_MORTEM_RRS.md`** — why the bot is where it is.

## Reading order for a new instance
1. `MANDATE.md`  2. `LATEST.md`  3. `../../POST_MORTEM_RRS.md`  4. `CLAUDE.md`  5. the 3 newest `entries/`

## The one rule that matters most
Report only what you measured, labeled by source (`[LIVE]` / `[BACKTEST]` / `[ESTIMATE]`). The
scoreboard is **realized bot TWR vs SPY over the same window** — nothing else counts as success.
An honest "this doesn't work, here's the proof" is a good session.
