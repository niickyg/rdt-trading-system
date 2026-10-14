# Operator Journal

Persistent memory for the autonomous operator of the RDT Trading System. The
operator is **stateless** across sessions; this directory is the only
continuity between runs.

## Files

- **`MANDATE.md`** — the operator's constitution: mission, hard safety
  constraints, and the per-session protocol. Read first, every session.
- **`LATEST.md`** — pointer + summary for the most recent session. Read second.
- **`entries/`** — one dated entry per session (`YYYY-MM-DD.md`). The full log.
- **`evidence/`** — machine-generated artifacts (backtest JSON, etc.) that back
  up claims in the entries. Never hand-edit; regenerate.

## Session loop (abbreviated — see MANDATE.md for the full protocol)

0. Orient: read MANDATE, LATEST, POST_MORTEM_RRS, CLAUDE.md, last 3 entries.
1. Assess: generate primary, cost-aware, benchmarked evidence.
2. Decide: one highest-value action toward the mission.
3. Execute: small, reviewable, reversible.
4. Verify: re-run the evidence; look for self-deception.
5. Journal: write an entry, update LATEST.md, commit, push (never merge main).

## The yardstick

Success = positive P&L **net of honest costs** that **beats SPY buy-and-hold**
over the same window. Everything else is a means to that end.
