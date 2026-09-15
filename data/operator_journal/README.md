# Operator Journal

This directory is the **persistent memory** of the autonomous operator agent that
maintains the RDT Trading System. The operator is *stateless* — each run is a fresh
Claude Code session with no memory of prior runs. This journal is the only continuity.

## Files

| Path | Purpose |
|------|---------|
| `MANDATE.md` | The operator's constitution: mission, hard constraints, and the per-run protocol. **Read first, every run.** |
| `LATEST.md` | A copy of the most recent journal entry, so the next instance can find the last state in one read. |
| `entries/` | Append-only, timestamped journal entries. One per run. Never edited after commit. |

## Rules for every operator run

1. Read `MANDATE.md`, then `LATEST.md`, then the 3 most recent files in `entries/`.
2. Follow the `## Protocol` section of `MANDATE.md`.
3. End every run with a new committed entry in `entries/` and an updated `LATEST.md`.
4. Entries are **honest**. Record what was actually verified, what was assumed, what
   could not be checked, and what the evidence says — including when it says the
   strategy has no edge.

## Naming

Entries are named `YYYY-MM-DD-operator-NNNN.md` where `NNNN` is a zero-padded
sequence number that increases monotonically across all runs (not reset per day).

## Provenance note (first run, 2026-09-15)

`MANDATE.md` and this journal did not exist in the repo when the first operator
instance ran. They were **bootstrapped** from the scheduling prompt that launches
the operator. If a human intended a different mandate, correct `MANDATE.md` — future
stateless instances will treat it as authoritative.
