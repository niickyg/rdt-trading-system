# Operator Journal

Persistent memory for the autonomous operator of the RDT Trading System. The operator
is **stateless** — each scheduled run is a fresh Claude Code instance with no memory of
prior runs. This directory *is* its memory.

## Files

- **`MANDATE.md`** — the constitution. Mission, hard constraints (paper-only, don't
  touch `risk/`, don't merge to main, journal every run), and the step-by-step
  protocol every run follows. **Read first, in full, every run.**
- **`LATEST.md`** — a pointer + summary of the most recent run and the ranked agenda
  for the next one. Read second.
- **`entries/`** — one dated Markdown file per run (`YYYY-MM-DD-run-NNN.md`), using the
  template in the mandate. Append-only history; never edit past entries.

## How a run works (short version)

1. Read `MANDATE.md`, `LATEST.md`, the 3 most recent `entries/`, `POST_MORTEM_RRS.md`,
   `CLAUDE.md`.
2. Assess reality from the **live IBKR account** (MCP tools) + the SPY benchmark.
3. Decide one focused, offline-verifiable change.
4. Execute on a dated branch (never `main`), paper-only.
5. Verify (compile/test; no unverified claims; state results vs SPY buy-and-hold).
6. Write a new entry, update `LATEST.md`, commit, push.

The bar for "profitable" is **beating SPY buy-and-hold**, not beating zero.
