# Operator Journal

Persistent, in-repo memory for the stateless autonomous operator that runs the RDT
Trading System. Each scheduled run reads the latest state here, acts, and appends a new
entry before ending.

## Structure

```
data/operator_journal/
├── MANDATE.md    ← MISSING. The operator's constitution: hard constraints + protocol.
│                   Must be authored/committed by the authorized human operator, not by
│                   the agent. Every run reads this FIRST.
├── LATEST.md     ← Pointer to the most recent entry + current status.
├── README.md     ← This file.
└── entries/
    └── YYYY-MM-DD-<slug>.md   ← One per run, append-only. Never edit past entries.
```

## Rules for each run

1. Read `MANDATE.md` fully first, then `LATEST.md`, then the 3 most recent `entries/`.
2. Follow the MANDATE's `## Protocol` exactly.
3. End every session with a committed entry in `entries/` and update `LATEST.md`.
4. Paper trading only. Never set `AUTO_TRADE=true`. Flag any `risk/` change in the entry.

> This scaffold was bootstrapped on 2026-10-08 because the framework did not previously
> exist in the repo. `MANDATE.md` is still required and must be supplied by the human.
