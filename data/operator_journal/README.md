# Operator Journal

Persistent memory for the autonomous `rdt-bot-operator` sessions. Each run is **stateless** —
this directory is the only continuity between instances.

## Files

- **`MANDATE.md`** — the constitution: mission, hard constraints, and protocol. Read first.
- **`LATEST.md`** — points at the most recent entry. Read second.
- **`entries/`** — one dated markdown file per session, append-only history. Never edit a prior
  entry; write a new one.

## Entry naming

`entries/YYYY-MM-DD-short-slug.md`. If multiple entries land on one day, suffix `-2`, `-3`, etc.

## Entry template

```
# YYYY-MM-DD — <title>

**Instance:** <model/date>  **Branch:** <branch>  **Verdict:** <one line>

## Context read
## Assessment (evidence, with numbers)
## What I changed
## How I verified
## Honest verdict vs. the mission
## Handoff for the next instance
```
