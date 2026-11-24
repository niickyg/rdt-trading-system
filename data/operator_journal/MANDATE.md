# MANDATE — STUB / PLACEHOLDER (NOT THE REAL MANDATE)

> ⚠️ **THIS IS NOT THE AUTHENTIC OPERATOR MANDATE.**
>
> The real `MANDATE.md` — the operator's constitution, hard constraints, and
> `## Protocol` — was **missing** from this repository as of 2026-11-24. A fresh
> cloud checkout had no copy of it, no git history of it, and it was not
> gitignored. This stub exists only so the missing-file condition is explicit
> and fail-closed, not silently absent.

## HALT CONDITION — read this before acting

If you are an operator instance reading this file and it still contains this
"STUB / PLACEHOLDER" banner, then **the real mandate has not been restored.**
In that state:

- **DO NOT** make any changes to trading logic, scanner, strategy, ML models,
  broker configuration, or anything in `risk/`.
- **DO NOT** improvise, author, or infer your own "constitution" or hard
  constraints. The constraints that bind an autonomous trading agent must be
  authored by the human operator, not by the agent.
- **DO NOT** enable `AUTO_TRADE=true` or touch live broker credentials. Paper
  trading only, always.
- **DO** write a journal entry documenting that the mandate is still missing,
  update `LATEST.md`, commit, and end the session.

## What a human must do to unblock

Replace this file with the authentic `MANDATE.md` and commit it, along with:
- `data/operator_journal/LATEST.md` (real history)
- `POST_MORTEM_RRS.md`
- the historical `data/operator_journal/entries/*`

Once this file no longer carries the stub banner above and contains a real
`## Protocol` section, normal operator runs may resume under those rules.

See `entries/2026-11-24-session-blocked-missing-mandate.md` for the full
investigation that produced this stub.
