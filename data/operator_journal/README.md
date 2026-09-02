# Operator Journal

Persistent memory for the stateless autonomous operator of the RDT Trading
System. Each operator run is a fresh instance; this directory is the only thing
that carries knowledge forward.

## Files

- **`MANDATE.md`** — the constitution. Read first, in full, every run. Prime
  directive, hard constraints, honesty rules, success/escalation criteria, and
  the step-by-step protocol.
- **`LATEST.md`** — a pointer + short summary of the most recent entry. Read
  second. Always update it at the end of your run.
- **`entries/`** — one dated markdown file per run:
  `YYYY-MM-DD-<slug>.md`. Append-only history; never rewrite a past entry.

## Entry format

Each entry should answer, honestly:

1. **Assessed** — what state you found, what real evidence you looked at.
2. **Decided** — the single focused thing you chose to do, and why.
3. **Changed** — files touched, with rationale.
4. **Verified** — commands run and their *real* output. Mark UNVALIDATED work.
5. **Open** — what's still unresolved; the current central question.
6. **Recommendation** — the concrete next step for the following instance.

## Ground truth reminder

The bot is **paper trading only**. Numbers in `CLAUDE.md` and the various
strategy docs are *self-reported claims to verify*, not established facts. The
only success that counts is: positive P&L net of honest costs, beating SPY
buy-and-hold over a real out-of-sample window. If the evidence keeps saying no
edge exists, the honest, mandated outcome is to say so and recommend wind-down.
