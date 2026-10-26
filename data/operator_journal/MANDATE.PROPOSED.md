# ⚠️ PROPOSED MANDATE — NOT YET AUTHORITATIVE

> **This is a DRAFT written by the operator agent on 2026-10-26 because the real
> `MANDATE.md` was missing from the repository.** It is **not** binding and the
> operator did **not** act under it. A human must review, edit, and — if
> acceptable — rename it to `MANDATE.md` to make it authoritative. Until a file
> named exactly `MANDATE.md` exists, future runs should treat the mandate as
> absent and escalate rather than make trading/risk changes.
>
> The hard constraints below are transcribed from the standing scheduled-task
> prompt (the only authoritative source available). The protocol is a reasonable
> reconstruction for the human to correct, not a reconstruction of any lost
> original — the original's contents are unknown.

---

## Mission
Make this paper-trading bot **actually profitable**: positive P&L net of honest
transaction costs, **beating SPY buy-and-hold** over the same period. Not metric
optimization, not methodology adherence for its own sake. If the evidence keeps
saying no strategy works, **say so in the journal and recommend escalation or
wind-down** — that is a valid, expected outcome, not a failure to hide.

## Hard constraints (absolute)
1. **PAPER TRADING ONLY.** Never set `AUTO_TRADE=true`. Never modify or add live
   broker credentials.
2. **Never modify anything under `risk/` without explicitly flagging it** in that
   run's journal entry (and prefer not to at all without human sign-off).
3. **Every session ends with a committed journal entry** in
   `data/operator_journal/entries/` and an updated `LATEST.md`.
4. Work on a branch (`operator/YYYY-MM-DD`). **Never merge to `main`** — the
   human reviews and merges.
5. Honesty over optimism: report P&L, costs, and benchmark comparison truthfully;
   never assert profitability you have not measured.

## Protocol (DRAFT — human to refine)
1. **Read** this mandate, `LATEST.md`, `POST_MORTEM_RRS.md`, `CLAUDE.md`, and the
   3 most recent journal entries.
2. **Assess** current state: is there a demonstrated edge vs SPY, on real
   (not 2-sample) outcome data, net of costs?
3. **Decide** the single highest-value, *verifiable* action. Prefer measurement
   and falsification over adding complexity. If nothing is verifiable this run,
   the correct action may be to measure or to escalate.
4. **Execute** narrowly: focused, reviewable commits. No risk/execution changes
   without a journal flag and (ideally) human sign-off.
5. **Verify**: `python -c "import py_compile; ..."` on changed files; run relevant
   tests; reproduce before/after where a backtest is involved.
6. **Journal**: evidence, decision, rationale, what the next operator should do.
7. **Commit & push** the branch. Do not merge.

## Open questions for the human to resolve in the real MANDATE.md
- What counts as "enough" evidence to justify a code change vs. escalate?
- Hard P&L / drawdown thresholds that trigger automatic wind-down recommendation.
- Exactly which branch the human-review pipeline pulls from.
- Whether the operator may ever change `risk/` at all, and under what gate.
- How real paper outcomes get recorded so profitability is measurable.
