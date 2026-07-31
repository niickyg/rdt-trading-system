# LATEST — pointer to the most recent operator run

**Most recent run:** [run-001 — 2026-07-31](entries/2026-07-31-run-001.md)

## One-paragraph summary for the next instance

Run-001 was the first-ever operator run and a bootstrap: the entire operator journal
system (MANDATE, entries, this file) and `POST_MORTEM_RRS.md` **did not exist** and had
to be created. The honest baseline, from the live IBKR account: net liq **$5**, **zero
trades** in every queryable period, **−61.5% TWR** since inception (Feb 2026), and
**dormant all of July**. SPY buy-and-hold did **+22% (1y) / +10% YTD** over the same
window. The bot's own best backtest (~3.4%/yr) was **never benchmarked against SPY** and
is dominated by it. **Verdict: the bot is not profitable and there is no evidence it
beats buy-and-hold.** No trading code was changed (nothing was safely validatable this
run); the value delivered was working memory + an honest, evidence-based baseline.

## Start here next run (from run-001's ranked agenda)

1. **Resolve account identity** — is the MCP account the real bot account, or a
   separate/drained one? Until known, "is the bot even trading?" is unknown.
2. **Add SPY buy-and-hold benchmark to the backtest** so every result is reported as
   excess return vs SPY (the real bar). Small, verifiable, high-leverage.
3. **Give the backtest an offline data path via IBKR MCP `get_price_history`** — Yahoo
   Finance is rate-limited (429) in the operator environment, so backtests can't run
   as-is. This is the tooling gate for validating any future change.
4. Then, and only then, judge whether any strategy config shows positive excess return
   vs SPY out-of-sample. If not, escalate wind-down with numbers.

## Note for the human reviewer

The scheduled prompt requested branch `operator/2026-07-31`, but the remote-agent
harness pinned this session to `claude/adoring-feynman-xgl6r8` and forbids pushing
elsewhere. Run-001 pushed there to respect the guardrail. Please reconcile the intended
branch convention. Nothing was merged to `main`; no risk code changed; paper-only.
