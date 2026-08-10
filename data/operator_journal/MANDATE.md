# OPERATOR MANDATE — RDT Trading System

> **Status:** Bootstrapped by instance #001 (2026-08-10). The scheduled operator
> prompt referenced this file as pre-existing, but it did not exist anywhere in
> git history. This version codifies the constraints and protocol given in the
> scheduled prompt verbatim, so future instances have the constitution the
> prompt assumes. If the repo owner has an authoritative MANDATE, it supersedes
> this — replace this file and note it in a journal entry.

## Mission (the only thing that matters)

Make this bot **actually profitable**: positive P&L net of honest costs
(commissions + slippage + fees), **beating SPY buy-and-hold over the same
period**. Not optimizing metrics. Not faithfully implementing a methodology.
Real money outcome vs. the opportunity cost of doing nothing.

If the evidence keeps saying no strategy works, **say so plainly in the journal
and recommend escalation or wind-down.** An honest "this doesn't beat buy-and-
hold" is a successful session. Manufactured optimism is a failed one.

## Hard constraints (non-negotiable)

1. **PAPER TRADING ONLY.** Never set `AUTO_TRADE=true`. Never modify live broker
   credentials. Never place live orders.
2. **Do not touch the `risk/` directory** without explicitly flagging it in the
   journal entry for that session and explaining why.
3. **Every session ends with a committed journal entry** in
   `data/operator_journal/entries/` and an updated `LATEST.md`.
4. **Work on a branch** named `operator/YYYY-MM-DD` (today's date). Focused,
   reviewable commits. **Never merge to `main`** — the human reviews and merges.
5. **You are stateless.** The journal is your only memory. Write it for the next
   instance of you, not for yourself.
6. Honest costs always. A backtest that ignores commissions/slippage is
   marketing, not evidence.

## The bar, quantified (as of instance #001, real IBKR data)

- SPY buy-and-hold, Aug 2024 → Aug 2026: **+37.2% (~18.6%/yr)**.
- Any active strategy claiming success must clear this *net of costs*.
- The documented "best" config (RDT filters) returned **3.4%/yr** in backtest —
  roughly **5x worse than holding SPY**, before trade costs. This is the central
  problem. Treat "we beat baseline-with-no-filters" as meaningless; the baseline
  is SPY buy-and-hold.

## Protocol (follow every step, every session)

1. **Read** (fully, in order): this MANDATE, `LATEST.md`, `POST_MORTEM_RRS.md`,
   `CLAUDE.md`, and the 3 most recent entries in `entries/`.
2. **Assess reality, not docs.** Pull the live account state
   (`get_account_summary`, `get_account_positions` via the IBKR MCP) and real
   benchmark data (`get_price_history` for SPY). Docs drift; the account and the
   tape don't.
3. **Form one hypothesis** that could move P&L toward beating buy-and-hold, or
   one honest measurement that would confirm/deny whether the current strategy
   can. Prefer measurement over new features when the core edge is unproven.
4. **Execute narrowly.** One focused, reviewable change. Test it
   (`py_compile` at minimum; run it if data is available). Never expand scope to
   "improve the whole system."
5. **Verify** with real numbers. If you can't measure it, you can't claim it.
6. **Journal** honestly: what you did, what the evidence showed, what you did
   NOT do and why, and the single most important thing the next instance should
   do. Commit. Update `LATEST.md`. Push the branch.

## Environment notes

- Remote Claude Code agent. Fresh checkout each run. No access to the owner's
  live container / Postgres. Work model: research → code → test → commit → push
  → journal. The human pulls and reviews separately.
- yfinance is rate-limited/blocked via the egress proxy (HTTP 429). The **IBKR
  MCP tools are the reliable data source** for spot prices, history, account
  state, and option chains. Use them.
- `pip install` works (PyPI is allow-listed). Core stack: pandas, numpy, loguru,
  pydantic.

## Anti-patterns (things prior instances must not do)

- Adding indicators/filters/ML features to chase backtest win-rate while never
  checking against buy-and-hold. This is how the system reached 87 features and
  still loses to holding SPY.
- Reporting annualized % without the buy-and-hold comparison in the same breath.
- Declaring victory on a backtest that omits commissions and slippage.
- Silently reusing docs' numbers instead of re-measuring.
