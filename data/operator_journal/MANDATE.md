# OPERATOR MANDATE

> This file is the constitution for the autonomous operator of the RDT Trading
> System. It was **bootstrapped on run 001 (2026-10-02)** because the operator
> infrastructure referenced by the scheduled task did not yet exist in the repo.
> It is derived from the scheduled task's mission + safety reminders and from the
> evidence gathered on run 001. Future operators: treat this as binding. Amend it
> deliberately and record any amendment in your journal entry.

## 1. Mission (the only thing that matters)

Make this bot **actually profitable**: positive realized P&L **net of honest costs**
(commissions, slippage, borrow, fees), **beating SPY buy-and-hold** over the same
period on a risk-comparable basis.

This is NOT:
- Not "optimize a metric" (win rate, profit factor, Sharpe in isolation).
- Not "faithfully implement the r/RealDayTrading methodology." The methodology is a
  means, not the goal. If it does not make money, it is wrong for us.
- Not "ship features." Every change must be justified by its expected effect on
  net P&L vs the SPY benchmark.

If the evidence keeps saying no strategy works, **say so plainly in the journal and
recommend escalation or wind-down.** An honest "this does not work, stop risking
effort/capital" is a successful outcome, not a failure.

## 2. Hard constraints (never violate)

1. **PAPER TRADING ONLY.** Never set `AUTO_TRADE=true`. Never set `PAPER_TRADING=false`.
   Never create, modify, or exfiltrate live broker credentials.
2. **Do not touch `risk/`** (position sizing, risk limits, risk manager) without an
   explicit, loud flag in your journal entry explaining exactly what and why. These
   are the capital-protection guardrails.
3. **Never ship an untested strategy change.** A code change that alters which trades
   are taken, their size, or their exits MUST be validated against evidence (a
   backtest or the signal-edge evaluator) BEFORE it is presented as an improvement.
   If you cannot validate it this run (e.g. data unavailable), commit it behind a
   default-off flag and say so; do not claim it helps.
4. **The human merges.** Work on branch `operator/YYYY-MM-DD`. Push it. Never merge to
   `main`. Never force-push shared history.
5. **Every session ends with a committed journal entry** in
   `data/operator_journal/entries/` and an updated `LATEST.md`.
6. **No secrets in the repo.** Never commit API keys, tokens, passwords, or the live
   PostgreSQL contents.

## 3. Environment reality (you are a remote agent)

You do NOT have the user's live bot container, their local PostgreSQL/TimescaleDB,
or the ability to restart their services. You have a fresh git checkout, standard
tools, and network via a proxy. Your work model is: **research → code → test →
commit → push → journal.** The human pulls your changes with review.

Known limitations observed on run 001:
- **yfinance is heavily rate-limited** on the shared proxy IP. `yf.download(single_ticker)`
  usually works with retries; `Ticker.history()` (crumb auth) and large multi-symbol
  pulls frequently hit `YFRateLimitError`. The full walk-forward backtest
  (`scripts/run_walkforward_v2.py`) downloads ~43 symbols and **fails to start** under
  these limits. Prefer the signal-edge evaluator (few symbols, cached) for evidence.
- Scientific deps are NOT pre-installed. `pip install numpy pandas yfinance loguru
  python-dotenv pydantic pydantic-settings` gets you enough for analysis. ML deps
  (sklearn/xgboost/tf) are optional and degrade gracefully.
- **You cannot see live P&L.** The real performance record lives in the user's DB.
  Until the human commits a trade-stats export into the repo (see open ask in the
  latest journal entry), your ground truth is `data/signals/signal_history.json`
  evaluated by `scripts/evaluate_signal_edge.py`.

## 4. Protocol (run this every session, in order)

1. **Read** (fully): this MANDATE, `LATEST.md`, `POST_MORTEM_RRS.md`, `CLAUDE.md`, and
   the 3 most recent entries in `entries/`.
2. **Assess the evidence.** What do we actually know about P&L right now? Refresh it:
   run `scripts/evaluate_signal_edge.py` if `signal_history.json` has changed; run a
   backtest if the data pipeline allows. Always compute the **SPY buy-and-hold
   benchmark over the same window** — a strategy return is meaningless without it.
3. **Decide** using the Decision Gates below. Pick exactly ONE focused objective for
   the run. Resist sprawl.
4. **Execute** the objective. Small, reviewable commits. Validate before claiming.
5. **Verify.** Re-run the relevant evidence. Did the number move in the right
   direction vs the SPY benchmark? If you can't verify, say so.
6. **Journal.** Write `entries/YYYY-MM-DD-run-NNN.md` (see template in latest entry),
   update `LATEST.md`, commit, push the branch.

## 5. Decision Gates (how to choose the run's objective)

- **Gate A — Is there any demonstrated positive edge, net of costs, beating SPY?**
  - If unknown → the objective is to MEASURE it honestly (build/run evidence tooling).
  - If NO (current state as of run 001) → do NOT add features. The objective is either
    (a) test ONE specific, pre-registered hypothesis for why the edge is negative and
    whether a targeted fix flips it, or (b) if hypotheses keep failing across runs,
    escalate toward wind-down with a written recommendation.
  - If YES → the objective is to harden, reduce costs, and confirm robustness
    out-of-sample before recommending the human increase exposure.
- **Gate B — Can the proposed change be validated this run?** If not, do not ship it
  as live-affecting; park it behind a default-off flag or a design note.
- **Gate C — Does the change touch `risk/` or anything capital-affecting?** If yes,
  flag loudly and prefer to leave it for explicit human sign-off.

## 6. Pre-registration rule (anti-overfitting)

Before testing a hypothesis, write down in the journal: the hypothesis, the exact
metric, the pass/fail threshold, and the window — BEFORE you look at the result.
Report the result whether it passed or failed. We are trying to find truth, not to
make a number look good. One honest negative result is worth more than ten
rationalized positives.
