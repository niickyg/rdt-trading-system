# 2026-10-30 — Bootstrap & Honest Baseline Assessment

**Operator run #1 (bootstrap).** Stateless agent, fresh checkout. Session:
https://claude.ai/code/session_016rEf9EKhxhUPbi695QkQP6

## TL;DR

- The operator journal infrastructure described in the invoking prompt **did not exist**
  (`data/operator_journal/`, `MANDATE.md`, `POST_MORTEM_RRS.md`, `LATEST.md` were all absent).
  This run creates it. This is run #1, not a continuation.
- **There is no live trading and no real track record to assess.** The connected IBKR
  account holds **$5**, zero positions, **zero trades in the last 90 days**. The app's own
  `signal_metrics.json` records only **2 outcomes ever** (1 win, 1 loss) and is stale since
  2026-03-05.
- **The bot does not beat SPY buy-and-hold — not close.** Verified with real SPY prices
  this session: buy-and-hold returned **+20.2% in 13 months** and **~16.3%/yr over 2 years**.
  The bot's own best-documented backtest is **+6.9% over 21 months (~3.4%/yr)**, with a
  self-admitted **negative Kelly** and ~38% win rate. Buy-and-hold wins by ~5x on return
  while the bot adds day-trading risk, cost, and operational complexity.
- **Recommendation:** Do not add features. The next run must answer one question with a
  real, cost-aware backtest: *does the raw RRS signal have any positive expectancy net of
  honest costs?* If it cannot be shown to, recommend wind-down of active trading and
  reallocating to the benchmark. Details below.

## What I assessed and how

### 1. Infrastructure reality
`MANDATE.md`, `LATEST.md`, `POST_MORTEM_RRS.md`, and `data/operator_journal/` do not exist
in the repo. The invoking prompt assumes a prior instance created them; none did (or it was
never committed). Created `MANDATE.md` (constitution, bootstrapped from the invoking prompt),
`entries/`, and this entry. `LATEST.md` now points here.

Also note: `CLAUDE.md` describes a `research/` factor-analysis framework (factors.py,
factor_tester.py, etc.). **That directory is empty in this checkout.** Documentation
describes capabilities that are not present — a recurring pattern to watch for.

### 2. Live account — read-only IBKR MCP connector
- `get_account_summary` → net_liquidation **$5**, buying_power $5, gross_position_value $0.
- `get_account_trades(DAYS_90)` → **`{"trades": []}`**.
- This is NOT the "DUP995654 / $25K paper account" described in CLAUDE.md. Whatever account
  the MCP connector is wired to is an empty shell. **No live evidence of profitability exists
  — positive or negative.** All profitability claims in the repo rest on backtests.

### 3. App-recorded track record
`data/signals/active_signals.json`: 1 stale signal (ROST long, generated 2026-03-05).
`data/signals/signal_metrics.json`: 880 scans, 120 signals, but **only 2 tracked outcomes**
(1 target hit, 1 stop-out), last scan 2026-03-05. Effectively **no outcome data**. Note also
119 of 120 signals were *short* — an odd skew worth investigating if trading resumes.

### 4. The benchmark that must be beaten — SPY buy-and-hold (REAL DATA)
Source: IBKR `get_price_history`, SPY (contract 756733), STK, ONE_MONTH bars, 2yr.
Method: monthly closes, point-to-point simple return.

| From | To | SPY close | Return | Annualized |
|------|----|-----------|--------|-----------|
| 2024-10-31 | 2025-11-03 (~13 mo) | 568.64 → 683.39 | **+20.2%** | ~18%/yr |
| 2024-10-31 | 2026-10-01 (24 mo) | 568.64 → 769.36 | **+35.3%** | **~16.3%/yr** |

### 5. The bot's best documented result (UNVERIFIED — quoted from repo)
From `CLAUDE.md` walk-forward table and `ACTIONABLE_100X_STRATEGY.md` (I did not reproduce
these this run; yfinance is not installed so the repo's backtest scripts don't run as-is):
- Config C "RDT Filters": **+6.9% total over ~21 months (Feb 2024–Nov 2025) = ~3.4%/yr**,
  win rate 49.5%, profit factor 1.24, 279 trades.
- `ACTIONABLE_100X_STRATEGY.md` states plainly: win rate ~38%, profit factor ~1.29,
  **Kelly ≈ −0.02 (negative)** → "the current edge is marginal... increasing position size
  actually increases risk of ruin without improving returns."

## The gap (numeric, honest)

Over an overlapping window, **SPY buy-and-hold ~16–18%/yr vs the bot's best ~3.4%/yr.**
Buy-and-hold wins by roughly **5x on return**, with far lower variance, no day-trading
commissions/slippage, no overnight operational risk, and no model/infra maintenance. By the
mandate's definition of success (beat buy-and-hold, net of honest costs), **the bot currently
fails.** The repo's own math (negative Kelly) says this is structural, not a tuning problem.

## What I did NOT do, and why

- **Did not add any trading feature, ML model, or filter.** Per the mandate's standing gate:
  core edge is unproven, so added complexity is noise.
- **Did not touch `risk/`.**
- **Did not run a new backtest.** yfinance isn't installed; a proper cost-aware backtest is
  the right task for the next run with a reproducible harness, not a rushed one here.
- **Did not place or modify any order.** Broker use was read-only.
- **Branch note:** the invoking prompt asked for a branch `operator/2026-10-30`, but the
  harness git policy for this session designates branch `claude/adoring-feynman-5ikgcz` and
  forbids pushing elsewhere without explicit permission. I committed to the designated branch
  and am flagging the naming discrepancy here for the owner to reconcile.

## Open risks / unknowns

- Which account the live bot actually trades (the $5 MCP account vs the documented $25K
  paper account) is unclear. The owner should confirm what, if anything, is running live.
- The 119:1 short:long signal skew in `signal_metrics.json` is unexplained.
- Several repo docs (research framework, backtest numbers) could not be verified here.

## Recommendation for the next instance (do this, in order)

1. **Prove or kill the edge.** Stand up a *reproducible, cost-aware* backtest of the raw RRS
   signal (before the filter stack) on real data. Install deps into the scratchpad if needed,
   or rebuild a minimal harness using IBKR price history. Charge realistic costs: commission +
   ≥1 tick spread/slippage per side. Question: **is net expectancy per trade > 0 after costs?**
2. If **yes**: measure whether the filtered strategy's risk-adjusted return (Sharpe, max DD)
   beats buy-and-hold. Only then consider sizing/selection improvements.
3. If **no** (expected, given negative Kelly): write the wind-down recommendation plainly —
   the honest answer is to hold the benchmark (SPY/broad index) rather than day-trade a
   negative-edge system. That is a *successful* outcome under this mandate.
4. Either way: resist the project's documented drift toward a "sell the signals as SaaS"
   pivot as a substitute for a real edge. A signal service with no demonstrated edge is
   selling noise.

---
*No live orders were placed. Broker connector used read-only. Paper-only mandate upheld.*
