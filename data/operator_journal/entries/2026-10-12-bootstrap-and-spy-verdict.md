# Operator Entry — 2026-10-12 — Bootstrap + SPY Benchmark Verdict

- **Instance:** First autonomous operator run (bootstrap).
- **Branch:** `claude/adoring-feynman-vq5egq` (see "Branch note" below).
- **Base commit:** `ae4350a` ("feat: SaaS product overhaul …").
- **Model/runtime:** Remote Claude Code agent (research/code/test/commit/push only).

---

## Objective chosen this session

The routine told me to read `MANDATE.md`, `LATEST.md`, `POST_MORTEM_RRS.md`, and the 3
most recent journal entries first. **None of those files existed** — the entire
`data/operator_journal/` tree, the mandate, and the post-mortem were absent from every
branch and from disk. So this was necessarily a **bootstrap** session with two objectives:

1. Create the operator journal infrastructure the mission depends on (constitution + history).
2. Answer the one question that actually matters, with real evidence: **does this bot beat
   SPY buy-and-hold?**

## What I did

1. **Confirmed the journal infra was missing** (`git log --all`, tree search, disk search —
   no MANDATE/LATEST/POST_MORTEM/operator_journal anywhere).
2. **Authored the foundation:**
   - `data/operator_journal/MANDATE.md` — constitution. Hard constraints transcribed from
     the scheduled routine prompt (paper-only, don't touch `risk/` without flagging, journal
     every session, branch/no-merge, honesty). Added an epistemic-rules section and a
     step-by-step protocol for future stateless instances, plus an anti-drift note.
   - `POST_MORTEM_RRS.md` — honest reconstruction of why the bot is where it is, built from
     the repo's own docs + git history, with unverified claims labeled.
3. **Built the missing benchmark tool** `scripts/benchmark_spy.py` — computes SPY
   buy-and-hold (dividend-adjusted) over the same window the walk-forward uses. The repo had
   **no** SPY-benchmark comparison anywhere, despite that being the mission's yardstick.
4. **Reproduced the repo's own walk-forward backtest** (`scripts/run_walkforward_v2.py`) on
   **fresh, out-of-sample data (2024-10 → 2026-10)** and ran the SPY benchmark on the same
   window. (Had to install deps; used `fastparquet` because `pyarrow` needs numpy 2.0 while
   the repo pins numpy 1.26.4.)

## Evidence & numbers (reproduced this session — see `2026-10-12-walkforward-evidence.txt`)

Window: trailing 730 days, $25,000 starting capital. All three strategy configs vs the
benchmark the mission actually cares about:

| | Total return (2yr) | Dollars on $25k | Annualized |
|---|---|---|---|
| A) Baseline (RRS 2.0, no filters) | +7.51% | +$1,878 | 3.7% |
| B) Old filters | +8.72% | +$2,180 | 4.3% |
| **C) RDT filters (the repo's "best")** | **+10.80%** | **+$2,699** | **5.3%** |
| **SPY buy-and-hold (same full window)** | **+36.34%** | **+$9,085** | **16.83%** |
| SPY buy-and-hold (post-warmup window the strategy covers) | +33.61% | +$8,404 | 19.51% |

**The best strategy config returned about one-third of simply holding SPY** (+$2,699 vs
+$9,085), and the baseline about one-fifth.

### Why this is even worse than it looks
- **No trading costs are modeled.** `backtesting/engine_enhanced.py` contains zero
  commission/slippage/spread logic (verified by grep). Config C makes **278 trades** over 2
  years; realistic round-trip slippage+spread would erode a meaningful fraction of the
  +$2,699. SPY buy-and-hold is **2 trades**, negligible cost. The real gap is wider.
- **This window was a strong bull market** (SPY +36%). A long-biased momentum strategy
  should be near its best here — and it still lost to the index by ~3.4x. That is the
  damning part: in favorable conditions, active single-name stock-picking underperformed
  doing nothing.
- The walk-forward "total return" is the **sum of 6 non-overlapping ~4-month window P&Ls**,
  not a single compounded account — if anything this flatters the strategy (resets drawdowns).
- Win rate ~50%, profit factor ~1.3 — consistent with the repo's docs and with a **marginal
  edge**. `ACTIONABLE_100X_STRATEGY.md` itself computes **Kelly ≈ −0.02 (negative)**.

## Verdict on profitability (standing verdict as of now)

**NOT PROFITABLE RELATIVE TO THE BENCHMARK.** On reproduced, out-of-sample data, the bot's
best configuration underperforms SPY buy-and-hold by roughly 3.4x in dollars and ~11.5
percentage points annualized, before trading costs, during a bull market that favors its
style. The core RRS edge is marginal-to-negative (Kelly ≈ 0). There is currently **no
evidence that any configuration of this system beats buy-and-hold.**

This is exactly the situation the mandate says to name plainly rather than paper over.

## Risk-directory changes

**None.** I did not modify anything under `risk/`. (Flag per MANDATE §2.2: no change made.)

## Recommendation / next objective for the next instance

1. **Do not add leverage, raise risk-per-trade, or build more filters/ML on top of this
   edge.** The math (negative Kelly) says sizing into it increases risk of ruin without
   improving expectancy. The `*_100X*.md` docs and the aggressive `DEPLOYMENT_SUMMARY.md`
   profile point the wrong way.
2. **Stop the SaaS/signal-service drift.** Selling signals does not make the trading
   profitable and is outside the mission (MANDATE §6).
3. **Highest-value next experiments** (pick ONE, measure against SPY, include costs):
   - Add realistic commission+slippage to the backtest engine and re-run — quantify how much
     of the already-thin edge survives. (Cheap, high information value. Likely the best next step.)
   - Test whether the strategy has *any* regime where it beats SPY (e.g., does it outperform
     in down/choppy markets? isolate 2022-style data). If it only loses in bull markets and
     also loses in bear markets, the case for wind-down is complete.
   - Honest market-neutral framing: compare long/short RRS to SPY on a risk-adjusted
     (Sharpe) basis, not just total return, over a full cycle including a drawdown year.
4. **If the next 1–2 sessions confirm no edge survives costs across regimes: recommend
   wind-down** of trading ambitions to the human owner, keeping at most a paper-only research
   sandbox to search for a genuinely different edge. Escalate rather than keep iterating on a
   dead strategy.

## Open questions / blockers

- **Branch note / conflict:** The scheduled routine said to work on `operator/2026-10-12`,
  but the harness "Git Development Branch Requirements" explicitly designate
  `claude/adoring-feynman-vq5egq` and say never to push elsewhere without explicit
  permission. Per MANDATE §2.4 I used the harness-designated branch and am flagging the
  conflict here for the human to resolve.
- No realized paper-trading P&L has ever been recorded in-repo; all numbers remain
  backtests. A real forward paper run would be the ultimate check but requires the owner's
  live container (outside a remote agent's reach).
- Backtest window is a single bull-market regime; the negative verdict would be even more
  robust with a bear-year included. yfinance only reliably gave ~2 years of daily here.
