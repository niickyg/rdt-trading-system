# POST-MORTEM: RRS Strategy & the RDT Trading System

> History of why the bot is in its current state. Bootstrapped 2026-11-13 by the first operator
> instance. This file is evidence, not aspiration. Every claim here is sourced. Future operators:
> append, correct, and date your additions — do not quietly rewrite history.

---

## TL;DR

The RDT Trading System automates an intraday Real Relative Strength (RRS) momentum strategy. After
~9 months of live paper trading it has **lost 61.5% of its value (time-weighted)** while SPY rose
~13%, and it has been **dormant for 3.5 months**. The best backtest ever recorded returns 3.4%
annualized — which also loses to SPY buy-and-hold. The strategy has **never demonstrated a
cost-aware edge over simply holding an index.** Internal strategy docs tacitly concede this (they
compute a *negative* Kelly fraction and pivot to selling a signal service to hit return targets).

---

## 1. Ground-truth evidence (pulled live 2026-11-13 via IBKR MCP connector)

### Live paper account
- `get_account_summary`: net liquidation = **$5**, no open positions, buying power $5.
- `get_pa_performance_all_periods` (TWR, the honest return measure that strips cash flows):
  - Inception **2026-02-25**, measured through **2026-11-13**.
  - Cumulative time-weighted return: **−0.615 (−61.5%)**.
  - NAV path (reflects deposits/withdrawals too): $50 → bled to $21 by mid-March → funded up to
    $521 (2026-03-20) → steady decline $521 → $506.5 → $492 → $477.5 → **collapsed to $5 on
    ~2026-07-31** → flat at $5 ever since.
- `get_account_trades` for `DAYS_90` and `LAST_QUARTER`: **empty** — zero trades. The bot has not
  traded since roughly late July 2026.

> Note: the connected account is NOT the $25K paper account `DUP995654` described in CLAUDE.md.
> It is a small account that peaked near $521. Whether this is a different paper account, a reset,
> or a wind-down of the original is **unverified**. Either way the *time-weighted return* (−61.5%)
> is a cash-flow-independent verdict on the trading itself, and it is deeply negative.

### The benchmark (same connector, SPY conid 756733)
- `get_price_snapshot`: SPY `cumulative_perf_ytd` = **+13.74%**, `cumulative_perf_1y` = **+14.07%**,
  last price $759.17. The bot's live window (late Feb → now) overlaps YTD almost entirely, so the
  buy-and-hold bar over the bot's own window is **≈ +12% to +14%**.

### The decisive comparison
| | Bot (live paper, TWR) | SPY buy-and-hold |
|---|---|---|
| Return, ~Feb 25 → Nov 13 2026 | **−61.5%** | **≈ +13%** |

Underperformance vs. doing nothing but holding SPY: **~75 percentage points.**

## 2. Signal / outcome tracking (from `data/signals/signal_metrics.json`, 2026-11-13)
- 880 total scans; 120 signals generated; **802 scans produced no signal**.
- Signal direction was wildly skewed: **119 short vs. 1 long** — implausible for a methodology that
  is primarily long-biased in uptrends, and a red flag that the live scanner's gating or regime
  logic was misbehaving during the sampled period.
- **Only 2 signals ever had tracked outcomes** (1 target hit, 1 stop out). There is effectively
  **no outcome/attribution data** — the system generated signals but almost never closed the loop
  on whether they worked. You cannot improve what you do not measure; this is a first-order gap.

## 3. What the internal strategy docs already admit
`ACTIONABLE_100X_STRATEGY.md` (checked into the repo) states, with its own math:
- Win rate ~38%, profit factor ~1.29, ~215 trades/yr at 1% risk.
- **Kelly criterion ≈ −0.02 (negative)** — i.e. the estimated edge is marginal-to-nonexistent and
  increasing position size raises risk of ruin without raising expectancy.
- To reach its "$25k profit / 100%" headline it relies mostly on **signal-service revenue and API
  access**, not trading P&L. That is a tell: if the trading edge were real, you would scale the
  trading, not sell subscriptions.

CLAUDE.md's own 2-year walk-forward table tops out at **$1,716 / 3.4% annualized** for the best
("RDT") filter set — and that figure is **gross of realistic commissions and slippage** across 279
trades, and is **not** compared to SPY (which returned far more over Feb-2024→Nov-2025). Net of
honest costs and against the benchmark, the best backtest is a loss relative to buy-and-hold.

## 4. Why it underperforms (assessment, not certainty)
1. **No demonstrated edge.** A ~38% win rate with ~1.3 profit factor and negative Kelly is, within
   noise, indistinguishable from no edge. Intraday US-equity momentum is one of the most
   competed-away, cost-sensitive spaces for a retail automated system.
2. **Costs dominate a thin edge.** Hundreds of intraday round-trips a year, each paying spread +
   slippage + commission, will convert a marginal gross edge into a net loss. The backtests do not
   honestly price this.
3. **Buy-and-hold is a brutally high bar in an up market.** 2026 was a strong SPY year (+~14%). A
   market-neutral-ish intraday churn strategy must clear that bar *and* its own costs to justify
   existing. It did not.
4. **Measurement was broken.** With only 2 tracked outcomes out of 120 signals and a 119:1
   short/long skew, the system was effectively flying blind; no feedback loop could have corrected
   it.

## 5. The one experiment that could change the verdict
Before recommending wind-down, a future operator should run **one** clean, falsifiable test — and
only this, not a parameter safari:

> **Out-of-sample, cost-aware walk-forward of the current best ("RDT filter") config over the bot's
> own live window (2026-02-25 → present), with (a) realistic per-trade costs subtracted
> [commission + half-spread slippage per side], and (b) SPY total return over the identical window
> reported alongside.**

- Harness already exists: `scripts/run_walkforward_v2.py` / `backtesting/engine*.py`.
- Requires installing deps (`pip install -r requirements-web.txt`) and market data access, which
  the fresh remote checkout lacks by default — budget for that or set it up for the next run.
- **Decision rule:** if net-of-cost return does not exceed SPY buy-and-hold over the same window
  with a margin larger than backtest noise, the verdict stands: **wind down and default the capital
  to an index.** If it clearly beats SPY net of costs, document exactly which config/regime did it
  and re-open posture (B).

## 6. Standing recommendation (2026-11-13)
**Default the capital to a broad index (e.g. SPY) and do not run this bot live** until the
experiment in §5 produces a cost-aware, out-of-sample result that beats buy-and-hold. The honest,
evidence-based position today is that the bot is a sophisticated way to underperform an index fund
while taking more risk and doing more work. Keep the codebase as a research sandbox; stop treating
RRS automation as a path to profit until proven otherwise.

---
*Appendix — sources for every number above: IBKR MCP `get_account_summary`,
`get_pa_performance_all_periods`, `get_account_trades`, `get_price_snapshot` (all 2026-11-13);
`data/signals/signal_metrics.json`; `ACTIONABLE_100X_STRATEGY.md`; `CLAUDE.md`. No figure here was
estimated without a cited source except the "≈ +13% over the bot's window" which is interpolated
from SPY YTD (+13.74%) and 1Y (+14.07%).*
