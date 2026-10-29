# POST-MORTEM: The RRS Strategy

> **⚠️ RECONSTRUCTED FROM REPOSITORY EVIDENCE ON 2026-10-29.**
>
> The operator task instructs every run to read `POST_MORTEM_RRS.md` for "the
> history of why the bot is in its current state." That file did not exist in the
> repo. This document reconstructs the post-mortem **strictly from artifacts
> present in the repository** (optimization JSON, signal metrics, backtest
> scripts, `CLAUDE.md`, and the "100X" strategy docs). It is not a contemporaneous
> record of decisions; dates refer to artifact timestamps, not meetings. A human
> who has the real history should correct it.

## The core question

Does the Real Relative Strength (RRS) day-trading strategy have an edge that,
net of honest costs, beats SPY buy-and-hold? On the evidence in this repo as of
2026-10-29: **no demonstrated edge, and it does not come close to SPY.**

## The evidence

### 1. Best optimized backtest is statistical noise
`data/optimization/optimization_2025-12-29.json` — 180 parameter combinations
tested, best result:
- Total return: **6.80%** (period ≈ 1 year, $25k)
- Win rate: **38.1%**
- Profit factor: **1.29**
- Max drawdown: 2.4%
- **Sharpe ratio: 0.11**
- 215 trades, ~4.4-day average hold

A Sharpe of 0.11 is indistinguishable from zero. And this is the **best of 180
combinations** — i.e., selected for in-sample performance, so the honest
out-of-sample expectation is lower, plausibly negative.

### 2. The strategy's own math says the edge is gone
`ACTIONABLE_100X_STRATEGY.md` computes the Kelly fraction at **≈ −0.02
(negative)** and states: "the current edge is marginal... Increasing position
size actually increases risk of ruin without improving returns." That document's
response was not to fix the edge but to (a) raise risk per trade to 3% anyway and
(b) pivot to **selling signals to other people** as a SaaS. Selling signals that
have no demonstrated edge is a governance/ethics red flag and does not satisfy
the mandate (profitable trading, net of costs, beating SPY).

### 3. Costs are not honestly modeled
- `scripts/run_walkforward_v2.py` (the 2-year, 3-way comparison whose "6.9%"
  headline appears in `CLAUDE.md`): **no slippage, no commission, no spread** —
  P&L is raw close-to-close.
- `scripts/run_backtest.py` defines `SLIPPAGE_PCT = 0.001` but **does not apply
  it to fills** ("the engines use close prices for entry, so we document the
  expected slippage impact rather than modifying data"); it reports an estimate
  separately.

At ~279 trades over 2 years, a 0.2% round-trip cost drag plausibly consumes
essentially the **entire** $1,716 gross profit of the best ("RDT filters")
config. The reported returns are therefore **gross, overstated, and may be
negative net of honest costs.**

### 4. No live track record
`data/signals/signal_metrics.json`: 880 scans, 120 signals, but only
**2 tracked outcomes** (1 target hit, 1 stop-out). There is effectively no
out-of-sample / paper-traded evidence either way.

### 5. Anomalous signal skew
Same file: **119 short signals vs. 1 long** out of 120. For a long-biased
momentum methodology (RDT), a 99% short skew is a red flag — either a
regime artifact of a narrow window or a scanner bias/bug worth investigating.

### 6. The benchmark the mandate requires beating
SPY total return (third-party trackers, nominal, dividends reinvested):
**2024 ≈ +24.9%**, **2025 ≈ +18%**. The bot's best *gross, overfit, cost-free*
backtest was ~6.8%/yr. The strategy does not beat, approach, or justify its
complexity against simply holding SPY.

## Conclusion

The RRS intraday strategy, as implemented and measured in this repo, has **no
edge demonstrable to an honest standard** and is far below SPY buy-and-hold. The
repository has drifted toward monetizing signals (a SaaS/"100X" direction)
instead of resolving the fundamental edge problem. Per the operator mandate, the
honest recommendation is **escalation / wind-down of the live-trading ambition**
until a genuine, cost-aware, out-of-sample edge is demonstrated — and in the
meantime, fixing the measurement (honest costs, no best-of-N overfitting,
walk-forward OOS, SPY benchmark baked into every report) so future claims are
trustworthy.
