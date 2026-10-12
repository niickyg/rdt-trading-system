# POST-MORTEM: RRS Strategy & the RDT Trading System

> **Note on provenance:** The operator routine expects this file to already exist as "the
> history of why the bot is in its current state." It did not exist in the repository as of
> 2026-10-12. This document was reconstructed by the first operator instance from the repo's
> own artifacts: `CLAUDE.md`, `ACTIONABLE_100X_STRATEGY.md`, `WEALTH_STRATEGY_100X.md`,
> `DEPLOYMENT_SUMMARY.md`, `README.md`, and the git history. Where a claim could not be
> independently reproduced, it is labeled **[unverified]**. Future instances should correct
> this record as they reproduce or refute its claims.

---

## 1. What the system is

An autonomous, agent-based day/swing-trading bot built around **Real Relative Strength (RRS)**:

```
RRS = (Stock % Change − SPY % Change) / ATR%
```

Long when a stock is strongly outperforming SPY (RRS > ~2), short when underperforming.
Layered on top: a stack of filters (SPY regime gate, 50/200 SMA gate, VWAP gate, lightweight
multi-timeframe alignment), plus VIX / sector / intermarket / regime adjustments, an ML
ensemble (advisory only), and an options module. Execution via IBKR / paper broker.

## 2. The core problem — the edge is marginal at best

The repository's own documents converge on an uncomfortable truth: **the RRS strategy has
little or no durable edge after costs.** From `ACTIONABLE_100X_STRATEGY.md`:

- Win rate ~38%, profit factor ~1.29, ~215 trades/yr, ~$1,700 net/yr on $25k (~6.8%).
- **Kelly criterion ≈ −0.02 (negative).** The document itself states: "the current edge is
  marginal ... Increasing position size actually increases risk of ruin without improving
  returns."

This is the central finding. A near-zero/negative Kelly fraction means position sizing,
leverage, and risk tuning cannot create profit — there is no edge to size into.

## 3. The numbers do not agree with each other

Different docs report materially different performance for "the" strategy. All are
**backtest-only**; none is realized paper or live P&L:

| Source | Win rate | Profit factor | Annualized | Notes |
|---|---|---|---|---|
| `CLAUDE.md` walk-forward V2 (best = "RDT filters") | 49.5% | 1.24 | **3.4%** | 2yr, 6 windows, $25k |
| `ACTIONABLE_100X_STRATEGY.md` | 38% | 1.29 | ~6.8% | "current optimization" |
| `DEPLOYMENT_SUMMARY.md` (aggressive profile) | 50.2% | 1.35 | 6.84% | 365-day backtest |

Inconsistency of this size across the project's own authoritative docs means **none of the
numbers can be trusted until independently reproduced.** This is itself a finding.

## 4. The benchmark comparison is damning

The mission benchmark is **SPY buy-and-hold over the same window.** Over the backtest windows
in question (roughly 2024–2025), SPY returned on the order of ~25% in 2024 alone and strong
double digits across the 2-year window. Every documented configuration of this bot
(3.4%–6.9% annualized, back-tested, before honest slippage) **underperforms simply holding
SPY** — while taking on single-name, intraday, and execution risk the index does not. [SPY
figures to be reproduced empirically by a future instance.]

## 5. The drift away from the mission

Faced with a marginal edge, the project did not wind down or escalate. Instead the artifacts
show a pivot:

- `ACTIONABLE_100X_STRATEGY.md`, `WEALTH_STRATEGY_100X.md`, `QUICK_START_100X.md`,
  `WEALTH_OPTIMIZATION.md`: a plan to hit "100% annual return / $25k→$50k" that openly
  concedes **"trading alone cannot achieve 100% returns"** and relies on **selling a signal
  service** ($49–$499/mo subscriptions) for 40–60% of the goal.
- `DEPLOYMENT_SUMMARY.md` (2025-12-29): deployed an **aggressive** profile (3% risk/trade,
  RRS threshold 1.75, "relaxed criteria") — i.e. sizing harder into a negative-Kelly edge,
  the exact mistake the math warns against.
- Git log: the most recent commit is **"SaaS product overhaul — toast system, skeletons,
  animations, landing/pricing/login/register pages, onboarding."** Effort is flowing into
  product/marketing, not into establishing a trading edge.

Selling signals to others is not the mission, and building SaaS polish does not make the
trading profitable. This is the drift the MANDATE's anti-drift note exists to counter.

## 6. What has NOT been done

- **No realized paper-trading P&L has ever been recorded** in the repo. Every performance
  figure is a backtest. The bot's actual behavior over a real month is unknown here.
- **No honest, reproduced, cost-inclusive comparison to SPY buy-and-hold** exists in writing.
- **No out-of-sample confirmation** that any filter stack generalizes beyond the window it
  was tuned on.

## 7. The honest standing verdict

As of 2026-10-12, on the available evidence: **the RRS strategy and its filter/ML/options
superstructure have not been shown to beat SPY buy-and-hold, and the core edge is documented
as marginal-to-negative.** The responsible next steps are, in order:
1. Reproduce the backtest honestly (with costs, out-of-sample) and compare to SPY. Either
   the edge survives or it does not.
2. If it does not survive: recommend wind-down of live ambitions and, at most, a minimal
   paper-only experiment to search for a *different* edge — or escalate to the human owner.
3. Stop pouring effort into SaaS/product and leverage until step 1 produces a positive verdict.

This file should be updated as evidence accumulates.
