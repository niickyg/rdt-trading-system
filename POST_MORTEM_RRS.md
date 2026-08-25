# POST-MORTEM: The RRS Strategy and How the Bot Got Here

> Reconstructed 2026-08-25 by the first autonomous-operator instance from git history,
> in-repo documents, the live IBKR paper account, and the persisted signal data.
> No prior post-mortem existed. This is the honest history, including the parts the
> project's own marketing documents obscured.

---

## TL;DR

The bot trades a "Real Relative Strength" (RRS) momentum strategy derived from the
r/RealDayTrading methodology. **By every independent measure available — in-sample
optimization, walk-forward backtest, the project's own Kelly analysis, and the live
paper account — the strategy has no positive edge.** The live paper account is down
**-61.5% time-weighted** over its ~6-month life while SPY rose **+11.3%** over the same
window. The account has collapsed to **$5** and has been dormant (zero trades) for ~2
months. Over that history the project's development effort visibly migrated away from
fixing the trading edge and toward building a SaaS/"signal service" product to sell the
signals instead — a tell that the trading itself never worked.

## What RRS is

```
RRS = (Stock % Change - SPY % Change) / ATR%
```
A momentum/relative-strength measure. RRS > ~2 → long candidate, < ~-2 → short. The
system layers many filters on top (SPY gate, 50/200 SMA, VWAP, multi-timeframe, VIX
regime, sector RS, intermarket analysis, news sentiment, an ML ensemble). The thesis:
filtering to only the strongest relative-strength names in a supportive market produces
a tradable edge.

## The evidence, four independent ways — all negative

### 1. In-sample parameter optimization (`data/optimization/optimization_2025-12-29.json`)
Best of 180 parameter combinations, **optimized on its own data** (i.e. the most
favorable, overfit-prone number available):
- Total return: **6.8%** (period), Win rate: **38%**, Profit factor: **1.29**
- **Sharpe ratio: 0.11** — statistically indistinguishable from noise.
An optimized in-sample Sharpe of 0.11 is a strategy with essentially no edge.

### 2. Walk-forward backtest (per `CLAUDE.md`, 2 years, 6 quarterly windows)
Best configuration ("RDT filters"): **6.9% total over 2 years = ~3.4% annualized**,
win rate 49.5%, profit factor 1.24. Over 2024–2025 SPY compounded far faster. The
"best" result still loses badly to buy-and-hold.

### 3. The project's own Kelly analysis (`ACTIONABLE_100X_STRATEGY.md`)
Quoted verbatim from the repo's own strategy document:
> "kelly = -0.02 # NEGATIVE! ... The Kelly Criterion is slightly negative, meaning the
> current edge is marginal. Increasing position size actually increases risk of ruin
> without improving returns."
The authors knew the edge was non-positive and wrote it down.

### 4. The live paper account (IBKR MCP, ground truth) — the decisive one
Time-weighted return (deposits/withdrawals stripped out), Feb 25 → Aug 25, 2026:

| | Bot (live paper acct) | SPY buy-and-hold |
|---|---|---|
| TWR over window | **-61.5%** | **+11.3%** |
| End state | $5, dormant | rising |

NAV path (from `get_pa_performance_all_periods`): started ~$50 → bled to $21 by early
March → topped up to ~$521 → drifted down to $477 through May/June → **collapsed to $5
at end of June and has been flat at $5 ever since**, with **zero trades year-to-date**.
The catastrophic drawdown is exactly what the negative-Kelly analysis predicted would
happen once the strategy was deployed at the AGGRESSIVE 3%-per-trade sizing recorded in
`DEPLOYMENT_SUMMARY.md` (2025-12-29): oversizing a negative edge produces ruin.

## The documentation-vs-reality gap (itself a finding)

- `CLAUDE.md` describes a **$25,000** paper account (`DUP995654`) "funded Feb 2026."
  The **connected live IBKR account is a ~$50-start micro account now worth $5.** The
  documentation does not track the reality of the account being traded.
- The repo contains `WEALTH_STRATEGY_100X.md`, `QUICK_START_100X.md`,
  `ACTIONABLE_100X_STRATEGY.md` — "100x" / "$25K→$50K / 100% annual return" framing —
  while the same documents' own math shows a negative Kelly and ~7% best-case return.
  The aspiration in the docs is ~10x the strategy's own demonstrated ceiling.

## The revealing pivot

Git history's most recent substantive commits are product/SaaS work:
`feat: SaaS product overhaul — toast system, skeletons, animations, landing/pricing/
login/register pages, onboarding`, `feat: add AI Confidence and Trading Journal
dashboard pages`. `ACTIONABLE_100X_STRATEGY.md` states the plan to reach its target is
**40–60% from "Signal Service Revenue"** — i.e. selling subscriptions to the signals
rather than profiting by trading them. When a trading project starts trying to sell its
signals instead of trading them, that is the market telling you the signals do not have
tradable edge.

## Why parameter tuning will not fix this

The failure is not a badly-tuned stop or threshold. Across 180 optimized parameter sets
the best Sharpe was 0.11 and Kelly was negative. The relative-strength momentum edge, if
it ever existed in this form, is arbitraged away at the scan/entry granularity this
system uses, and the stack of filters trades away most signals (98% filtered) without
producing a profitable residual. **No knob in the current design turns a negative edge
positive.** That is the core lesson.

## What an honest path forward looks like

1. **Stop adding machinery.** More features/agents/filters have not and will not create
   edge; they add surface area and the illusion of progress.
2. **Measure honestly or not at all.** Any future claim of improvement must be
   out-of-sample and net of commissions, slippage, and spread, benchmarked to SPY over
   the same window (see `MANDATE.md` §2–3).
3. **Be willing to conclude "no edge."** The mandate's decision gate exists precisely
   for this. Recommending wind-down here would be the correct, evidence-based outcome,
   not a failure.
4. **If anyone still wants to try:** the only intellectually honest experiments left are
   structurally different from RRS-at-this-granularity (e.g. a genuinely different
   holding period, instrument, or signal with a mechanism for *why* an edge would
   persist) — and each must clear the cost-aware, out-of-sample, beat-SPY bar before
   any capital or complexity is committed.

---
*This document is append-only history. Correct it with dated notes; do not erase it.*
