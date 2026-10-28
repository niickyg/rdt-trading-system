# POST-MORTEM: The RRS Strategy — Why The Bot Is Where It Is

> **STATUS: RECONSTRUCTED (2026-10-28).** The scheduled operator task references a
> `POST_MORTEM_RRS.md` that **did not exist** in the repository. This file was reconstructed by
> operator run #001 from the evidence present in the codebase (`CLAUDE.md`,
> `ACTIONABLE_100X_STRATEGY.md`, `data/signals/*`, and live SPY data). Treat the history below as
> *inferred from artifacts*, not as a first-hand record. Human: correct anything wrong.

## The core finding

The Real Relative Strength (RRS) momentum strategy, as built and tuned in this repo, has **no
demonstrated edge over simply buying and holding SPY** — in fact it underperforms SPY
buy-and-hold by roughly **5x** on its own best backtest window.

## The numbers (from the repo's own artifacts)

### The bot's best documented backtest — `CLAUDE.md`, "Walk-Forward Backtest Results (2 Years)"
3-way comparison, $25K, Feb 2024–Nov 2025, 6 quarterly windows. Best variant ("RDT Filters"):

| Metric | Best variant (C) |
|---|---|
| Total Return (2yr) | **$1,716 (6.9%)** |
| Annualized | **3.4%** |
| Win Rate | 49.5% |
| Profit Factor | 1.24 |
| Trades | 279 |

### The strategy author's own reality check — `ACTIONABLE_100X_STRATEGY.md`
- "Best backtest: **6.8% annual**" (inconsistent with CLAUDE.md's 3.4% annualized / 6.9% total
  — one of these is wrong; both are far below SPY either way).
- Profit factor 1.29, **win rate 38%**.
- **Kelly Criterion ≈ −0.02 (NEGATIVE).** The author's words: "the current edge is marginal.
  Increasing position size actually increases risk of ruin without improving returns."

### The benchmark the mission actually cares about — SPY buy-and-hold (IBKR monthly closes)
- 2024-10-29 → 2025-11-03: **+20.2%** (≈ the documented backtest window).
- 2024-10-29 → 2026-10-01: **+35.4% over 2yr ≈ +16.3% annualized.**

**Bottom line:** best bot backtest ≈ 3.4%/yr vs SPY ≈ 16–20%/yr over the same period. The
strategy loses decisively to doing nothing but holding the index.

## Why (mechanism, inferred)

1. **Marginal/negative Kelly.** A 1.2–1.3 profit factor with a sub-50% win rate at ~1% risk
   mathematically caps returns in the low single digits and leaves ~no edge after costs.
2. **Signal-limited, not capital-limited.** "98% of raw signals are filtered out" (CLAUDE.md).
   Heavy RDT filtering raises per-trade quality slightly but starves the strategy of
   opportunity; the surviving edge is too thin to compound meaningfully.
3. **Costs not honestly modeled in the headline number.** The backtest P&L is small enough that
   realistic commission + slippage could erase most or all of it. (To verify.)
4. **The pivot tell.** `ACTIONABLE_100X_STRATEGY.md` concludes trading "alone cannot achieve"
   the goal and routes >half the target to **selling signals to other people**. When the plan to
   make a trading bot profitable depends on subscription revenue rather than the trades, the
   trading edge has been conceded.

## Operational state (as of 2026-10-28)

- `data/signals/signal_metrics.json`: 880 scans, 120 signals (119 short / 1 long — lopsided),
  **only 2 completed outcomes ever** (1 target, 1 stop). There is **no statistically meaningful
  live/paper track record.**
- Last scan recorded: **2026-03-05**. Today: **2026-10-28**. The live scanner appears to have
  been **dormant for ~7.5 months.**

## Implication for the operator loop

Per the mandate, "if the evidence keeps saying no strategy works, say so and recommend
escalation or wind-down." Run #001 is the first honest look, so this is a **first flag, not a
final verdict** — but the current, documented evidence says the RRS bot does **not** meet the
mission bar (beat SPY buy-and-hold). See `entries/2026-10-28-run-001.md` for the decision tree.
