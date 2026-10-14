# POST-MORTEM: The RRS Strategy and Why This Bot Is Where It Is

> **Status: BOOTSTRAPPED 2026-10-14 by the operator agent.** The scheduled
> operator task referenced this file as required reading, but it did not exist
> in the repository. This is a first reconstruction from the repo's own
> documentation (`WEALTH_OPTIMIZATION.md`, `WEALTH_STRATEGY_100X.md`,
> `ACTIONABLE_100X_STRATEGY.md`, `CLAUDE.md`) plus fresh evidence generated on
> 2026-10-14. Future instances should extend it, not trust it blindly.

## The one-paragraph version

The system implements the r/RealDayTrading methodology (Real Relative Strength
momentum, "market first" filter gates, options overlay, ML advisory layer). An
enormous amount of engineering has gone into it. But every honest measurement —
including a fresh, cost-aware, benchmarked backtest run on 2026-10-14 — says the
**trading strategy does not beat buying and holding SPY.** Net of modest costs
it returns low-single-digit percent per year while SPY returned ~15%/yr over the
same window. The repo's headline results looked better only because they
**ignored transaction costs and never compared to a benchmark.**

## Timeline of the documented self-assessments

The repo's own strategy docs tell the story, each one quietly admitting the
previous target was missed:

1. **`WEALTH_OPTIMIZATION.md`** — baseline "2.85% annual return ($712)",
   targeting 15-25%. Added a parameter-optimization engine, bigger watchlist.
2. **`WEALTH_STRATEGY_100X.md` / `ACTIONABLE_100X_STRATEGY.md`** — "current
   state 6.84% annual return ($1,711)", targeting **100%** ($25k profit on
   $25k). Crucially, these docs concede: *"Pure trading at reasonable risk
   levels caps at ~$8,000-15,000/year. The signal service is essential to
   reaching the $25,000 profit target."* i.e. the plan to hit the goal relied
   on **selling signals / education**, not on trading alpha.
3. **`CLAUDE.md`** walk-forward table — "RDT Filters" best at **+6.9% over 2
   years (3.4% annualized)**, win rate 49.5%, profit factor 1.24. Presented as
   the success case. No costs. No benchmark.

## The two fatal omissions (found 2026-10-14)

1. **No transaction costs anywhere in `backtesting/`.** `grep -rniE
   'commission|slippage|fee|spread'` over `backtesting/*.py` returns nothing.
   The engine does `capital -= shares*price` on entry and `capital +=
   shares*price + pnl` on exit — zero friction. For a strategy taking
   200-280 trades, this materially inflates results.

2. **No SPY buy-and-hold benchmark.** The headline numbers were never placed
   next to the obvious alternative of doing nothing. The mandate's definition
   of success *is* beating that benchmark.

## Fresh honest evidence (2026-10-14)

`scripts/honest_benchmark_backtest.py` runs one contiguous, compounding backtest
over the trailing ~2 years with realistic round-trip costs (IBKR-like $0.005/sh
+ 5 bps/side slippage) and a SPY buy-and-hold benchmark over identical dates.

Window 2024-12-06 → 2026-10-13 (463 trading days, ~1.85y), $25k:

| Config | Gross | Costs | **Net total** | **Net annualized** |
|---|---|---|---|---|
| Baseline (no filters) | +7.20% | $807 | **+3.97%** | **+2.13%** |
| RDT-filtered ("best") | +10.12% | $1,092 | **+5.75%** | **+3.07%** |
| **SPY buy-and-hold** | — | — | **+30.96%** | **+15.69%** |

**The RDT-filtered strategy underperformed SPY buy-and-hold by ~25 percentage
points.** Costs alone cut the strategy's gross return roughly in half.

### Honest caveats (so the next instance doesn't overclaim)

- The strategy holds ≤8 risk-sized positions, so it is only partially invested;
  SPY buy-and-hold is fully invested in a strong bull market. Part of the gap is
  **exposure/beta, not stock-picking skill.** A fair fight would also test the
  strategy at matched exposure or in a flat/down market.
- The cost model is deliberately generous (scale-out fills not charged extra).
  Real costs are likely higher, not lower.
- This is one contiguous window in a bull regime. It is evidence, not proof for
  all regimes. But it directly answers the mandate's question for the period
  that matters most recently.

## The uncomfortable conclusion

As a tool for the stated mission — grow $25k by beating SPY buy-and-hold net of
costs — the current strategy **fails**, and has failed every honest
self-assessment in the repo. The "100X" documents are not a trading plan; they
are a plan to sell a trading plan. Future operator instances should resist
adding complexity on top of a negative edge and instead either (a) find a
regime/exposure in which the edge is real and net-positive vs benchmark, or
(b) recommend wind-down of live-style trading in favor of the benchmark.
