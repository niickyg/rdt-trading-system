# POST-MORTEM: The RRS Strategy and How the Bot Got Here

> Bootstrapped 2026-09-17 by the first autonomous operator instance. This file did
> not previously exist; the scheduled prompt referenced it as if it did. It
> reconstructs the history from the repository's own documents
> (`ACTIONABLE_100X_STRATEGY.md`, `DEPLOYMENT_SUMMARY.md`, `CLAUDE.md`, commit log)
> and from a fresh backtest run this session. Future instances should append, not
> rewrite.

---

## The core problem, stated plainly

The bot is built on **Real Relative Strength (RRS)**: go long stocks outperforming
SPY (RRS > threshold), short those underperforming, normalized by ATR. Layered on
top are filter gates (SPY trend gate, 50/200 SMA, VWAP, multi-timeframe), VIX
regime sizing, sector RS, regime-adaptive thresholds, intermarket analysis, and an
advisory ML stack.

**The strategy does not have a demonstrated edge over buy-and-hold.** Every honest
number the project has produced says the same thing:

| Source | Metric | Value |
|--------|--------|-------|
| `ACTIONABLE_100X_STRATEGY.md` | Profit factor | 1.29 |
| `ACTIONABLE_100X_STRATEGY.md` | Win rate | 38% |
| `ACTIONABLE_100X_STRATEGY.md` | **Kelly criterion** | **−0.02 (NEGATIVE)** |
| `DEPLOYMENT_SUMMARY.md` | Best annual return | 6.84% (~$1,711 on $25k) |
| `CLAUDE.md` walk-forward | Best annualized | 3.4% (RDT filters) |

A negative Kelly means the edge is, at best, marginal and, at worst, nonexistent.
The project's own documents admit this: *"the current edge is marginal. Increasing
position size actually increases risk of ruin without improving returns."*

## Why the returns don't scale (the "signal-limited" finding)

The team tested Conservative → Very Aggressive risk profiles and found returns were
**nearly identical** across all of them. Their conclusion: the strategy is
"signal-limited, not capital-limited" — only ~240 quality setups/year, and tight
0.75×ATR stops naturally cap position sizes. This is presented as a feature. It is
actually a symptom: **if 3× the risk produces the same return, the per-trade edge
is too small to compound.** That is what a marginal-edge system looks like.

## The benchmark the project never centered

The mission benchmark is **SPY buy-and-hold.** The project optimized profit factor,
win rate, and drawdown — but rarely asked "did we beat just holding SPY?" A fresh
backtest this session (see 2026-09-17 journal entry) makes the gap concrete: over
the same ~2-year window the walk-forward trades, **SPY buy-and-hold returned roughly
+47% (~$11,800 on $25k)** while the RRS strategy's documented best is low-single-digit
annualized. The strategy is not just failing to beat the benchmark — it is being
lapped by it.

## The drift into revenue theater

Faced with a strategy that caps around 6-7%, `ACTIONABLE_100X_STRATEGY.md` pivoted
the "path to 100% returns" toward **selling a signal service** ($49–499/mo tiers,
API access, "even at 15% capture = $9,800/year"). The recent commit history
confirms the drift: *"SaaS product overhaul — toast system, skeletons, animations,
landing/pricing/login/register pages, onboarding."*

This is the most important thing for a future instance to understand: **the effort
went into monetizing signals instead of finding an edge that would make the signals
worth buying.** Selling a negative-Kelly signal to subscribers is selling something
that loses money. Per the mandate, this is out of scope and off-mission.

## What has NOT been honestly ruled out (candidate next steps)

The strategy has been *tuned* extensively but a few structurally-different
hypotheses remain under-tested with honest, out-of-sample, cost-inclusive backtests
and a SPY benchmark side-by-side:

1. **Regime-conditional participation.** The one robust fact in the data is that SPY
   went up ~47%. A system that simply stays long/flat with the SPY trend (the SPY
   gate, taken seriously as a *participation* rule rather than a *filter*) might
   capture more of that than stock-picking on RRS does. Worth measuring against b&h.
2. **Honest cost modeling.** Confirm whether the backtest engine models slippage +
   spread. A 1.29 profit factor can flip to <1.0 once realistic costs hit ~240
   trades/year. If costs aren't modeled, the "edge" may already be gone.
3. **Does RRS have *any* forward IC?** The `research/` framework computes factor
   information coefficients. Before tuning gates further, confirm the base signal
   has predictive power at all. If IC ≈ 0, no amount of gating helps.

## The uncomfortable conclusion

Multiple rounds of optimization have moved the strategy from ~2.8% to ~6.8% in
tuned backtests, but never past a marginal/negative real edge, and never near the
SPY benchmark. The honest hypothesis a future instance must be willing to confirm
or refute: **this strategy may have no edge, and the correct outcome may be to
report that and recommend wind-down rather than continue tuning.** The mandate
explicitly permits — and prefers — that honesty over manufactured progress.
