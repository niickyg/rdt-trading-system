# POST-MORTEM: The RRS Strategy and Why This Bot Is Where It Is

> Bootstrapped 2026-07-29 by the first operator instance. This file did not
> previously exist (the operator prompt referenced it, but it was never
> committed). It is the honest history the operator role is supposed to preserve.
> Future instances: append, correct, and date your additions.

---

## The premise

The system implements the r/RealDayTrading (RDT) methodology: trade individual
stocks that show **Real Relative Strength / Weakness (RRS)** versus SPY, filtered
through a "market first" stack of gates (SPY trend, 50/200 SMA, VWAP, multi-
timeframe alignment, VIX regime, sector RS, intermarket). The thesis: buy the
strongest stocks in an up market, short the weakest in a down market, and let
disciplined filters keep you out of low-quality setups.

The engineering is genuinely substantial: agent architecture, ML ensemble,
options module, multi-broker execution, walk-forward harness, a large security
and reliability audit trail. The problem was never the engineering.

## The problem: the edge was never there

The repo's own documents tell the story if you read them net of optimism.

**1. The best backtest is small — and gross.**
`CLAUDE.md` and `ACTIONABLE_100X_STRATEGY.md` report the best walk-forward
configuration ("RDT Filters", Config C) at **+$1,716 / +6.9% over 2 years**,
≈ **3.4% annualized**. Critically, the backtest engines
(`backtesting/engine_enhanced.py`) model **no commissions and no slippage**, and
the walk-forward script **never compared against SPY buy-and-hold**. So even the
6.9% is an overstatement of the real, investable result.

**2. SPY buy-and-hold crushed it.**
Verified live via the IBKR MCP on 2026-07-29: SPY went **$550.81 → $740.86** over
the trailing two years, **+34.5% total, ~16%/yr price-only (~17.5% with
dividends)**. On $25,000 that is roughly **+$8,600** of profit for doing nothing
but holding — about **5x the bot's best gross backtest**, with no single-name
blow-up risk and no infrastructure to babysit.

**3. The math was flagged internally and then rationalized away.**
`ACTIONABLE_100X_STRATEGY.md` computes the strategy's **Kelly criterion as
slightly negative** (win rate ~38%, profit factor ~1.29) and states plainly:
"the current edge is marginal… increasing position size actually increases risk
of ruin without improving returns." That is the whole ballgame. A marginal-to-
negative edge does not become profitable through leverage, more symbols, or
tighter stops. It becomes profitable only if the edge itself is real, and the
honest evidence says it is not — at least not enough to beat a passive index.

## The pivot that wasn't allowed

Faced with a capped ~7% ceiling, the documented plan (`ACTIONABLE_100X_STRATEGY.md`,
`WEALTH_STRATEGY_100X.md`) pivoted the definition of success from *trading
profit* to *selling the signals*: a subscription "signal service," API access,
tiered pricing, a landing page. The most recent git history confirms where the
effort actually went — commits like "SaaS product overhaul — toast system,
skeletons, animations, landing/pricing/login/register pages, onboarding."

This is a real business idea, but it is **not the mandate**. The mandate is
actual positive trading P&L that beats SPY. Selling signals whose backtested edge
is marginal-to-negative, net of costs and versus SPY, is not making the bot
profitable — it is monetizing the *appearance* of an edge. The operator mandate
explicitly rules this out of scope (MANDATE §3, §0).

## What is actually true (standing conclusions)

- The RRS strategy, as built and backtested, has **no demonstrated tradeable
  edge that beats SPY buy-and-hold net of honest costs.**
- The backtests that suggested otherwise were **gross of costs and un-benchmarked.**
  (Fixed 2026-07-29: `backtesting/benchmark.py` + walk-forward now reports the
  net-of-cost SPY comparison.)
- The connected paper account currently shows **$5**, not $25K — live results are
  presently unverifiable from the operator's vantage point.

## What would change the conclusion

To overturn "no edge," a future instance must produce, per MANDATE §2:
a **net-of-cost, SPY-benchmarked, out-of-sample** return that beats SPY over a
meaningful window — ideally validated on intraday bars (since the live system's
VWAP and first-hour gates cannot be simulated on daily data, the daily backtest
may *understate* or *overstate* the real strategy; that ambiguity is itself a
reason the current evidence is weak). Absent that, the honest recommendation
trends toward wind-down (MANDATE §6).

## Why this file exists

So that no future instance wastes a session re-discovering the negative Kelly, or
re-building the SaaS, or quoting the 6.9% as if it were investable. Start from the
truth and push on the one question that matters: is there a real, net-of-cost
edge over SPY? Everything else is secondary.
