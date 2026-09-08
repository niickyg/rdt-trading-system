# POST-MORTEM: The RRS Strategy and Why This Bot Is Where It Is

> Bootstrapped 2026-09-08 by the first operator instance. This file records,
> honestly and from evidence in the repository, why the RDT/RRS trading system
> is in its current state. It is deliberately unflattering where the evidence
> is unflattering. Future instances: extend it, do not sanitize it.

## The one-paragraph version

This is a large, well-engineered trading system built around Real Relative
Strength (RRS) and the r/RealDayTrading "market first" methodology. Enormous
effort went into filters, agents, ML, options, security hardening, and a SaaS
web layer. **None of that engineering changed the core fact: the strategy does
not beat SPY buy-and-hold, and its own edge math is marginal-to-negative.** The
system has generated thousands of signals but almost no tracked outcomes, so
for most of its life it has been unable to even *measure* whether it works. The
prior strategic document quietly conceded the trading edge is capped near ~7%/yr
and pivoted toward selling signals as a subscription business — a tell that the
trading edge itself was never established.

## What the evidence actually shows

### 1. The strategy trails the index it is supposed to beat
- **Bot's own best-case backtest** (`CLAUDE.md`, 2-year walk-forward, "Config C
  / RDT Filters"): **+6.9% total, ~3.4% annualized** over Feb 2024 – Nov 2025.
- **SPY buy-and-hold, same window** (real IBKR price history, conid 756733):
  508.08 → 685.99 = **+35.0% total, ~18.7% annualized** (before dividends).
- **The index beat the strategy by ~5.5x, at far lower risk** (no intraday
  exposure, no leverage, no operator). This is the central failure.

### 2. The edge is marginal-to-negative by the system's own math
From `ACTIONABLE_100X_STRATEGY.md` (prior analysis, retained in repo):
- Win rate ~38%, profit factor ~1.29, ~215 trades/yr at 1% risk.
- **Kelly criterion computes slightly negative (~−0.02).**
- Conclusion drawn there: "increasing position size actually increases risk of
  ruin without improving returns." Correct — and damning. A near-zero/negative
  edge cannot be sized into profitability.

### 3. The bot has barely measured itself
- `data/signals/signal_metrics.json`: 880 scans, 120 signals generated, but
  only **2 outcomes tracked** (1 target hit, 1 stop-out). `signal_history.json`
  holds 1,986 signals with **no outcome/P&L attached**.
- You cannot prove profitability — or improve toward it — without recording
  what happened to trades. This measurement gap is the most fixable and most
  important structural problem.

### 4. The ML layer is not carrying its weight
- Regime detector (`models/training_metrics.json`): silhouette score **−0.087**
  (worse than random clustering), Davies–Bouldin ~0.97, and **1030 of 1056**
  samples collapsed into a single "low_volatility" regime. The regime signal is
  effectively noise.
- Per `CLAUDE.md`'s own ML status: Exit Predictor 43.3% accuracy ("SKIP"),
  and "Rule-based filters provide all measurable improvement; ML is
  advisory-only." The team already knew the ML was not adding edge.

### 5. Effort went where it was measurable, not where the edge was
The repo shows deep investment in filters (SPY gate, SMA gate, VWAP, MTF, VIX,
sector, intermarket), options infrastructure, security remediation, and a
full SaaS front-end (landing/pricing/login/onboarding). This is real, competent
engineering. But the filters' own backtest lifts return from 3.3% → 6.9%
annualized-equivalent — still a fraction of the index. The activity was real;
the edge was not.

## Why this happened (the pattern to break)

1. **Methodology treated as the goal.** RDT-faithfulness became the success
   criterion instead of beating the index. A strategy can be perfectly faithful
   to a philosophy and still lose to buy-and-hold.
2. **Building instead of proving.** It is more satisfying to ship a new filter,
   agent, or dashboard than to run the cold experiment that might say "no edge."
   Measurement (outcome tracking) was neglected precisely because it was the one
   thing that could deliver bad news.
3. **The pivot tell.** When trading returns capped near 7%, the plan shifted to
   monetizing signals via subscriptions. That is a business decision, not a
   solution to "make the bot profitable" — and it implicitly concedes the
   trading edge was never there.
4. **Sizing mistaken for edge.** "Go to 3% risk for 3x returns" recurs in the
   docs. With negative Kelly this increases risk of ruin, not expectancy.

## What must be true before anyone claims this bot is profitable

1. **Outcome tracking works** and persists realized P&L for every signal/trade.
2. **An out-of-sample / walk-forward backtest, net of honest costs, beats SPY
   buy-and-hold** over the test window.
3. The result is **reproducible** by a fresh operator instance from data in the
   repo or via a sanctioned data channel.

Until all three hold, the honest status is: **not proven profitable; currently
trailing the index.**

## For the next instance
- Do not add another filter/agent/feature hoping it helps. First make the bot
  measurable, then run the cold experiment.
- If the cold experiment keeps saying "no edge," the mandate's §7 (escalate or
  wind down) is the correct, honorable outcome — not more tuning.
