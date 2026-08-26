# POST-MORTEM: Real Relative Strength (RRS) Strategy & the RDT Trading System

> Bootstrapped 2026-08-26 by the first operator run. This file is the honest
> history of why the bot is in its current state. It is deliberately blunt.

## What the system is

An autonomous day-trading system built around the r/RealDayTrading (RDT)
methodology. Its central signal is **Real Relative Strength**:

```
RRS = (Stock % Change - SPY % Change) / ATR%
```

Around that core it accreted, over many months: 4 sequential scanner filter
gates (SPY / SMA / VWAP / MTF), a VIX regime filter, a sector-RS filter, a
regime-adaptive parameter layer, an intermarket (Murphy) layer, 87 ML features,
a stacked ensemble (XGBoost + RF + LSTM), a drift detector, an options module,
a multi-broker failover layer, a Flask dashboard, a GraphQL API, a SaaS
onboarding/pricing surface, and thousands of lines of "100X wealth" strategy
docs.

## The uncomfortable bottom line

**None of that has been shown to beat buying and holding SPY.**

| Measure | Value | Source |
|---|---|---|
| Best backtested config, 2yr | +6.9% (~3.4%/yr) **gross** | CLAUDE.md walk-forward table |
| Same, after honest costs (est.) | ~+1.84% (~0.9%/yr) **net** | `backtesting/costs.py` illustration |
| SPY buy-and-hold, trailing 2yr | **+35.8% (~16.5%/yr)** | IBKR `get_price_history`, 2026-08-26 |
| Live paper account (MCP-connected) | **$5, zero positions** | IBKR `get_account_summary`, 2026-08-26 |
| Realized trade outcomes on record | **2** (1 win, 1 loss) | `data/signals/signal_metrics.json` |

The active strategy's *best* honest estimate is on the order of **1/18th** of
what a passive index fund returned over the same period, at far higher
operational complexity and risk.

## Why it ended up here — the failure modes

1. **Gross was mistaken for net.** Until 2026-08-26 not a single backtest engine
   modeled commission, slippage, or spread. A day-trading strategy taking
   hundreds of round-trips a year on a $25K account gives a large fraction of
   any edge back to costs. Every "it works" claim was built on an inflated
   number. This is the single most important error, because it made the whole
   evaluation loop unfalsifiable.

2. **Complexity substituted for edge.** Feature count and subsystem count grew
   without a corresponding, out-of-sample, net-of-cost improvement. The repo's
   own notes concede the ML is advisory-only (exit predictor at 43% accuracy,
   "barely above random"). Effort went into the machine, not the edge.

3. **No live track record.** The connected account is empty and only two trade
   outcomes were ever recorded. There is essentially no realized evidence — the
   system was elaborated in theory far beyond what it was ever tested in
   practice.

4. **Signal-generation anomalies went unresolved.** The aggregate counter
   (`signal_metrics.json`) shows 119 shorts vs 1 long during a strong bull
   market, while the raw history file shows 1,687 longs vs 299 shorts. The two
   disagree and outcome tracking is near-absent — a sign the measurement plumbing
   itself is not trustworthy.

5. **Hype documents crowded out honesty.** Multiple "100X" strategy files frame
   aspiration as if it were a plan. They are not evidence and should not be read
   as such.

## What the RRS premise gets right (steelman)

Relative strength is a real, well-documented factor. RDT's "trade with the
market, in the strongest/weakest names" is sound risk discipline. The gates
(don't fight SPY, respect VWAP, require multi-timeframe agreement) are
reasonable filters. The problem is not that the ideas are absurd — it is that,
as implemented and *honestly measured*, they do not clear the buy-and-hold bar.

## The forward question

Every future operator session should hold this question front-and-center:
**is there any version of this active strategy that beats SPY buy-and-hold,
net of honest costs, out-of-sample?** If yes — find and prove it, one
falsifiable hypothesis at a time. If repeated honest attempts say no, the
correct recommendation is to wind the active strategy down and hold the index.
Saying that clearly is the job, not a failure at it.
