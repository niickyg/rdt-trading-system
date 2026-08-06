# POST-MORTEM: The RRS Strategy — Why the Bot Is Where It Is

> **Status of this document:** Reconstructed on 2026-08-06 by the first operator instance
> from artifacts committed in this repository. No prior post-mortem existed. Where a claim
> comes from a repo file, that file is cited. Where something is inferred, it is labelled
> INFERENCE. Future instances should correct this document as better evidence appears.

---

## 1. What the system is

An autonomous, agent-based day-trading system built around **Real Relative Strength (RRS)** —
the r/RealDayTrading idea of buying stocks that are strong *relative to SPY* in a strong market
and shorting relative-weak stocks in a weak market. Around that core sits a large amount of
scaffolding: multi-gate scanner filters (SPY gate, 50/200 SMA, VWAP, multi-timeframe), VIX and
sector and intermarket overlays, an options module, an ML ensemble, a Flask dashboard, and a
paper/IBKR execution path. (See `CLAUDE.md` for the full architecture.)

The engineering surface area is large. **The evidence of edge is small.** That mismatch is the
central fact of this post-mortem.

## 2. What the numbers actually say (from the repo's own files)

### 2a. The strategy's own math admits a marginal-to-negative edge
`ACTIONABLE_100X_STRATEGY.md` (committed) states, verbatim in substance:
- Best backtest: ~6.8% annual return (~$1,700 on $25K).
- Win rate ~38%, profit factor ~1.29 at 1% risk/trade.
- A Kelly-criterion calculation that comes out **negative** (`kelly ≈ -0.02`), with the doc's
  own words: *"the current edge is marginal."*
- Its path to the "100%/100X" goal is **~50% trading, ~50% selling a signal service**. In other
  words, the authors could not close the gap with trading edge alone and reached for
  subscription revenue. (The operator MANDATE §2.8 forbids counting that toward the mission.)

### 2b. The "aggressive" deployment raised risk, not edge
`DEPLOYMENT_SUMMARY.md` (dated 2025-12-29) shows the response to weak returns was to push risk
per trade from 1% → 3%, max daily loss 2% → 6%, positions 5 → 10, and loosen the RRS threshold
2.0 → 1.75. Expected return still only ~6.8%/yr. **INFERENCE:** turning up risk on a marginal
edge raises variance and risk-of-ruin without materially raising expected return — exactly what
the negative-Kelly result predicts.

### 2c. There is almost no live track record
`data/signals/signal_metrics.json`:
- 880 scans, 120 emitted signals, 802 scans with no signal.
- **Only 2 recorded outcomes total: 1 target hit, 1 stop out.**
So essentially every performance number in the repo is *backtested or theoretical*, not realized.

### 2d. A month of signals was emitted but never scored
`data/signals/signal_history.json` contains **1,986 emitted signals** spanning
2026-02-03 → 2026-03-05, each with entry/stop/target/direction — but **no outcome field**. The
system generated a large evidence set and then threw away the answer key. Median target R:R is
2.0, so breakeven win rate is ~33%; the doc's own ~38% is barely above that before costs.

### 2e. Benchmark gap
The mission is to beat SPY buy-and-hold. A ~6.8% *backtested* annual return does not beat SPY
over 2024–2025 (SPY returned substantially more). **The strategy, at its documented best, loses
to doing nothing but holding the index.**

## 3. How it got here (INFERENCE, from commit history and docs)

Commit history (`git log`) shows the recent trajectory has been **product/plumbing**, not edge:
a "SaaS product overhaul" (toasts, skeletons, landing/pricing/login pages, onboarding), new
dashboard pages, runtime-stability fixes, and swapping quote sources — on top of an already
enormous feature set. The gravitational pull has been toward *building more system* and
*productizing*, while the one question that decides everything — does the RRS signal have
positive expectancy net of costs? — was never definitively answered with realized or
forward-tested results.

**This is the trap the operator MANDATE §6 exists to break:** adding filters, ML, and UI to a
strategy whose base edge is unmeasured feels like progress but isn't.

## 4. The open question that actually matters

> Does the RRS momentum signal, as emitted in `signal_history.json`, produce positive
> expectancy on **real forward prices**, net of realistic costs — and does that beat SPY?

Everything else (which filter, which ML model, which dashboard) is downstream of this. Until it
is answered with real data, the honest status is **NO EDGE DEMONSTRATED**.

## 5. What would change the verdict

- A forward-return / walk-forward test on the 1,986 historical signals against **actual** OHLC
  prices, showing expectancy clearly positive after costs — and a matched-period SPY comparison
  showing the strategy wins. (The first operator instance began exactly this; see the journal.)
- Or a realized paper-trading track record of enough trades to be statistically meaningful
  (hundreds), with the same net-of-cost, beat-SPY test.

Absent that, the mandate's guidance applies: measure honestly, remove what doesn't work, and be
willing to recommend wind-down.
