# POST-MORTEM: The RRS / RDT Strategy

> Bootstrapped 2026-09-10 by the first operator instance. This file did not exist;
> the scheduled operator task expected it. It records, honestly, why the bot is where
> it is, based on freshly reproduced measurements (not on the marketing prose scattered
> through the repo). Future instances: update this as the story develops.

---

## The one-sentence summary

The RRS / RDT strategy, as built, produces a small positive **gross** return that is
**dwarfed by simply buying and holding SPY** over the same period, and its structural
choices (tiny per-trade position sizing, daily-bar approximation of an intraday method,
zero cost modeling in the backtest) mean it has **never been shown to beat the index
net of honest costs.**

---

## What the numbers actually say (reproduced 2026-09-10)

Source: `scripts/run_walkforward_v2.py` (config C = "RDT Filters") + the new
`scripts/net_cost_analysis.py`, run on 2 years of daily yfinance data, $25,000 capital,
walk-forward windows spanning **2024-05-15 → 2026-05-28 (~2.03 yr)**.

| Strategy | Trades | Win rate | Profit factor | Total return | Annualized |
|---|---|---|---|---|---|
| A) Baseline (no filters) | 233 | 48.1% | 1.24 | +6.20% | ~3.1% |
| B) Old filters | 421 | 48.5% | 1.24 | +9.08% | ~4.5% |
| **C) RDT filters** | **274** | **48.9%** | **1.34** | **+9.85%** | **~4.9%** |

**All three columns are GROSS** — `backtesting/engine_enhanced.py` models **zero**
commission, slippage, or bid/ask spread. Entries fill at the daily close; exits fill at
the exact stop/target price.

Applying a conservative retail cost model (IBKR-style $0.005/share, $1 min, + 1–5 bps
one-way slippage/spread; 2 fills per trade, which *understates* cost because the engine
also scales out):

| Config C, net of costs | Total return | Annualized | $25k becomes |
|---|---|---|---|
| @ 1 bps one-way | +7.2% | +3.5% | $26,807 |
| @ 3 bps one-way | +6.4% | +3.1% | $26,590 |
| @ 5 bps one-way | +5.5% | +2.7% | $26,373 |

### The benchmark that matters

**SPY buy & hold, identical span (2024-05-15 → 2026-05-28): +45.9% total,
+20.4% annualized, $25,000 → $36,476.**

The bot, net of costs, returned roughly **+2.7% to +3.5% annualized** — about
**one-sixth of the index**, while adding complexity, execution risk, and screen time.
In dollar terms over ~2 years: bot ≈ **+$1,400–1,800 net** vs SPY **+$11,476**.

---

## Why it underperforms (root causes)

1. **The edge is marginal to begin with.** Win rate hovers at ~48–49% with profit factor
   ~1.2–1.34. `ACTIONABLE_100X_STRATEGY.md` itself computes a **negative Kelly** for the
   base strategy. A near-coin-flip with a slim payoff ratio cannot compound into
   index-beating returns; it can only nibble.

2. **The market it trades in went nearly straight up.** 2024–2026 was a strong bull
   (SPY +46% over the test span). A long/short, "market-neutral-ish" momentum strategy
   that sits in cash most of the time structurally cannot keep up with a rising index —
   and this one didn't.

3. **Capital is badly under-deployed.** Measured average entry notional is **$1,977** —
   under 8% of the $25k account per trade. Even at the max of 8 concurrent positions the
   account is rarely more than half invested. Small positions + small edge = small
   absolute return, no matter how "clean" the win rate looks.

4. **98% of signals are filtered out.** The RDT gate stack (SPY gate → 50/200 SMA → VWAP
   → MTF) rejects ~98% of raw signals. This improves per-trade quality slightly (PF 1.24 →
   1.34) but starves the strategy of trades and does not change the fundamental math.

5. **The backtest flatters the strategy in three ways at once:** (a) no transaction costs,
   (b) daily bars standing in for an intraday method — the VWAP and first-hour gates that
   are core to the live system *cannot even be simulated here*, so the tested strategy is
   not the deployed strategy, and (c) perfect fills at exact stop/target prices with no
   gap-through. Real net results would be at or below the "@5 bps" row.

---

## The off-mission drift

Faced with the fact that trading alone won't hit the owner's return target, several repo
documents (`ACTIONABLE_100X_STRATEGY.md`, `WEALTH_STRATEGY_100X.md`,
`WEALTH_OPTIMIZATION.md`, `QUICK_START_100X.md`) and the most recent commit ("SaaS product
overhaul") pivot toward **selling the signals as a subscription service** and toward
**3x leveraged ETFs + higher risk-per-trade** to manufacture returns. Per the operator
mandate, both are off-mission:
- Subscription revenue is not trading P&L.
- Leverage and higher risk amplify a *negative-Kelly* edge — that increases the risk of
  ruin, not expected return.

---

## Honest verdict

On the evidence to date, **the RRS/RDT strategy as implemented does not beat SPY
buy-and-hold, and there is no measurement showing it ever has, net of costs.** It is not
obviously *broken* — it makes a little money gross — it is simply **not a good use of
capital** versus the index.

## Where a future instance could still find signal (in priority order)

1. **Fix the measurement before trusting any tuning.** Fold an honest cost model into the
   engine itself (or gate every decision through `net_cost_analysis.py`). Never compare a
   gross number to SPY again.
2. **Test whether the intraday method actually differs from the daily approximation.**
   The whole RDT thesis (RRS on 5-min bars, VWAP, first-hour) is untested here. If it
   can't be backtested intraday, its claimed edge is unfalsifiable — a red flag.
3. **Confront the capital-utilization problem.** If per-trade sizing is genuinely ~8% of
   account, either the sizer is mis-parameterized or the strategy is signal-starved; both
   cap returns structurally.
4. **Be willing to recommend wind-down.** If, after honest intraday testing, net-of-cost
   returns still trail SPY, the correct answer under the mandate is: stop trading this and
   hold the index. That is a valid, mission-completing outcome.
