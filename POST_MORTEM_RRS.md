# POST-MORTEM: The RRS Strategy and How This Bot Got Here

*Reconstructed 2026-08-12 by the first operator instance from repository evidence.
This is the honest history the operator journal is built on. Later instances:
correct this file if you find better evidence, but do not soften it.*

---

## The one-sentence summary

The RDT/RRS strategy in this repo has **never been shown to beat SPY buy-and-hold**,
its best documented backtest (~6.8%/yr, frictionless) is inconsistent across the
project's own docs, its one month of live paper signals (Feb–Mar 2026) recorded
**essentially zero trade outcomes**, and the project has repeatedly drifted away
from proving an edge toward return-chasing ("100X") and a SaaS "signal service"
pivot. There is, as of today, **no validated profitable edge**.

## What the strategy is

**Real Relative Strength (RRS)** = `(Stock %chg − SPY %chg) / ATR`. Long when a
stock is strongly outperforming SPY (RRS > ~2), short when underperforming. Layered
on top: SPY regime gate, 50/200 SMA gate, VWAP gate, multi-timeframe alignment,
VIX/sector/intermarket/regime adjustments. The thesis (r/RealDayTrading): relative
strength persists intraday, so buying the strongest stocks in an up market is a
positive-expectancy momentum bet.

## The evidence that exists (and its problems)

### 1. The documented backtests disagree with each other
- `CLAUDE.md` walk-forward (2yr, $25k): RDT filters → **$1,716 (6.9%), 49.5% win
  rate, PF 1.24, 279 trades**. Annualized **3.4%**.
- `ACTIONABLE_100X_STRATEGY.md`: "**38% win rate, profit factor 1.29**," ~215
  trades/yr, ~$1,700/yr.
- `WEALTH_STRATEGY_100X.md`: "6.84% annual, $1,711," win rate implied ~38%.

Same strategy, win rates quoted as **38% and 49.5%** in different files. This is a
measurement-discipline red flag: the numbers were not produced by one trusted,
reproducible harness. **None of them state costs clearly, and none compare to SPY
buy-and-hold over the same window.**

### 2. Even taken at face value, it loses to the benchmark
Best case ~6.8%/yr **frictionless**. SPY buy-and-hold is historically ~10%/yr. In
the actual live-signal window, SPY (IBKR daily) ran **~682 (Feb 17) → ~770 (Aug 12),
roughly +13% in six months**. A strategy that nets 3–7% a year before costs, while
the benchmark does double digits, is not an edge — it is a worse way to hold beta.

### 3. Live paper trading produced no outcome data
- `data/signals/signal_history.json`: **1,986 signals**, all `RRS_Momentum`,
  generated **2026-02-03 → 2026-03-05** (one month), then nothing. The bot has been
  **dormant ~5 months**.
- **Zero** of those 1,986 signals carry any outcome/exit/pnl field.
- `data/signals/signal_metrics.json`: 880 scans, 120 signals emitted, **total
  outcomes tracked = 2** (1 target hit, 1 stop). Two. There is no live track record.

So the loop "scan → signal → trade → record result → learn" was never closed. The
system generated signals into the void.

### 4. The project drifted from "prove an edge" to "monetize activity"
- `ACTIONABLE_100X_STRATEGY.md` openly concedes the strategy "**mathematically caps
  returns around 7%**" and is "**signal-limited, not capital-limited**," then
  proposes filling the gap to a 100% return target with **"Signal Service Revenue:
  $10,000–15,000."** Selling signals is not trading profitability.
- Recent git history (`feat: SaaS product overhaul`, landing/pricing/login/register
  pages, "AI Confidence" and "Trading Journal" dashboards) shows engineering effort
  going into a **product to sell**, not into establishing whether the underlying
  signal makes money.

## Why the bot is in its current state

1. **The measurement machinery was never trustworthy.** Multiple backtest engines
   (`backtesting/engine.py`, `engine_enhanced.py`, `engine_intraday.py`,
   `parameter_optimizer.py`) produced inconsistent, cost-ambiguous numbers that
   nobody reconciled against a benchmark.
2. **The feedback loop was never closed.** No outcome tracking on live signals means
   the "learning" agents (`learning_agent`, `adaptive_learner`, ML ensemble) had
   nothing real to learn from. The ML is, per CLAUDE.md's own notes, advisory-only
   and mostly at coin-flip accuracy (exit predictor 43%).
3. **Goal displacement.** When ~7% wouldn't reach an arbitrary 100% target, the
   response was to add complexity (17 more ML features, intermarket layers) and to
   pivot to selling signals — not to ask "does the base signal even beat SPY?"

## What has to happen (the mandate, restated)

Before any more features, ML, or product work, the project must answer one question
with a single reproducible harness that includes costs and a SPY benchmark:

> **Over a defined out-of-sample window, does acting on these signals beat SPY
> buy-and-hold, net of realistic commissions and slippage?**

If yes, scale it carefully. If no — which is what all current evidence suggests —
stop adding complexity and either find a genuinely different edge or wind down.
Everything the operator does should serve answering that question honestly.

---

*Ground truth beats narrative. If a future instance finds this too harsh and the
data disagrees, update it with the data. Do not update it with hope.*
