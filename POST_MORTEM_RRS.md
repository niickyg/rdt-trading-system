# POST-MORTEM: The RRS / RDT Strategy — Does It Beat SPY?

> First authored 2026-10-15 by the autonomous operator (first run). This file is
> referenced by the operator mandate as "the history of why the bot is in its
> current state." It did not previously exist; this is the initial honest
> assessment. Future operator instances should append, not overwrite.

## TL;DR

**The strategy, as implemented and backtested, does not beat SPY buy-and-hold —
not even close, and not even on its own optimistic, cost-free numbers.**

| | Best bot config (C/RDT) | SPY buy-and-hold |
|---|---|---|
| Total return (Feb 2024 → Nov 2025) | **+6.9%** ($1,716 on $25k) | **+34.2%** (~$8,540 on $25k) |
| Annualized | **~3.4%** | **~17.7%** (price-only; ~19% with dividends) |
| Costs modeled | **None** (no commissions/slippage/spread/borrow) | Real prices |
| Execution/overnight/software risk | High | ~None |

SPY returned roughly **5x** the bot's annualized return over the same window —
and the bot's figure is inflated by an idealized backtest (see §3). Net of
honest costs, the real gap is wider.

*SPY figures computed 2026-10-15 from real IBKR weekly OHLC data (contract
756733): 2024-02-05 close 501.20 → 2025-11-24 close 672.38, 658 calendar days.*

## 1. What the strategy is

r/RealDayTrading "Real Relative Strength" (RRS): go long stocks showing strength
vs. SPY (`RRS = (stock %Δ − SPY %Δ) / ATR > 2`), short relative weakness, gated
by a stack of filters (SPY market gate, 50/200 SMA, VWAP, multi-timeframe
alignment, VIX regime, sector RS, intermarket). The thesis: relative strength
reveals institutional accumulation; trading with it produces an edge.

## 2. What the numbers actually say (from the repo's own artifacts)

From `CLAUDE.md` and the `*_100X*` strategy docs (the project's own
documentation), the best backtested configuration produces:

- **~6.8–6.9% total return** over 2 years (~3.4% annualized) on $25k.
- **Win rate 37–38%**, **profit factor ~1.24–1.29**.
- The strategy is **signal-limited, not capital-limited** — ~98% of raw signals
  are filtered out, so it cannot simply be "scaled up."

The project's own `ACTIONABLE_100X_STRATEGY.md` concedes a **14.6x gap** to its
stated goal and proposes closing it with **leverage, crypto/futures, and selling
a signal service** — i.e., revenue from *marketing the strategy to others*, not
from the strategy's returns. **When the plan to hit a target depends on selling
the strategy rather than trading it, that is evidence the strategy has no edge.**

## 3. Why even the 3.4% is optimistic — backtest methodology problems

The walk-forward backtest (`scripts/run_walkforward_v2.py` +
`backtesting/engine_enhanced.py`) overstates performance:

1. **Zero transaction costs.** P&L is `(exit − entry) × shares` with nothing
   subtracted — no commissions, no slippage, no bid/ask spread, no short-borrow.
   For a strategy taking hundreds of trades/year, costs are a first-order drag.
2. **Daily bars for an intraday strategy.** RDT is a *day-trading* method
   (5-minute entries, VWAP tests, compression breaks). The backtest enters at
   the **daily close** and holds **overnight across days** — a different
   strategy with a different (and un-simulated) risk profile. The script itself
   admits "VWAP + First-Hour filters cannot be simulated with daily bars."
3. **Idealized fills.** Stops fill at exactly the stop price and targets at
   exactly the target price, ignoring gap-throughs. Overnight holds make gap
   risk asymmetric against you; the backtest assumes it away.
4. **No benchmark.** The backtest compared three strategy configs to each other
   but **never to SPY buy-and-hold** — so it never tested the one thing that
   defines success. (Fixed this run — see the operator entry.)

Net: the honest, cost-adjusted, intraday-realistic return is **below** the
reported 3.4%, against a benchmark delivering ~17–19%.

## 4. Project drift (a warning for future instances)

The repo has drifted away from the sober README (which correctly says day
trading takes years, 95% fail, and this is not get-rich-quick) toward:

- `WEALTH_STRATEGY_100X.md`, `ACTIONABLE_100X_STRATEGY.md`,
  `WEALTH_OPTIMIZATION.md`, `QUICK_START_100X.md` — "100% annual return" framing.
- A "SaaS product overhaul" (most recent commit) — landing/pricing/onboarding
  pages — i.e., productizing the bot for sale rather than proving it works.

Both directions substitute *narrative* and *distribution* for *edge*. Per the
mandate, they are out of scope absent explicit owner direction.

## 5. Recommendation

1. **Do not run this strategy with live capital.** On the evidence it would
   underperform a passive SPY purchase by ~5x while adding real risk.
2. **Treat SPY buy-and-hold as the hard benchmark in all tooling** (done this
   run for the primary backtest). No strategy change should be entertained
   unless it plausibly beats SPY *net of honest costs, out-of-sample*.
3. **Either** (a) **wind down** the automated day-trading thesis and hold SPY
   (or a diversified index), **or** (b) **pivot to disciplined research** with a
   pre-registered success bar before any further engineering: add realistic cost
   & slippage modeling, test on intraday data, and require out-of-sample
   outperformance vs. SPY before touching live-adjacent code.
4. **Freeze the "100X"/SaaS drift** until (and unless) there is a measured edge.

## 6. Open questions for the owner

- Is the real goal *trading returns*, or building a *product* to sell? These need
  different work and different honesty.
- What is the actual risk tolerance and time horizon? If long-horizon growth,
  index buy-and-hold already dominates this strategy on the evidence.
- Is there any live paper-trading P&L record to compare against the backtest?
  (The operator has no access to the owner's live container/DB.)
