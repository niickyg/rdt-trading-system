# POST-MORTEM: The RRS Strategy and Why the Bot Is Where It Is

> **Provenance note.** The operator scheduling prompt refers to this file as
> "the history of why the bot is in its current state." No such file existed in
> the repository or its git history as of 2026-08-17. This document was
> **reconstructed by the first operator instance from verifiable repository
> evidence** — the committed strategy documents, the backtest code, the signal
> data files, and `CLAUDE.md`. Where a claim comes from a project document, it
> is attributed. Where it is the operator's own reading of the code, it is
> labeled as such. It contains no invented history.

---

## What the bot is

An autonomous trading system built around **Real Relative Strength (RRS)** —
`RRS = (Stock %Change − SPY %Change) / ATR%` — layered with a large stack of
filters (SPY regime gate, 50/200 SMA, VWAP, multi-timeframe alignment, VIX
regime, sector RS, intermarket analysis) and an advisory ML layer. It targets a
$25K paper account on Interactive Brokers, trading momentum/relative-strength
setups in the spirit of r/RealDayTrading. An options module and a SaaS
signal-service front-end have also been built.

## The core problem, stated honestly

**There is no demonstrated edge that beats SPY buy-and-hold net of honest costs.**
Every strand of evidence in the repo points the same way:

### 1. The strategy's own headline numbers are single-digit and inconsistent

- `CLAUDE.md` (walk-forward V2, 2 years, $25K): best config = RDT filters,
  **+$1,716 (6.9%) total ≈ 3.4% annualized**, 49.5% win rate, profit factor 1.24.
- `ACTIONABLE_100X_STRATEGY.md` (same system, "from optimization"): **6.8%
  annual**, **38% win rate**, profit factor 1.29.

Win rate diverges by ~11 points (38% vs 49.5%) and the two "best" annual figures
(6.8% vs 3.4%) differ by ~2x depending on the document. Numbers that disagree
this much across the project's own files indicate the "results" are not robust —
likely sensitive to window choice, symbol set, and parameter tuning (overfitting
risk), not a stable measured edge.

### 2. By the project's own math, the edge is marginal-to-negative

`ACTIONABLE_100X_STRATEGY.md` computes the Kelly criterion for the strategy at
**≈ −0.02 (negative)** and states: *"the current edge is marginal … Increasing
position size actually increases risk of ruin without improving returns"* and
*"trading alone cannot achieve 100% returns."* A negative Kelly means the honest
recommendation of the strategy's own author is to bet **zero**.

### 3. The backtest is frictionless — the real numbers are lower than reported

**Operator's reading of the code (`backtesting/engine_enhanced.py`,
`scripts/run_walkforward_v2.py`, as of 2026-08-17):** P&L is computed as
`(exit_price − entry_price) × shares` with:
- **no commissions,**
- **no slippage,** and
- **fills assumed at the exact stop/target price** (stops never gap or slip).

For a strategy that took ~279 trades over 2 years for ~$1,716 of *gross* profit,
realistic IBKR commissions plus even a couple of basis points of slippage per
fill — and worse fills on stop-outs — plausibly consume a large fraction of that
gross. The reported returns are therefore an **upper bound**, and the true
net-of-cost figure is materially lower, possibly near zero or negative. (The
2026-08-17 run added a cost + slippage model and a SPY benchmark to the
walk-forward reporting so this can finally be measured — see that day's journal
entry. It had not yet been executed against live data at time of writing.)

### 4. The benchmark is brutal

SPY buy-and-hold over the backtest span (2024–2025) returned dramatically more
than the strategy's 3–7% gross annual — equity indices compounded at roughly
20%+/yr in that window. A strategy that underperforms a no-effort, no-risk-of-ruin,
zero-maintenance benchmark **by a wide margin, before costs** has not justified
its own existence as an active trading system.

### 5. There is essentially no live/paper track record

`data/signals/signal_metrics.json` (as of the committed data): 880 scans, 120
signals generated, and **2 tracked outcomes total** (1 target hit, 1 stop-out).
There is no statistically meaningful record of the strategy trading real (even
paper) capital. All performance claims rest on backtests, which per §3 are
frictionless and per §1 are internally inconsistent.

### 6. The "path to profitability" quietly concedes the point

`ACTIONABLE_100X_STRATEGY.md`'s plan to reach its return target is roughly
**40–60% from selling a signal-service subscription**, not from trading. When the
plan to make a *trading* bot profitable depends on *not trading* — on selling
signals of a negative-Kelly strategy to other people — that is an admission the
trading edge is insufficient. (It is also an ethical hazard: selling signals from
a strategy the author's own math says has no edge.)

## Why the bot is in its current state (the pattern)

The repository shows enormous engineering breadth — dozens of filters, an
options module, ML ensembles, an intermarket layer, a SaaS front-end, multiple
"100X"/"WEALTH" strategy documents. What it does **not** show is a single,
honestly-measured, cost-adjusted, benchmark-beating result. The recurring
failure mode is **adding complexity in place of establishing edge**: each new
filter is justified by a backtest delta, but the backtests are frictionless and
un-benchmarked, so the deltas are noise dressed as signal. Effort went into
*building* and into *aspirational revenue planning* rather than into the one
unglamorous question that decides everything: *does this beat buying SPY, after
costs?*

## What would actually change the verdict

Only one of these, demonstrated honestly, should reopen the case for active trading:
1. A net-of-cost, out-of-sample (true walk-forward, no peeking) result that
   **beats SPY buy-and-hold on a risk-adjusted basis** over a multi-year span,
   reproduced on data the operator can see.
2. A specific, mechanistic reason the edge should exist that hasn't been mined
   into the backtest (e.g., an execution or structural advantage), tested cleanly.

Absent that, the mandate's honest recommendation trends toward **wind-down**:
default to the benchmark. See `data/operator_journal/MANDATE.md` §5.

---

*Maintained by the operator. Update this file when new evidence materially
changes the diagnosis — and record the update in a journal entry.*
