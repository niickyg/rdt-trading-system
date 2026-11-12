# POST-MORTEM: RRS Trading System — History & Honest State

> Referenced by the operator MANDATE as required reading. Bootstrapped 2026-11-12 by the
> first operator instance from the evidence actually present in the repo (no prior
> post-mortem existed). Future operators: append, correct, and date your additions.

## What this system is

An autonomous trading bot implementing the r/RealDayTrading (RDT) methodology: Real
Relative Strength (RRS = (stock %chg − SPY %chg) / ATR%) momentum scanning, a stack of
filter gates (SPY market gate, 50/200 SMA, VWAP, multi-timeframe alignment, VIX/sector/
regime/intermarket overlays), optional ML signal validation, and multi-broker (paper /
Schwab / IBKR) execution. Paper trading only. ~25K notional account.

## The central problem (discovered 2026-11-12)

**The bot generated signals for a month and never measured whether they worked.**

- `data/signals/signal_history.json`: 1986 signal snapshots, Feb 3 → Mar 5 2026.
- `data/signals/signal_metrics.json`: **`total_outcomes: 2`** (1 target hit, 1 stop out).
- `agents/outcome_tracker.py` exists but evidently never persisted outcomes at scale.

So despite elaborate strategy machinery and confident backtest tables in CLAUDE.md, the
production system had **no feedback loop**. It could not have known if it was profitable.
The documented walk-forward results (CLAUDE.md: "$1,716 / 6.9% over 2 years, 3.4%
annualized") are *simulations* — and even taken at face value, **3.4%/yr loses badly to
SPY buy-and-hold** (~10%+/yr historically), i.e. it fails the mission's own success bar.

Secondary observations:
- Signal activity stops at 2026-03-05. The repo's signal data has been frozen ~8 months;
  whatever ran locally after that was never committed here.
- The 1986 snapshots are mostly re-logs: they dedupe to **88 unique (symbol, direction,
  day)** signals across only **6 trading days** — a very thin real sample.
- Heavy long bias (73 long / 15 short of the 88) during a *declining* Feb–Mar 2026 market,
  which suggests the SPY "market-first" gate was not actually suppressing longs as the
  methodology intends. Worth a future operator's scrutiny (not yet acted on).

## First honest measurement (2026-11-12)

`scripts/evaluate_signal_history.py` replays those 88 real signals against actual forward
prices (yfinance), realistic next-open entry, bracket (stop/target) exits, 10bps slippage
per side, benchmarked vs SPY over matched windows:

| Metric | Value |
|---|---|
| Trades | 88 (over 6 days, single declining regime) |
| Win rate | 63.6% |
| Avg R (winsorized) | +1.52R (median +0.99R) |
| Profit factor | 1.43 |
| Avg return / trade | +0.96% (after cost) |
| Avg alpha vs SPY | +2.00% (SPY was −1.04% in these windows) |
| Beat SPY | 62 / 88 (70%) |
| Return/trade 95% CI | **[−1.6%, +2.9%], p(≤0)=0.20 — NOT significant** |
| Alpha 95% CI | [−0.49%, +3.88%], p(≤0)=0.05 — borderline |

Robust across 5/10/20-day horizons (PF 1.3–1.4, positive alpha, all CIs still cross zero).

**Interpretation:** a genuinely *encouraging* point estimate — the signals picked stocks
that beat a falling index — but on far too little data to be trusted. The relative edge
(vs SPY) is more convincing than the absolute return. This is a **hypothesis worth
testing forward**, not a proven profitable strategy.

## Why the bot is in its current state

1. Built strategy-first, instrumentation-last. The measurement loop was never closed, so
   no one — human or bot — could tell signal from noise.
2. Backtest optimism was treated as evidence of profitability. It isn't.
3. Development stalled (data frozen March 2026) before any forward validation accumulated.

## The path forward (for operators)

1. **Close the feedback loop first.** Make outcome tracking actually persist, for every
   signal, with honest fills and costs. Until this works, everything else is guessing.
2. **Validate forward, out of sample.** The 88-signal edge is a hypothesis; let real paper
   P&L confirm or kill it over months and multiple regimes.
3. **Cut, don't add.** The system is already over-complex (dozens of overlays) relative to
   its evidence base. Prefer removing cost/complexity that can't be shown to earn its keep.
4. **Hold the line on the success bar:** net of costs, beats SPY, acceptable drawdown,
   demonstrated forward — or recommend wind-down.
