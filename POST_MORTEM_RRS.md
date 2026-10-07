# POST-MORTEM: RRS Strategy — State of the Bot

> ⚠️ **RECONSTRUCTED 2026-10-07.** The operator's scheduled prompt references a
> `POST_MORTEM_RRS.md` "history of why the bot is in its current state." No such
> file existed in the repository or its git history. This document was
> reconstructed from the evidence available in the repo on 2026-10-07. It is a
> starting point, not the original narrative; correct it where the owner has
> better information.

## Purpose

Capture, for future stateless operator instances, **why the bot is where it is**
and what the evidence actually says — so no one wastes cycles re-discovering it.

## The core problem (one sentence)

The RRS long/short single-name intraday strategy, as built and documented, has
**no demonstrated net-of-cost edge over simply buying and holding SPY**, and has
**almost no real track record** to argue otherwise.

## Evidence as of 2026-10-07

### 1. Essentially no live/paper track record
- `data/signals/signal_metrics.json`: `total_outcomes: 2` (1 target hit, 1 stop
  out) across the bot's entire recorded life.
- `last_scan_at: 2026-03-05` — the scanner has not run in ~7 months.
- Two signal stores disagree: `signal_metrics.json` reports 120 signals
  (119 short / 1 long); `signal_history.json` holds 1986 entries
  (1687 long / 299 short). The counters are inconsistent and **neither carries
  P&L outcomes**. Fixing this instrumentation is a prerequisite to judging the
  strategy from live data.

### 2. The backtest loses to buy-and-hold
From `CLAUDE.md` ("Walk-Forward Backtest Results, 2 Years, $25K Capital",
Feb 2024 → Nov 2025):

| | Baseline | Old filters | **RDT filters (best)** |
|---|---|---|---|
| Total return | +3.3% | +5.1% | **+6.9% (+$1,716)** |
| Annualized | 1.6% | 2.5% | **3.4%** |
| Trades | 235 | 457 | 279 |

**SPY buy-and-hold over the identical window** (IBKR monthly closes,
508.08 → 718.66): **+41.4% total, +21.9% annualized, +$10,362 on $25K.**

The best bot configuration captured **~1/6 of the return of doing nothing but
holding the index**, while taking 279 directional trades with their associated
commissions, slippage, and (for shorts) borrow costs — costs the headline
figure may not fully reflect.

### 3. Structural headwind
A relative-strength long/short book that is frequently net-short (per the
metrics file) during a strong bull market is fighting the tape. Even the
mostly-long history log did not generate meaningful realized P&L. In a rising
market, the hurdle to beat buy-and-hold with market-timed single-name trades is
very high.

## Why the system looks sophisticated but unproven

The codebase is large and feature-rich (87 ML features, options module,
intermarket analysis, regime detection, multi-broker failover, a polished web
dashboard). CLAUDE.md documents extensive audits and fixes. **None of that is
evidence of profitability.** Engineering breadth and code quality are not P&L.
The one place the system states a bottom line (the backtest) shows
underperformance vs the index. The operator's job is to be clear-eyed about
this distinction.

## What would change the conclusion

1. **Live paper P&L, measured since inception, net of real costs, beating SPY
   over a meaningful out-of-sample window.** Nothing less establishes an edge.
2. An **independently reproduced backtest** that: includes commissions +
   slippage + short borrow, plots SPY B&H and 60/40 benchmarks on every report,
   and shows net outperformance out-of-sample across regimes.
3. If the honest goal is diversification rather than raw return, a **risk-
   adjusted** case (low correlation to SPY, positive after costs) judged on
   those terms — not raw return vs a bull-market index.

## Recommendation to the owner

Treat "does this beat buy-and-hold net of costs?" as an **open, unproven
question** and answer it with measurement before committing more build effort.
If repeated measurement says no, the mandate's stop condition applies:
**recommend wind-down or reframe the objective.**
