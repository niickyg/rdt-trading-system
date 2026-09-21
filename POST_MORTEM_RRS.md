# POST-MORTEM: The RRS Strategy and Why This Bot Is Where It Is

> Reconstructed on 2026-09-21 by the first operator instance from the evidence in the repo
> (CLAUDE.md, the strategy docs, signal metrics/history, and a fresh SPY benchmark pull).
> No prior post-mortem existed; this file is the bootstrap history. Correct it as facts sharpen.

## The one-paragraph version

This system implements the r/RealDayTrading (RDT) "Real Relative Strength" (RRS) momentum
methodology with a large stack of filters (SPY gate, SMA gate, VWAP, multi-timeframe, VIX, sector,
regime, intermarket) and an advisory ML layer. Enormous engineering effort went into it. But on its
**own** best 2-year walk-forward backtest it returns about **+6.9% total / +3.4% annualized**, while
**SPY buy-and-hold over the identical window (Feb 2024 – Nov 2025) returned +42.7% total / +21.5%
annualized.** The strategy underperformed doing nothing by ~36 percentage points — and that is the
optimistic *backtested* number. The *realized* track record is **2 closed trades.** The bot has been
dormant since its last scan on **2026-03-05**.

## The numbers (with the benchmark the repo never shows)

| Source | Total return | Annualized | Win rate | Profit factor | Sample |
|---|---|---|---|---|---|
| CLAUDE.md walk-forward "RDT filters" (best) | +6.9% | +3.4% | 49.5% | 1.24 | 279 backtest trades |
| WEALTH docs "current" | ~6.8% | ~6.8%* | 37–38% | 1.29 | 215–285 trades/yr |
| Earliest baseline (WEALTH_OPTIMIZATION) | — | 2.85% | 33.7% | 1.23 | 83 trades/yr |
| **SPY buy-and-hold, same window** | **+42.7%** | **+21.5%** | — | — | passive |
| **Realized (signal_metrics.json)** | — | — | 1W/1L | — | **n=2** |

\* The docs quote annual and total interchangeably; treat all bot numbers as backtested and optimistic.

Fresh SPY pull (this session, `yfinance`, auto-adjusted): SPY 2024-02-01 → 2025-11-28, 666 days,
+42.7% total, +21.5% annualized.

## Why it underperforms (root causes, best current understanding)

1. **Structurally low edge.** ~38% win rate with tight 0.75×ATR stops and 1.5–2.0×ATR targets and
   profit factor ~1.24–1.29 is a thin edge. At the mandated ~1–1.5% risk/trade it caps returns in
   the low single digits *even if the backtest edge is real*. The math is acknowledged in the repo's
   own WEALTH docs.
2. **The benchmark was never in the frame.** Every comparison in the repo is config-vs-config
   (baseline vs old filters vs RDT filters). None compares to SPY buy-and-hold. Once you add it, the
   whole program is revealed as underperforming a passive hold during a strong bull market.
3. **Overfitting risk is high.** The improvement from baseline → "RDT filters" came from stacking
   filters and regime parameters tuned on the same ~2 years. "98% of raw signals are filtered out"
   means the surviving sample is tiny and the parameters are many — a classic overfit signature.
   Out-of-sample robustness is unproven.
4. **Costs are not modeled at all (VERIFIED this session).** `backtesting/engine_enhanced.py`
   settles every exit at the raw stop/target price with **zero commission, slippage, or spread**
   (P&L = `(exit_price - entry_price) * shares`, lines 423–428 and 567–580; the harness has no
   `commission`/`slippage`/`spread` term anywhere, and `scripts/run_walkforward_v2.py` prints no
   SPY buy-and-hold row). **Break-even sensitivity:** the best config nets ~$1,716 over ~279
   backtest trades = **~$6.15 of net profit per trade.** Any all-in round-trip friction above
   ~$6/trade turns the strategy **net-negative**. For a system firing hundreds of trades/yr,
   $6 all-in (commission + spread + slippage) is easily exceeded — so the true, cost-honest edge
   is plausibly at or below zero, before we even discuss the SPY benchmark.
5. **The response to the gap was a pivot, not a fix.** ACTIONABLE_100X_STRATEGY.md and
   WEALTH_STRATEGY_100X.md propose closing the 14.6× gap to "100% annual" by adding a **signal-service
   revenue business, education, API sales, crypto/futures expansion, and leverage** — i.e. by *not
   trading better*. That pivot is a confession that trading edge over the benchmark was not found.
6. **No realized evidence.** Only 2 outcomes were ever tracked, then the system went dormant in
   March 2026. There is no live track record to validate or refute the backtest.

## What was actually built (credit where due)

The engineering is real and mostly sound: agent architecture, secure model loading, IBKR/paper
brokers, options module, a research/factor-testing framework, hardened auth, and thorough audits.
The problem is **not** code quality. The problem is that no amount of engineering has produced a
strategy that beats buying and holding SPY. This is a strategy/edge problem wearing an engineering
costume.

## The honest question going forward

Not "how do we get to 100%?" — that target is a fantasy given the measured edge (see MANDATE §6).
The real, humble question is binary: **Can this system beat SPY buy-and-hold, net of honest costs,
out-of-sample — at all?** Until that is answered "yes" with evidence that includes the benchmark and
real friction, every other feature is decoration. If the answer keeps coming back "no," the correct
recommendation is to stop and hold SPY.

## Immediate priorities for the next operator instance

1. Build a **benchmark-and-cost-honest** walk-forward harness: every result printed next to SPY
   buy-and-hold for the same window, with commission+slippage+spread charged. The audit is already
   done (this session): `scripts/run_walkforward_v2.py` + `backtesting/engine_enhanced.py` charge
   **no** friction and print **no** benchmark. The concrete task is to add (a) a per-round-trip cost
   parameter applied at every fill in `engine_enhanced.py`, and (b) a SPY buy-and-hold column in the
   walk-forward report. Start conservative: ~$3–7 all-in per round trip, then sweep.
2. Re-run the "RDT filters" config through that harness on **out-of-sample** data (2026 YTD, which the
   original 2024–2025 tuning never saw).
3. Report the benchmark-relative, cost-net result. If it loses to SPY, say so and escalate per MANDATE §7.
