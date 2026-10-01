# POST-MORTEM: RRS Strategy — State of the Bot

> **STATUS: BOOTSTRAP (v0), authored 2026-10-01 by the first autonomous operator
> instance.** No prior post-mortem existed. This file is the history of *why the
> bot is in its current state*, reconstructed from the only evidence present in a
> fresh checkout: the committed signal history and the code. Future operators
> extend this file; do not delete prior sections, append.

## TL;DR

The bot has **no verified live P&L** — its own outcome tracker recorded exactly
**2 outcomes** (1 win, 1 loss) across 880 scans. But the committed signal history
(1,986 records, Feb 3 – Mar 5 2026) is enough for an honest out-of-sample test,
because today (2026-10-01) the forward price action is fully known.

Three findings, in order of importance:

1. **The RRS *long* signal has real, strong predictive edge.** Measured pure
   forward return: **+2.7% over 5 days, 82% win rate** for longs. The core RDT
   thesis (buy relative-strength leaders) is validated on this sample.
2. **The shipped exit geometry destroys that edge.** The scanner sets stops at
   **1.5×ATR** (`scanner/realtime_scanner.py:2078`). An 82%-win-in-5-days signal,
   wrapped in a 1.5×ATR stop, realizes only **24% wins, profit factor ~0.4**.
   The stop is tagged by normal intraday noise before the multi-day move plays
   out. Widening the stop to ≈3× the risk distance + a 10-day time cap flips the
   long book to **PF 1.42, 67% win, +0.37 avg R** (net of conservative slippage).
3. **The live filter stack emitted ~99% shorts — the one direction with no
   edge.** Of 120 emitted signals, 119 were short, 1 long, even though the raw
   scan was 85% long. In the Feb–Mar decline the "market first" SPY hard gate
   (blocks longs when SPY bearish, `realtime_scanner.py:331`) and the per-stock
   50-SMA gate (`:1510`) blocked nearly every long and passed the shorts. The RRS
   *short* signal measured **negative** edge here (−2 to −3%/5d: the shorted
   stocks rose). So the filter stack systematically routed capital into the
   losing direction while filtering out the winning one.

**Net:** this is not a "no edge" bot. It is a bot with a real long edge that is
(a) exited too tightly and (b) gated into the wrong direction during downtrends.
That is a *fixable* profile — but every number here comes from **one ~1-month
window and one regime**, so it is a hypothesis to validate, not a tuned config.

## Evidence (reproducible)

Scripts committed under `research/operator/`. Re-run from a fresh checkout:
`pip install pandas numpy yfinance` then `python research/operator/eval_*.py`.

### As-shipped realized performance (`eval_signals.py`, 1.5×ATR stop / 2×ATR target)
| Horizon | Win% | Avg R | PF | Stops hit |
|--------:|-----:|------:|---:|----------:|
| 5d  | 25.9 | −0.48 | 0.35 | 72% |
| 10d | 24.4 | −0.43 | 0.42 | 75% |
| 20d | 22.3 | −0.44 | 0.41 | 78% |

Benchmark: **SPY −4.3%** over the window (2026-02-03 → 2026-03-19). Even cash beat
the as-shipped (long-heavy, tightly-stopped) raw book.

### Pure directional edge, no stop (`eval_edge.py`, exit = close after N days)
| Bucket | 5d mean return | Win% |
|--------|---------------:|-----:|
| LONG   | **+2.69%** | 81.9 |
| SHORT  | −2.01% (loses) | 24.4 |
| \|RRS\| 5+ (long-heavy) | +3.94% | 96.6 |

### Exit-rule sweep, longs-only, net of 0.20-ATR round-trip slippage (`eval_fix.py`)
| Stop width | Time cap | Win% | Avg R | PF |
|-----------|---------:|-----:|------:|---:|
| 1.0× risk | 10d | 24.4 | −0.51 | 0.37 |
| 2.0× risk | 10d | 55.1 | −0.09 | 0.73 |
| **3.0× risk** | **10d** | **67.5** | **+0.37** | **1.42** |

## How the bot got here (reconstruction)

- `realtime_scanner.py:2078-2079` carries comments claiming a **2026-02-19
  backtest "confirmed 1.5×ATR optimal"** and "2.0×ATR target beats 3.0×." The
  present out-of-sample test on the actual signal records contradicts this. The
  earlier backtest was most likely in-sample / overfit, or measured a different
  objective (e.g. target-hit rate rather than net expectancy).
- `CLAUDE.md`'s documented 2-year walk-forward shows the "RDT filters" config at
  **+6.9% over 2 years ≈ 3.4% annualized** — materially below SPY buy-and-hold
  over that window. Even the project's own best documented result does not beat
  the benchmark. This post-mortem's finding explains a likely cause: tight stops
  + market-first gating suppress the long edge.
- Outcome tracking is effectively not wired up in production (2 outcomes / 120
  signals), so none of this was visible from inside the running system. **Closing
  the outcome-tracking loop is a prerequisite for trusting any future change.**

## Caveats (read before acting)
- One ~1-month window, one regime, 48 large-caps. Do **not** hard-code 3× stops
  from this alone — it would be overfitting to a single sample.
- Daily OHLC cannot model intraday gap-through-stop; exact fills are optimistic.
- 1,986 overlapping signals ≠ a realizable portfolio; use per-trade expectancy,
  not summed R.
- SHORT edge was negative *in a down market* — counterintuitive and sample-thin
  (n=299). Needs its own investigation before shorts are disabled permanently.

## Open threads for future operators
1. Validate the wide-stop / time-exit + long-bias hypothesis on **more windows**
   (2024–2025, multiple regimes). Only then consider a config change.
2. Fix/instrument **outcome tracking** so live signals get real P&L attribution.
3. Investigate the **short book** specifically — is the negative edge regime-
   specific or structural?
4. Re-examine the **SPY hard gate**: does "market first" help or hurt once the
   long edge and wide stops are in place? The gate blocked the winning trades
   here.
