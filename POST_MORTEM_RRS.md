# POST-MORTEM: Does the RRS / RDT strategy actually beat buy-and-hold?

> **Provenance.** The scheduled operator task references a `POST_MORTEM_RRS.md`
> describing "the history of why the bot is in its current state." No such file
> had ever been committed. This document is a **reconstruction from first
> principles** by the first operator run (2026-11-19), built from (a) the code as
> it exists, (b) the numbers `CLAUDE.md` itself claims, and (c) real SPY price data
> pulled from IBKR. It is evidence, not invented narrative. Where a number is an
> estimate, it is labeled as such.

## TL;DR

By the system's **own** best-case backtest, and **before** any transaction costs,
the RDT/RRS strategy returned **+6.9% over ~21 months (≈3.4%/yr)**. Over that exact
window (Feb 2024 → Nov 2025), **SPY buy-and-hold returned ~+34.5% price-only
(~+36.7% total return), ≈18.5%/yr** (real IBKR monthly closes). Once honest costs
are applied, the strategy's net return falls to roughly **0.4%–2.5%/yr**.

**The active-trading apparatus destroyed ~30 percentage points of return versus
doing nothing.** This is the central fact the project must confront.

## The evidence

### 1. SPY buy-and-hold over the backtest window (real data)

IBKR monthly closes, SPY (conid 756733):

| Date | SPY close |
|------|-----------|
| 2024-02-01 | 508.08 |
| 2025-11-03 | 683.39 |

- Price return: 683.39 / 508.08 − 1 = **+34.5%**
- + ~1.25%/yr dividend over ~1.75yr ≈ +2.2% → **~+36.7% total return**
- Annualized (price only): **~18.5%/yr**

### 2. The bot's own claimed result (gross of costs)

`CLAUDE.md` → "Walk-Forward Backtest Results (2 Years, $25K Capital)", config C
("RDT Filters"), the best of the three configs:

- Total Return: **+$1,716 (+6.9%)**
- Annualized: **3.4%**
- 279 trades, 49.5% win rate, profit factor 1.24

### 3. The backtest applies ZERO transaction costs

Confirmed by reading the engine:

- `backtesting/engine_enhanced.py` computes P&L as pure
  `(exit_price − entry_price) × shares` at every exit (lines ~448, ~598). No
  commission, no slippage, no spread anywhere.
- `scripts/run_backtest.py::apply_slippage()` prints a "0.1% slippage" banner but
  its body is a **no-op** — it returns the price data unchanged with a comment that
  slippage is "documented" rather than applied.

So the +6.9% is a **gross, frictionless** number. Every real trade pays commission
and crosses the bid/ask.

### 4. Net of honest costs

A conservative cost model for liquid large-cap US equities via IBKR — $0.005/share
commission + ~5 bps/fill slippage, ≈**0.115% round-trip** on notional — applied to
the 279 trades (estimate; exact figure depends on realized per-trade notional):

| Avg position notional | Est. total cost | Net of claimed $1,716 gross |
|-----------------------|-----------------|------------------------------|
| $2,000 | ~$614  | **+$1,102** (≈2.5%/yr) |
| $3,500 | ~$1,074 | **+$642** (≈1.5%/yr) |
| $5,000 | ~$1,534 | **+$182** (≈0.4%/yr) |

A $25K account configured for up to 8 concurrent positions trades toward the larger
notionals, where costs consume **~90%** of the gross edge. And these slippage
assumptions are *optimistic* for a strategy that enters on intraday momentum (chasing
fills), so real net is likely at the bottom of this range or worse.

## Why RRS/RDT underperforms here (interpretation)

1. **The edge is tiny and fragile.** A profit factor of 1.24 and a ~3.4%/yr gross
   return is a thin edge that transaction costs, by construction, are designed to
   eat. 98% of raw signals are filtered out (per CLAUDE.md) to reach it — a sign the
   raw signal has little standalone predictive power.
2. **It fights a rising market.** 2024–2025 was a strong bull. An active long/short
   momentum strategy taking fewer, "higher-quality" trades sat in cash most of the
   time and captured a fraction of the trend that a passive holder captured for free.
3. **Costs were never in the objective.** The whole optimization loop (filters,
   regimes, ML, Murphy features) was tuned against a frictionless backtest. You
   cannot tune your way to net profit while optimizing a gross metric.
4. **Benchmark was never in the objective either.** No backtest in the repo compares
   results to SPY buy-and-hold. The system could call +6.9% a "win" while silently
   losing to the index by 5×.

## Recommendation

- **Do not add features to this core.** Every hour spent on the RRS strategy, ML
  ensembles, options overlays, or (especially) SaaS productization is an hour spent
  polishing a vehicle that loses to buy-and-hold net of costs.
- **Change the objective first.** Any future backtest must (a) apply honest costs
  and (b) report return vs SPY buy-and-hold over the same window, or it is not
  measuring the thing that matters. (The first run wired honest costs into the
  engine; the benchmark comparison is the next required step — see the journal.)
- **Then re-test the hypothesis honestly.** If, net of costs and vs SPY, the strategy
  still cannot win across multiple regimes, **recommend wind-down of active trading**
  and escalate: the capital is better in the index.

---

## ADDENDUM — 2026-11-20: estimate → MEASUREMENT, and a second regime

The body above was the first run's *reconstruction* (gross numbers + an estimated cost
haircut). The 2026-11-20 run **actually executed** the walk-forward with the honest cost
model on, with a SPY buy-and-hold benchmark built into the tool, over two regimes. The
estimate held and sharpened. (Numbers are net of ~0.115% round-trip costs; SPY uses
`auto_adjust=True` = total return. Windows differ from the body's Feb2024–Nov2025 span
because the tool anchors to "today − 730d" unless `WF_END_DATE` overrides it.)

**Bull window 2024-07-26 → 2026-08-07 (SPY total return +45%):** best config C) RDT Filters
**+8.93% net**; SPY buy-and-hold **+37.80%** (apples-to-apples). Strategy loses by **29
points** — it captured <¼ of the index.

**Bear/flat window 2021-10-05 → 2023-10-13 (includes the 2022 crash; SPY +5.75%):** best
config A) raw Baseline **+5.72% net** — a **dead-heat tie** with SPY (−$7.58). The *filtered*
configs **lost money** (B −2.65%, C −1.95%, both profit factor < 1).

**Two conclusions, now measured, not estimated:**
1. **No configuration beats SPY buy-and-hold in either regime, net of costs.** Best case = tie.
2. **The RDT filter stack is not an edge.** In a non-trending market it is a *liability*: it
   turned a profitable, PF-1.41 baseline into a PF-0.87 loss. The CLAUDE.md claim that the
   filters produce "fewer but higher-quality trades" is **falsified** out of the bull sample.

The recommendation in the body stands and is strengthened: **do not add to this core;
escalate the wind-down / pivot decision to the human.** The remaining open question — is
there *any* raw signal with a profit factor high enough to survive costs? — should be
answered with factor-level IC analysis (`research/factor_tester.py`) before any further
effort. See `data/operator_journal/entries/2026-11-20-session-01.md` for full tables,
steelman, and caveats.
