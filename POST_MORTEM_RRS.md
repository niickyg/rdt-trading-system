# POST-MORTEM: RRS Strategy & the Profitability Question

> **STATUS: BOOTSTRAP RECONSTRUCTION (2026-11-03).** The scheduled operator prompt references
> this file as pre-existing history. It did not exist in the repo. This is an honest
> reconstruction from committed evidence (CLAUDE.md, the `*_100X_*` strategy docs,
> `data/signals/*.json`, config). Dates/figures are cited from those artifacts. The human
> should correct anything that misremembers the real history.

## What the system is

An autonomous RDT (r/RealDayTrading) momentum bot: Real Relative Strength (RRS) scanning,
multi-gate "market first" filtering (SPY → 50/200 SMA → VWAP → MTF), optional ML advisory
layers, an options module, and a Flask dashboard. Paper trading by default (IBKR paper
account DUP995654, per CLAUDE.md).

## The core finding: the strategy does not clear the mission bar

The mission is positive P&L net of honest costs **beating SPY buy-and-hold**. The best
evidence committed to this repo says the strategy does not do that:

| Source (committed) | Best result | Period / basis |
|---|---|---|
| `CLAUDE.md` walk-forward V2, "RDT Filters" config | **$1,716 (6.9% total), 3.4% annualized** | 2yr, 6 quarterly windows, $25K |
| `ACTIONABLE_100X_STRATEGY.md` | **~6.8% annual (~$1,700)**, PF 1.29, 38% WR | optimization run |
| `WEALTH_STRATEGY_100X.md` | **6.84% annual (~$1,711)** | stated current state |

SPY buy-and-hold has historically returned ~10%/yr. A 3.4–6.9% annualized result — **before**
a full accounting of commissions, slippage, and option spreads — does not beat it. The
project's own documents concede this: the RRS strategy "mathematically caps returns around
7%" because it is "signal-limited, not capital-limited."

## The pivot that reveals the problem

Both `*_100X_*` documents respond to the 14.6x gap between ~7% and the 100% target not by
finding a trading edge, but by pivoting to **selling the signals** (signal service, API
access, education) and adding leverage/options/crypto/futures. The recent `main` history
("SaaS product overhaul," landing/pricing/login pages) confirms the project drifted toward
being a **sellable product** rather than a **profitable trading bot**. Selling signals is a
business model, not evidence the bot makes money trading — and it is outside this operator's
mission of actual trading P&L.

## Data-quality red flags (as of 2026-11-03)

- **Stale data.** `signal_metrics.json` `last_scan_at = 2026-03-05`; `signal_history.json`
  spans Feb–Mar 2026. There is **no committed live P&L** proving profitability in production.
- **Directional skew.** `signal_metrics.json`: 120 signals = **119 short, 1 long**. Either a
  directional bug, a mislabel, or an extreme one-sided period. Unexplained and unresolved.
- **Essentially no outcome data.** `total_outcomes = 2` (1 target hit, 1 stop-out). Two
  trades is not evidence of an edge in either direction.

## Honest verdict

On committed evidence, **the RRS strategy as configured does not beat SPY buy-and-hold**, the
project's own analysis says it cannot without changing the risk profile (leverage/options) or
changing the business (selling signals), and there is **no live P&L** to overturn that. This
is the mandate's escalation trigger, not a cue to tune parameters.

## Open questions for the human

1. Is the goal still **trading profitability**, or has it become a **SaaS/signal product**?
   These require opposite work and the operator should not guess.
2. What explains the 119-short / 1-long skew? Bug or market period?
3. Is there any committed **live** paper-trading P&L the operator can assess, or only
   backtests? If only backtests, the profitability claim is unproven.
