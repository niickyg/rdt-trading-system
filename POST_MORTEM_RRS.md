# POST-MORTEM: The RRS Strategy and Why the Bot Is Where It Is

> **Provenance:** Bootstrapped 2026-08-19 by the first operator instance. The
> scheduled operator prompt tells every instance to read `POST_MORTEM_RRS.md`, but
> it did not exist. This file reconstructs the history from the repository's own
> documents and from *real* market/account data pulled this session. Future
> instances should append, not rewrite.

---

## The one-paragraph version

This system trades a Real Relative Strength (RRS) momentum strategy on a fixed
watchlist of large-cap US stocks, wrapped in an elaborate stack of filters,
agents, ML models, options modules, and a SaaS web app. Its **own best honest
backtest** is roughly **3.4% annualized** (2-year walk-forward, `CLAUDE.md`) to
**6.8% annualized** (a looser, single-window, likely-overfit run,
`DEPLOYMENT_SUMMARY.md`). Over the *same* period, **SPY buy-and-hold returned
~17% annualized** (measured this session from real IBKR data). The strategy
therefore **underperforms a one-click index purchase by ~3–5x**, while adding
trading cost, execution risk, and enormous operational complexity. The project's
own "100X" documents implicitly concede this: they pivot from trading returns to
*selling signals as a subscription* to hit their revenue goal.

---

## The evidence (measured, not asserted)

### Real SPY benchmark (IBKR `get_price_history`, monthly bars, this session)

| Window | SPY start | SPY end | Total | Annualized (price only) |
|---|---|---|---|---|
| Aug 2024 → Aug 2026 (2yr) | 563.68 | 767.45 | +36.1% | **+16.7%** |
| Aug 2024 → Nov 2025 (≈ walk-forward window) | 563.68 | 683.39 | +21.2% | **+17.3%** |

Add ~1.2%/yr in dividends for total return. This is the bar the mandate says to beat.

### The strategy's own best backtests (from repo docs)

| Source | Method | Return | Notes |
|---|---|---|---|
| `CLAUDE.md` (RDT filters, "Config C") | 2yr walk-forward, 6 windows | **6.9% total / 3.4% annualized** | The most honest number in the repo |
| `DEPLOYMENT_SUMMARY.md` | 365-day single backtest | 6.84% annual | Looser params, likely overfit |
| `ACTIONABLE_100X_STRATEGY.md` (self-reported) | optimization | 6.8% annual, PF 1.29, 38% WR | Concedes strategy is "signal-limited" |

Every one of these is **below SPY buy-and-hold for the same era**, and none of
them, as far as the repo shows, models commission + slippage + spread rigorously.
Net-of-honest-cost numbers would be *lower* than reported.

### Why more risk doesn't help (per the repo's own optimization)

`DEPLOYMENT_SUMMARY.md` and `ACTIONABLE_100X_STRATEGY.md` both find returns are
flat across risk profiles because the strategy is **signal-limited** (~240 quality
setups/yr) — cranking risk-per-trade just widens variance, not expectancy. A 38%
win rate with profit factor ~1.29 mathematically caps returns near single digits.

---

## What is actually running (measured this session)

- **Connected IBKR account:** ~$5 net liquidation, **0 positions, 0 trades in 90
  days.** NAV history (`get_pa_performance_all_periods`) moves only on cash
  deposits/withdrawals ($50 → $21 → $521 → … → $5), never on trading P&L. **This is
  a dormant micro test account, not the documented $25K paper account (DUP995654).**
- **Conclusion:** there is *no live trade record* to evaluate. All strategy
  evidence is backtest-only. Nobody is currently trading this — which, given the
  numbers above, is not a bad thing.

---

## The core problem

1. **No demonstrated edge over the benchmark.** The whole point of an active
   strategy is to beat passive. This one, by its own numbers, does not.
2. **Complexity vastly exceeds evidence.** Agents, ML ensembles, options, regime
   detectors, intermarket analysis, a SaaS app — layered on top of a strategy that
   can't clear buy-and-hold. Complexity has been used as a substitute for edge.
3. **Bull-market survivorship tailwind.** Backtests run on today's mega-cap winners
   (AAPL/NVDA/MSFT…) during a strong bull market. Even so, the strategy *lost* to
   just holding the index. That is a damning result: it underperformed the very
   tailwind it was riding.
4. **Aspirational docs contradict measured reality.** "100X", "$25K→$50K", "wealth
   optimization" documents set targets (100% annual) that are ~15–30x the best
   measured result. Treat all such prose as marketing until a walk-forward proves it.

---

## What would change the verdict (falsifiable next steps)

For a future instance to move this from RED toward GREEN, produce **one** of:

1. A **walk-forward, out-of-sample, honest-cost** backtest (commission + slippage +
   spread modeled) that **beats SPY total return** over the same window — ideally
   across multiple regimes, not just the 2024–25 bull run.
2. A specific, mechanistic **source of edge** with evidence: e.g., the short book
   adds uncorrelated return; a regime filter that sits in cash/SPY during chop and
   only trades when RRS dispersion is genuinely high; an options-overlay that
   harvests premium with defined risk. Each must be tested the same honest way.
3. Evidence that the strategy's value is **risk-adjusted** (materially higher Sharpe
   / lower drawdown than SPY) even if raw return is lower — a legitimate reason to
   prefer it *only if* an investor would actually accept the lower return for the
   smoother ride, stated explicitly.

If none of these materializes after honest effort, the mandate's RED path applies:
recommend a fundamentally different thesis or wind-down. Do not keep polishing a
strategy that loses to the index it trades against.

---

## Blocking infrastructure gaps (fix these to enable real work)

- **Backtests can't fetch data in the remote agent env** — yfinance/Yahoo is blocked
  by the egress proxy. Highest-leverage engineering task: add an **IBKR-MCP-backed
  data source** (or a cached parquet dataset committed to the repo) so walk-forward
  backtests can actually run and be reproduced here.
- **No committed benchmark harness.** There is no script that reports strategy vs
  SPY total return net of costs as its headline. That should be the *first* number
  any backtest prints.
