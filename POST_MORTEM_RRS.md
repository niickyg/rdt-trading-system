# POST-MORTEM: The RRS Strategy (reconstructed)

> **Reconstructed 2026-09-11 by the bootstrap operator.** The scheduled operator prompt
> references this file as an existing history of "why the bot is in its current state," but
> it **did not exist in the repository**. This is a best-effort reconstruction from the
> artifacts that *are* committed: `CLAUDE.md`, `data/optimization/optimization_2025-12-29.json`,
> `data/signals/signal_history.json`, `data/signals/signal_metrics.json`, and the
> `ACTIONABLE_100X_STRATEGY.md` / `WEALTH_STRATEGY_100X.md` docs. Treat dates and causal
> claims as inferred, not authoritative. Correct this file as better records surface.

## What the bot does

Implements the r/RealDayTrading (RDT) "Real Relative Strength" methodology: scan a watchlist,
compute `RRS = (stock %chg − SPY %chg) / ATR%`, and take momentum trades (long strong RS,
short strong RW) that pass a stack of filter gates (SPY regime, 50/200 SMA, VWAP, multi-
timeframe alignment, VIX, sector RS, intermarket). Paper broker + IBKR paths exist; an
options module and a large web/SaaS layer were added on top.

## The core problem: the edge is weak, and it is unmeasured

### 1. The backtested edge is thin and likely below buy-and-hold

From `data/optimization/optimization_2025-12-29.json` (180-combo parameter sweep, $25k):

| Metric | Best config | Notes |
|---|---|---|
| Total return | ~6.8% | over the backtest window |
| Win rate | 38% | |
| Profit factor | 1.29 | |
| **Sharpe** | **0.11** | essentially indistinguishable from noise |
| Max drawdown | ~2.4% | |
| Trades | 215 | |

`CLAUDE.md`'s 2-year walk-forward table reports the best ("RDT filters") config at **$1,716
(6.9%) total → ~3.4% annualized**, win rate 49.5%, PF 1.24.

Over the same 2024–2025 window, **SPY buy-and-hold returned dramatically more** (SPY alone
was up ~24% in 2024). A strategy delivering ~3–7% *total* over one-to-two years, with a
Sharpe near 0.1, **does not beat buy-and-hold** — and it does so while incurring active-
trading risk, PDT constraints, execution costs, and operational complexity.

The project's own `ACTIONABLE_100X_STRATEGY.md` concedes this: *"At 1% risk per trade, this
mathematically caps returns around 7%… the strategy is signal-limited, not capital-limited."*

### 2. There is no honest track record — the bot cannot measure itself

- `data/signals/signal_history.json`: **1,986 signals** generated over ~1 month
  (2026-02-03 → 2026-03-05), all `RRS_Momentum`. **Zero outcome fields** — no realized P&L,
  no exit, no win/loss on any of them.
- `data/signals/signal_metrics.json`: **`total_outcomes: 2`** (1 target hit, 1 stop out)
  against those ~2,000 signals. The outcome counters are also internally inconsistent with
  the history file (metrics say 119 short / 1 long; history says 1,687 long / 299 short),
  indicating the counters were reset or were never wired to the same data.
- Two disconnected outcome mechanisms exist and neither yields a benchmarked realized-P&L
  series: `scanner/signal_metrics.py::record_outcome` (manual, driven 2 times) and
  `agents/outcome_tracker.py` (tracks *rejected* signals into the DB to test filter
  strictness — not taken-trade P&L).

**Consequence:** the only quantitative statements anyone can make about this bot come from
in-sample-ish backtests. There is no living, honest measurement of whether the deployed
system makes money. This is the single most important defect.

### 3. Scope drift

`WEALTH_STRATEGY_100X.md` / `ACTIONABLE_100X_STRATEGY.md` respond to the weak edge not by
questioning the edge but by proposing leverage, margin, options, and a **signal-selling /
paid-API / education business** to manufacture a "100% annual return." This is mission drift:
the objective is a profitable *bot*, not a media business, and leverage on a Sharpe-0.11
edge multiplies risk, not edge.

## Honest verdict (as of 2026-09-11)

On the evidence committed to this repo, **there is no basis to claim the RRS bot beats SPY
buy-and-hold**, and weak backtests suggest it materially underperforms on a risk-adjusted
basis. It is also **flying blind** — it does not record the outcomes of its own signals.

## What must happen before any strategy tuning

1. **Build honest measurement first.** A benchmarked, realized-P&L ledger for taken paper
   trades vs SPY buy-and-hold over the identical window, with costs modeled. Until this
   exists, every tuning decision is guesswork.
2. **Re-run walk-forward with realistic costs** and report net-of-cost, out-of-sample
   numbers against the SPY benchmark — not in-sample sweeps.
3. **Then, and only then,** decide whether the edge is real. If it is not, recommend
   wind-down or a fundamentally different approach — honestly.

*See `data/operator_journal/entries/2026-09-11-bootstrap-and-assessment.md` for the session
that produced this reconstruction.*
