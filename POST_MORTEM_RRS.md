# Post-Mortem: The RRS Strategy

> Bootstrapped by operator run 001 (2026-11-18). The scheduled operator prompt
> referenced this file as existing history, but it was never in the repo. This
> is an honest reconstruction from the code, the committed docs, the signal
> files, and a fresh independent backtest. Future runs: append, don't rewrite.

## What the bot is

An autonomous system built around **Real Relative Strength (RRS)**:

    RRS = (Stock % Change - SPY % Change) / ATR

Longs when a stock is strong vs SPY (RRS > ~2), shorts when weak (RRS < ~-2),
layered with a large stack of filters (SPY gate, 50/200 SMA gate, VWAP, MTF,
VIX regime, sector RS, intermarket, regime-adaptive thresholds), an options
module, an ML ensemble (advisory only), and a Flask dashboard. The philosophy is
drawn from r/RealDayTrading.

## The core problem: the edge was never established

The system's own `ACTIONABLE_100X_STRATEGY.md` contains the key admission:

> "The Kelly Criterion is slightly negative, meaning the current edge is
> marginal. Increasing position size actually increases risk of ruin without
> improving returns."

That is the whole story. A strategy with a negative/marginal Kelly has no
demonstrated edge. Everything built on top — more filters, ML, options, a SaaS
product — was added before that foundational question was answered.

## What actually happened (timeline)

- **2025-12 → 2026-03:** Rapid development. Scanner, filters, IBKR integration,
  options module, ML, dashboards.
- **2026-03-05/06:** Last substantive commits. The final push was a *"SaaS
  product overhaul"* — landing/pricing/login pages, onboarding, toasts — i.e. a
  pivot toward **selling signals** rather than proving the signals make money.
- **2026-03-05:** Last recorded scan. `signal_metrics.json` shows **880 scans,
  120 signals (119 short, 1 long), and only 2 tracked trade outcomes** (1 target
  hit, 1 stop out). Essentially no live track record.
- **2026-03 → 2026-11:** Dormant. No commits, no scans.
- **2026-11-18 (run 001):** The connected IBKR account shows **net liquidation
  $5, zero positions, zero trades YTD** — not the $25K paper account described in
  CLAUDE.md. There is no meaningful live P&L to evaluate.

## Independent backtest (run 001)

Using the repo's **own** walk-forward V2 engine, extended with a SPY benchmark
(`scripts/run_walkforward_vs_spy.py`), over 6 windows spanning 2024-07 → 2026-08,
$25K fresh per window, **no commissions or slippage modeled**:

| Config          | Total return | Annualized | Profit factor | Trades |
|-----------------|-------------:|-----------:|--------------:|-------:|
| Baseline        |       +6.84% |      ~3.4% |          1.27 |    234 |
| Old Filters     |       +6.21% |      ~3.1% |          1.15 |    424 |
| RDT Filters     |      +10.57% |      ~5.2% |          1.38 |    274 |
| **SPY B&H**     |  **+44.78%** | **~22.4%** |             — |      1 |

The best config loses to SPY buy-and-hold by roughly **4x annualized**. SPY beat
every config in aggregate. And this flatters the strategy: no trading costs,
a hindsight large-cap watchlist (survivors), and no cross-window path risk.
Full data: `data/operator_journal/evidence/2026-11-18-walkforward-vs-spy.json`.

The filter stack is not worthless *relative to baseline* — RDT filters roughly
double baseline's risk-adjusted result and the "98% of signals filtered out"
design does improve trade quality. But "better than a bad baseline" is not the
bar. The bar is SPY, and the strategy is far below it.

## Why it fails (hypotheses, not yet disproven)

1. **No real alpha source.** Daily-bar relative strength on liquid large-caps is
   a crowded, well-arbitraged signal. The filters reduce bad trades but do not
   manufacture an edge that isn't there.
2. **Directional, not market-neutral.** It still ate drawdowns in up-markets
   (e.g. flat/negative while SPY rose +14.5% in one window), so it is not
   providing uncorrelated return that could justify low absolute performance.
3. **Costs would turn marginal into negative.** 274–424 trades/span with a
   profit factor near 1.1–1.4 is exactly the regime where realistic slippage and
   commissions erase the thin gross edge.
4. **Effort went to the wrong layer.** Dashboards and a SaaS funnel were built
   before the edge was proven. Selling signals with no demonstrated edge is the
   worst possible outcome for everyone involved.

## Where that leaves us

Do not deploy capital. Do not increase risk or add leverage. Treat this as a
research project with one gate: *produce reproducible, cost-aware evidence of an
edge over SPY, or wind down.* See `MANDATE.md` and the latest journal entry.
