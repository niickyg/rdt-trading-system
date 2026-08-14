# POST-MORTEM: The RRS Trading System — Why It Is Where It Is

> Bootstrapped 2026-08-14 by the first autonomous operator instance. This file is the
> honest history of how the bot reached its current state, written to save future
> instances from re-learning it. It will be extended by later operators; append, don't
> rewrite, unless a fact is proven wrong.

## The one-paragraph summary

This is a relative-strength momentum day-trading bot built on the r/RealDayTrading
"RRS" concept: buy stocks outperforming SPY (RRS > 2), short those underperforming,
with ATR-based stops and a fixed 2:1 reward:risk. Over ~two years of development it
accreted an enormous amount of machinery — multi-timeframe gates, VIX/sector/regime/
intermarket/news overlays, an 87-feature ML stack, an options module, a multi-broker
layer, and even a SaaS product skin — **without ever establishing, with reproducible
out-of-sample evidence, that the core signal has a cost-surviving edge.** As of this
writing there is no P&L ledger in the repo and only 2 tracked trade outcomes. The
headline backtest the docs cite (~3.4% annualized) would, if accurate, still lose to
simply holding SPY (~10%/yr; ~+15% over Feb–Aug 2026).

## How it got here (the pattern)

The git and CLAUDE.md history shows a consistent loop:

1. Start with a plausible idea (RRS momentum, per r/RealDayTrading).
2. Backtest is disappointing or ambiguous.
3. **Instead of concluding "weak/no edge," add another layer** — a filter, an overlay,
   a regime model, an ML feature set, a new asset class.
4. Re-backtest (in-sample, one regime, via yfinance), get a slightly better number,
   declare victory in the docs.
5. Repeat. Complexity compounds; validated edge does not.

Symptoms visible in the tree today:
- **No outcome tracking that matters.** `signal_history.json` (1986 signals, Feb–Mar
  2026) records signals but *zero* realized outcomes. `signal_metrics.json`: 2 outcomes.
- **Backtests can't be reproduced in the agent environment.** `run_walkforward*.py`
  and `train_from_history.py` depend on yfinance, which is blocked by the egress proxy.
  Every headline number in CLAUDE.md is therefore currently *unverified*.
- **Complexity that no edge measurement justifies.** 87 ML features, 5+ overlay
  filters, options, SaaS — none tied to a demonstrated, cost-net edge on the base signal.
- **Direction instability.** One metrics snapshot showed 119 shorts / 1 long; the
  larger history is 1687 long / 299 short. The signal's directional bias is regime-
  driven and unvalidated.

## What "profitable" has to mean here

Per the operator mandate: **positive P&L net of honest costs (commission + slippage +
spread), beating SPY buy-and-hold.** Win rate alone is meaningless; a 2:1 R:R needs
only ~33% wins to break even *gross*, but costs raise that bar, and beating SPY raises
it further.

## Empirical results — independent first-touch backtest (2026-08-14)

The first operator ran an independent evaluation of all 1986 historical signals
against **real IBKR daily price data** (yfinance being blocked). Methodology: enter at
each signal's stated entry_price, walk subsequent daily bars, first-touch of stop vs
target (daily high/low), same-bar ambiguity → stop (conservative), timeout →
mark-to-market. Costs modeled at ~$0.05/share round trip. Full script and outputs are
referenced in the 2026-08-14 journal entry.

<!-- RESULTS_PLACEHOLDER: filled in below once simulation completes -->

## Guidance for future instances

- **Do not add a new overlay/feature/model to "fix" a weak edge.** That is the exact
  failure loop above. First prove the base signal has edge, out-of-sample, net of cost.
- **Measure before you build.** The IBKR MCP tools give you real data. Use them.
- **If the edge isn't there, say so** and follow the mandate's escalation/wind-down
  path. That is the valuable outcome, not another green in-sample number.
