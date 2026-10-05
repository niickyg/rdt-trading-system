# POST-MORTEM: RRS Strategy & Why The Bot Is Where It Is

> **Status: BOOTSTRAPPED 2026-10-05.** This file was referenced by the operator
> schedule but did not exist. The first operator instance created it from a
> direct reading of the codebase and its own backtest evidence. It is a factual
> assessment, not a defense of the strategy.

## The one-paragraph truth

The RDT Trading System is a large, well-engineered codebase implementing the
r/RealDayTrading Real-Relative-Strength methodology, with scanners, filter
gates, ML add-ons, an options module, and a dashboard. **But as of this writing
there is no honest evidence that it beats SPY buy-and-hold, and strong reason to
believe it does not.** Every performance claim in the repo was produced by
backtests that (a) charged **zero** transaction costs or slippage and (b) never
compared against buy-and-hold. The headline that survives even its own
optimistic framing — **~3.4% annualized** over Feb 2024–Nov 2025 — loses badly
to SPY, which rose far more over that window. The engineering is real; the edge
is unproven.

## Evidence (found in-repo, 2026-10-05)

### 1. The backtests never modeled costs
- `backtesting/engine.py`, `engine_enhanced.py`, `engine_intraday.py`,
  `parameter_optimizer.py`: **no** commission, slippage, spread, or fee logic.
  Entries filled at the exact signal price; stops/targets filled exactly at
  their trigger (ignoring gap/slippage, which hurts stops most).
- `scripts/run_walkforward_v2.py` — the script that produced CLAUDE.md's
  headline 3-way comparison ($815 / $1,267 / $1,716) — has its own inline P&L
  engine with **no cost modeling** either.
- For a strategy whose claimed edge is a few hundred dollars over two years
  across **279 trades**, round-trip costs are not noise — they can be the
  entire "edge." Options trades (this bot trades options too) carry far larger
  per-contract fees and spreads, none of which were modeled.

### 2. The backtests never measured the mission benchmark
- No engine computed SPY buy-and-hold over the test window. `spy_data` was used
  only to compute RRS, never as a benchmark. So "the RDT filters win" was only
  ever **relative to other configurations of the same losing strategy**, never
  relative to just holding the index.

### 3. Even the optimistic headline underperforms
- CLAUDE.md's best config: **$1,716 on $25K over ~2 years = 6.9% total, 3.4%
  annualized, gross of costs.** SPY over Feb 2024–Nov 2025 returned
  substantially more than that. Net of costs the gap only widens.

### 4. Repo hygiene issue that hides breakage
- `utils/secrets.py` is **gitignored** (`.gitignore: *secrets*`) but
  `utils/__init__.py` imports from it. A fresh clone (and CI) therefore cannot
  import `utils` → `config` → most of the app, and `pytest` cannot collect the
  suite. This masks whether tests actually pass on clean checkouts.

## What this means

The methodology may or may not have an edge in the hands of a skilled
discretionary trader. **As an automated system measured honestly, it has not
been shown to.** Before any more features, filters, or ML, the bot needs a
truthful measuring stick: costs in, buy-and-hold benchmark in, out-of-sample
only. Everything else is building on sand.

## The path forward (in priority order)

1. **Honest measurement first** (STARTED 2026-10-05): transaction costs +
   SPY buy-and-hold benchmark + alpha reported by every backtest engine.
2. Re-run the 2-year walk-forward **net of costs vs buy-and-hold** and record
   the real alpha. If negative (likely), that is the finding.
3. Only if a genuine, cost-aware, out-of-sample edge appears: scale it.
4. If, after honest measurement, no configuration beats buy-and-hold:
   **recommend wind-down of active trading** and repurpose the system as a
   research/alerting tool, per the mandate.

See `data/operator_journal/entries/` for session-by-session progress.
