# POST-MORTEM: RRS Strategy — State of the Bot

> **Reconstructed 2026-09-18 by the first operator run.** The original
> `POST_MORTEM_RRS.md` referenced by the scheduled operator prompt did not
> exist in this repository. This is **not** the original history — it is an
> honest reconstruction from the evidence that *does* exist in the repo
> (`CLAUDE.md`, `data/signals/*`, the backtest scripts, git log). Where a claim
> is inferred rather than known, it says so. Future runs should extend, not
> trust blindly.

## What the bot is

An autonomous, agent-based trading system implementing the r/RealDayTrading
methodology: Real Relative Strength (RRS) momentum scanning, layered filter
gates ("market first"), optional ML validation (advisory only), and paper
execution via IBKR. Full architecture in `CLAUDE.md`.

## Timeline (from evidence)

- **Initial commit → March 2026:** Rapid build-out. Git log shows scanner
  hardening, IBKR delayed-snapshot quotes, options module, dashboards, and a
  "SaaS product overhaul" (landing/pricing/login pages). The product surface
  grew large and multi-purpose.
- **Feb 3 – Mar 5, 2026:** The only window with recorded signal activity.
  `data/signals/signal_history.json` holds 1,986 signal snapshots (601 unique
  trade ideas after de-duping re-emitted signals), 48 symbols.
- **Mar 5, 2026 → present (Sep 18, 2026):** **No recorded activity.** Signal
  data is ~6.5 months stale. Whether the bot ran and didn't log, or stopped, is
  unknown from the repo.

## The core problem: the bot could not measure itself

`data/signals/signal_metrics.json` records **2 outcomes total** (1 target hit,
1 stop-out) against 120 counted signals. Not one of the 1,986 signal snapshots
carries an outcome / exit / P&L field. **The system generated signals but never
systematically recorded what happened to them.** You cannot make profitable a
thing you never measured. This — not the strategy — was the central failure.

## What the documentation claimed vs. what was verifiable

- `CLAUDE.md` reports a 2-year walk-forward: best config "RDT Filters" =
  **+6.9% total / ~3.4% annualized**. Note: **3.4% annualized underperforms SPY
  buy-and-hold (~10% long-run).** Even the bot's own best documented result did
  not beat the benchmark the mission requires.
- The walk-forward scripts (`scripts/run_walkforward*.py`) print to stdout but
  **persist no results file**, so those documented numbers are **not
  reproducible from the repo** as-is.
- Several repo-root docs (`WEALTH_STRATEGY_100X.md`,
  `ACTIONABLE_100X_STRATEGY.md`, `QUICK_START_100X.md`) promise "100X" outcomes.
  These are **aspirational marketing, not evidence**, and are in tension with
  the mission's demand for honest, benchmark-relative P&L. Treat with skepticism.

## First real measurement (this run — 2026-09-18)

Because outcomes were never recorded, the first operator run built
`scripts/operator_signal_forwardtest.py` and forward-tested the 601 unique
recorded signals against **real historical daily bars**, next-session-open
entry, realistic gap fills, and 10 bps round-trip costs. After excluding 24 DD
signals corrupted by a stock-split price-basis mismatch:

- 577 trades, **win rate 64.5%**, **avg +1.38% net/trade**, **profit factor
  2.63**, over a window when **SPY fell −5.7%**.
- The edge is **entirely in the longs** (522 trades, 66% win, +1.53%). Shorts
  (55 trades, 47% win, −0.06%) were breakeven-to-negative.

**Caveats (important):** one month, one regime; daily bars not intraday fills;
577 overlapping trades cannot all be taken under the 8-position cap, so the raw
sum is not a realizable equity curve; this measures *signal* edge, not full
end-to-end system P&L. This is a **hint of edge, not proof.** See
`data/operator_journal/entries/2026-09-18-bootstrap.md` for full detail.

## Standing recommendations

1. **Fix outcome tracking first.** Nothing else matters until every signal's
   realized result is recorded automatically. This is priority #1.
2. **Validate the long-signal edge out-of-sample** across more months/regimes
   before trusting it.
3. **Question the short signals** — current evidence says they add no edge.
4. **Ignore the "100X" docs** as a basis for decisions.
