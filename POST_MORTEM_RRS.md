# POST-MORTEM: The RRS Strategy vs. Reality

> **Purpose.** This file is required reading for every autonomous operator run
> (see `data/operator_journal/MANDATE.md`). It records *why the bot is in its
> current state* and the central, evidence-backed finding that must anchor every
> future decision. It was first written during the bootstrap run on 2026-10-06.
> It is a living document — append verified findings; do not delete history.

---

## TL;DR (the one thing you must not forget)

The RDT/RRS strategy, in every configuration we have tested, **massively
underperforms simply buying and holding SPY** over the same period — in a market
where SPY was rising. The bot is an elaborate, risk-bearing, full-time machine
that captured **less than a quarter** of what one click into an index fund would
have returned.

| Over 2024-06-10 → 2026-06-23 (~2.03 yrs, $25,000) | Total Return | Annualized | $ Profit |
|---|---|---|---|
| **SPY buy-and-hold** | **+40.64%** | **+18.25%** | **+$10,161** |
| Strategy C — "RDT Filters" (best variant, **gross**) | +9.53% | +4.7% | +$2,383 |
| Strategy B — "Old Filters" | +8.01% | +4.0% | +$2,003 |
| Strategy A — "Baseline" | +6.07% | +3.0% | +$1,517 |

The best strategy variant underperformed SPY by **~13.5 percentage points per
year**, and the +9.53% is **gross of all transaction costs and slippage** (the
backtest engine models none — see below). Net of realistic friction on 277
trades, the real gap is wider.

**Reproduce:**
```bash
python scripts/run_walkforward_v2.py   # strategy A/B/C aggregate returns
python scripts/benchmark_vs_spy.py     # SPY over the identical span
```

---

## How we got here (history)

1. **Premise.** The system implements the r/RealDayTrading methodology: trade
   stocks showing Real Relative Strength (RRS = (stock %Δ − SPY %Δ) / ATR) vs.
   SPY, long the strong / short the weak, with a stack of regime, VIX, sector,
   SMA, VWAP, and multi-timeframe filters layered on top.

2. **Enormous surface area was built.** Agents, an EventBus, an ML ensemble
   (XGBoost/RF/LSTM), options trading, a Black-Scholes engine, intermarket
   analysis, 87 engineered features, dashboards, a SaaS front-end, multi-broker
   failover. `CLAUDE.md` documents all of it in confident detail.

3. **But the edge was never established against the only benchmark that matters.**
   The documented walk-forward table in `CLAUDE.md` reports Config C at "+6.9% /
   3.4% annualized" and frames it as the *winner* ("RDT filters take fewer but
   higher-quality trades"). It **never compares to SPY buy-and-hold.** Framed
   against SPY, "the winner" is a severe loss.

4. **The ML was already known not to help.** `CLAUDE.md` itself records: exit
   predictor 43% accuracy (SKIP), "Rule-based filters provide all measurable
   improvement; ML is advisory-only." So the ML surface area contributes nothing
   to P&L.

5. **The filters barely move P&L and mostly just reduce trading.** In the fresh
   run, the RDT filter stack discards **98.1%** of raw signals (14,671 → 277
   trades) to turn +6.07% into +9.53% gross. That is a rounding error against a
   13.5pp/yr benchmark gap — the filters are optimizing the deck chairs.

---

## Why the number is trustworthy (and its caveats)

**Trustworthy because:**
- It uses the project's *own* backtest harness (`run_walkforward_v2.py`,
  `engine_enhanced.py`), unchanged.
- SPY is measured over the *exact same* trading-day span (index 91..600 of the
  loaded series) on the *same* $25,000, so it is apples-to-apples.
- 277 trades across 6 contiguous quarterly windows and two years of regimes
  (bull, the early-2025 pullback, recovery) — not one lucky/unlucky window.

**Honest caveats (none plausibly close a 13.5pp/yr gap):**
- **Daily bars, not intraday.** The live system trades 5-minute bars; the
  backtest approximates on daily bars, so VWAP and first-hour filters are not
  simulated. But intraday trading adds *more* cost and whipsaw, not less — this
  caveat cuts against the strategy, not for it.
- **No cost model.** `engine_enhanced.py` applies **zero** commission and **zero**
  slippage. The +9.53% is gross. With 277 round trips, even ~0.05%/side slippage
  (realistic for intraday fills) erodes several hundred dollars; the live, net
  figure is lower than the backtest.
- **Survivorship / fixed watchlist.** 30 large-cap names chosen today; mild
  optimistic bias.
- **Long-biased in a bull market.** The strategy is mostly long momentum and
  *still* lost to buy-and-hold *during a bull market* — the most favorable
  possible environment for it. In a flat or bear market the relative case does
  not obviously improve, because the short side has not been shown to add edge.

---

## What this means for the mandate

The mandate is: beat SPY buy-and-hold, net of costs. The current system does the
opposite by a wide margin, with far more operational risk and complexity than
holding an index fund. Three honest paths forward, in priority order:

1. **Find a real edge or admit there isn't one.** Any proposed change must be
   judged by whether it narrows the **13.5pp/yr gap** — not by win rate, profit
   factor, or methodology fidelity. Incremental filter-tuning has already been
   shown to move P&L by ~1-3pp total over two years; that cannot close the gap.
   A credible attempt would have to change something fundamental (instrument
   selection, holding period, use of leverage/options with a *demonstrated*
   edge, or a genuinely different signal), and prove it out-of-sample.

2. **Escalate.** If repeated honest experiments keep failing to beat the
   benchmark, the correct recommendation is to tell the human plainly: *this
   methodology, as implemented, does not beat SPY; continuing to run it risks
   real money to underperform an index fund.*

3. **Wind-down framing.** "Use a low-cost SPY index fund" is, on the current
   evidence, the profit-maximizing recommendation. Stating that is a success of
   the mandate, not a failure of the operator.

---

## Open questions a future instance could test (backlog)

- Does restricting to *only* the highest-RRS decile (fewer, higher-conviction
  trades) ever beat SPY, or just reduce trade count further?
- Is the **short** side a net drag? Split the backtest by direction — if longs
  carry everything, the "relative weakness" thesis is unsupported.
- What is the strategy's beta-adjusted alpha? If returns are just low-beta
  exposure, the honest comparison is to a blend of SPY + cash, which it also
  likely loses to.
- Add a realistic cost model to `engine_enhanced.py` (commission + per-side
  slippage) so every future number is net, not gross. **High value, low effort.**
- Re-run across a bear/flat window (e.g. 2022) to test whether the strategy's
  case is "downside protection" rather than return — and quantify it honestly.
