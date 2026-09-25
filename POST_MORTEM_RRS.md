# POST-MORTEM: The RRS Strategy and Why This Bot Is Where It Is

_Bootstrapped 2026-09-25 by the genesis run of the autonomous operator. This file
did not previously exist in the repo; it was reconstructed from the committed
documentation, code, and one independent verification run (see below). Future
operator runs: treat the "Verified facts" section as load-bearing and the
"Reconstructed history" section as best-effort inference from repo artifacts._

---

## The one-paragraph version

This bot implements the r/RealDayTrading "Real Relative Strength" (RRS) momentum
methodology: go long stocks outperforming SPY, short those underperforming, gated
by a stack of market-regime filters. After extensive filter engineering, ML
add-ons, and a Murphy-style intermarket layer, the **best** walk-forward
configuration returned **~6.9% total over ~2 years (~3.4% annualized)** on $25K
paper capital. Over that **exact same window**, simply holding SPY returned
**+42.7% (+21.5% annualized)**. The strategy captured roughly one-sixth of
buy-and-hold, at far higher operational complexity and before honest costs. The
core problem is structural, not a tuning bug: the strategy is signal-limited with
a ~38% win rate and a ~1.29 profit factor, which caps returns near 7% regardless
of how the filters are tuned.

---

## Verified facts (independently checked this session)

- **SPY buy-and-hold, 2024-02-01 → 2025-11-28** (the documented backtest window):
  **+42.7% total, +21.5% annualized, -18.8% max drawdown.** (yfinance,
  auto-adjusted close, fetched 2026-09-25.)
- **QQQ buy-and-hold, same window:** +48.2% total, +24.1% annualized, -22.8% max DD.
- **RDT best config ("C) RDT Filters"), same window** (from `CLAUDE.md` walk-forward
  table, not independently re-run this session): +$1,716 / +6.9% total, ~3.4%
  annualized, 49.5% win rate, 1.24 profit factor, 279 trades, worst day -$257.
- **Gap:** SPY beat the best strategy config by **~6.2x on total return**, with a
  comparable drawdown profile. The strategy did not beat, or come close to
  beating, buy-and-hold.

## Reconstructed history (from repo artifacts)

1. **Original thesis:** RRS momentum per r/RealDayTrading — trade only in the
   direction of relative strength, "market first," filter aggressively.
2. **Filter engineering era:** SPY hard gate, 50/200 SMA gate, VWAP gate,
   lightweight MTF, VIX regime, sector RS, regime-adaptive thresholds, news
   sentiment, and a Murphy intermarket layer (TLT/UUP/GLD/IWM). Documented result:
   filters cut ~98% of raw signals and lifted the best config from ~3.3% to ~6.9%
   total over two years — real but small, and still far below SPY.
3. **ML era:** StackedEnsemble (XGBoost + RF + LSTM), 87 features, drift detection.
   Per `CLAUDE.md`: the exit predictor is 43% accurate (barely above random and
   marked SKIP); ML is explicitly "advisory-only" and "rule-based filters provide
   all measurable improvement." ML did not move the needle on edge.
4. **The "100X" pivot:** `WEALTH_STRATEGY_100X.md`, `ACTIONABLE_100X_STRATEGY.md`,
   and `WEALTH_OPTIMIZATION.md` acknowledge the ~7% cap explicitly ("signal-limited,
   not capital-limited") and propose closing the 14.6x gap to a 100% target via
   (a) leverage/margin/concentration and (b) **non-trading revenue** — a signal
   subscription service, API access, and education. This is the critical drift to
   flag: **selling signals is not the trading bot being profitable.** Leverage
   multiplies a losing-to-SPY edge; it does not create edge.

## Root cause (why tuning won't fix it)

A momentum day-trading strategy with a **~38% win rate and ~1.29 profit factor at
1% risk/trade** is mathematically pinned near ~7% annual return. The docs derive
this themselves. The only levers are: more trades (signal frequency is the
binding constraint — loosening filters historically *lowered* quality), higher
win rate (ML failed to deliver it), or higher profit factor (the filters already
squeezed this to ~1.24–1.29). None of these has produced an edge that survives
honest costs, and none approaches SPY. Meanwhile buy-and-hold required zero trades,
zero slippage, zero borrow, and one taxable event.

## What "honest costs" would further subtract

The 6.9% figure is a backtest that likely under-counts: per-share/leg commissions
across ~279 round trips, real slippage on entries and stops, short-borrow fees on
the short book, and short-term capital-gains tax treatment on essentially all P&L.
Net-of-cost, the live edge is plausibly at or below zero.

## The honest conclusion

On the mission's own terms — *positive P&L net of honest costs, beating SPY
buy-and-hold* — the RRS strategy as built **fails**, and the failure looks
structural rather than fixable by more filter/ML tuning. See the genesis journal
entry for the recommendation (escalation / re-scoping toward strategies with a
plausible path to beating a passive benchmark, or wind-down of the active-trading
premise).
