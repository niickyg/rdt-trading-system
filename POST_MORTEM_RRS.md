# POST-MORTEM: The RRS Strategy and the State of the Bot

> **⚠️ BOOTSTRAPPED FILE — PENDING HUMAN CONFIRMATION**
>
> The scheduled operator task instructed the operator to read `POST_MORTEM_RRS.md`
> as "the history of why the bot is in its current state." No such file existed in
> the repository on 2026-10-27. This document was reconstructed by the first
> operator instance from primary sources already in the repo: `CLAUDE.md` (the
> audit history), the project's own strategy docs, the committed model/signal
> metrics, and a fresh walk-forward backtest run that day. If a real post-mortem
> exists elsewhere, replace this file with it.

---

## TL;DR

The RDT bot implements a Real Relative Strength (RRS) momentum day-trading
strategy. After two years of development, multiple audits, an ML layer, and a
battery of filters, **its best honest result is ~6–7% gross annualized return —
and it loses to SPY buy-and-hold by more than 2x over the same recent window.**
The project's own documentation concedes this and pivots the "wealth" plan from
trading alpha to *selling signals to subscribers*. The ML layer is statistically
indistinguishable from a coin flip. This is not a tuning problem; it is an
**edge** problem.

---

## 1. What the strategy is

```
RRS = (Stock % Change - SPY % Change) / ATR
```
Long when a stock is strongly outperforming SPY (RRS > ~2), short when strongly
underperforming. Signals pass four sequential "market-first" filter gates (SPY
regime, 50/200 SMA, VWAP, multi-timeframe), then VIX / sector / regime / news /
intermarket adjustments. The philosophy is from r/RealDayTrading: trade
relative strength, "market first", momentum over mean-reversion (mean-reversion
is explicitly disabled).

The engineering is genuinely substantial and, after the Feb/March 2026 audits,
largely sound: thread-safe singletons, config validation, cost-free but
coherent backtest engine, security hardening, options module, etc. **The
problem has never been the code quality. It is whether the signal has edge.**

## 2. The documented performance ceiling

The project's own strategy documents state the ceiling plainly:

- `ACTIONABLE_100X_STRATEGY.md`: *"Best backtest: 6.8% annual return (~$1,700
  profit)… profit factor of 1.29 and 38% win rate… the strategy is
  signal-limited, not capital-limited."*
- `WEALTH_STRATEGY_100X.md`: *"Current State: 6.84% annual return ($1,711
  profit)."* And its growth plan reaches "100%" only by adding **signal-service
  revenue ($10–15k)** and **more leverage** — i.e., not by improving trading
  alpha, which it admits caps at ~$8–15k/year.
- `CLAUDE.md` walk-forward table: best config 6.9% total return over 2 years
  (≈3.4% annualized), win rate 49.5%, profit factor 1.24.

When a project's own "how to 10x returns" document concludes the only path is to
**sell the signals to other people**, that is the authors telling you the
trading edge is not there.

## 3. The ML layer is noise

Committed metrics (`models/ensemble/metrics.json`, `models/training_metrics.json`):

- Ensemble cross-validated **AUC = 0.54** (0.50 = random). Train AUC 0.99 with
  CV AUC 0.54 is textbook overfitting. Reported train precision/recall/F1 = 0.0.
- Regime detector (HMM): **silhouette score −0.087** (negative = clusters worse
  than random), degenerate regime distribution (1030/1056 samples in one state).
- `CLAUDE.md` itself: *"Rule-based filters provide all measurable improvement; ML
  is advisory-only,"* and the exit predictor is 43% accurate ("barely above
  random — SKIP").

The ML adds complexity and a false sense of sophistication, not edge.

## 4. Live signal generation is degenerate

Committed `data/signals/signal_metrics.json` (last scan 2026-03-05):

- 880 scans, 120 signals, **119 short : 1 long.** A near-exclusively-short book
  in a market that then rose is structurally positioned to bleed.
- **Only 2 signals ever had a tracked outcome** (1 target hit, 1 stop). There is
  effectively **no live track record** — the "performance" is entirely backtest.

## 5. Fresh walk-forward run (2026-10-27, this operator session)

`scripts/run_walkforward.py`, 30 mega-cap watchlist + SPY + 11 sector ETFs + VIX,
daily data 2025-08-28 → 2026-10-26. Evidence saved at
`data/operator_journal/walkforward_2026-10-27.log`.

| Window | Baseline | Filtered |
|---|---|---|
| W1 2026-01-08→05-18 | +5.76% | +5.82% |
| W2 2026-05-19→09-25 | +0.01% | −0.42% |
| W3 2026-09-28→10-26 | −0.87% | −0.06% |
| **Aggregate (201 td)** | **+4.90% (~6.1% ann.)** | **+5.35% (~6.7% ann.)** |

**SPY buy-and-hold, same window (2026-01-08 → 2026-10-26): +11.87% (~15.1% ann.).**

Three damning observations:
1. **The bot captured less than half of buy-and-hold** while taking day-trading
   risk, short exposure, and operational complexity.
2. **The entire return came from one window** (Jan–May). It was flat-to-negative
   for the following ~5 months while SPY kept climbing. That is a lucky window,
   not a robust edge.
3. **The filters made it worse on the metrics that matter:** profit factor fell
   1.54 → 1.34, max drawdown doubled ($450 → $900), worst day got worse. The
   filters mostly just traded more.

## 6. The costs are not even counted

`backtesting/engine_enhanced.py` models **zero commission and zero slippage**
(grep confirms: no commission/slippage/fee terms anywhere in `backtesting/`). The
~6.7% is therefore **gross**. With ~90–170 trades/year plus multi-leg scaled
exits, realistic transaction costs and slippage would erode an already-
sub-benchmark return further. The watchlist is also 30 hand-picked surviving
mega-caps, which flatters the result (survivorship bias).

## 7. Why the bot is in its current state — the honest history

1. A plausible, well-documented methodology (RRS / r/RealDayTrading) was
   implemented with real engineering rigor.
2. Backtests showed a small positive return (~7% gross). This was interpreted as
   "it works, now optimize," rather than "it's below buy-and-hold, does the edge
   survive costs and out-of-sample?"
3. Response to the weak edge was to **add complexity** (ML, 8+ filter layers,
   options, intermarket analysis) and later to **add risk** (the 2025-12-29
   "AGGRESSIVE" deployment: 3% per trade, 20% position size, RRS lowered to
   1.75). More complexity did not create edge; more leverage only amplifies
   variance around a ~0 edge.
4. The "path to 100%" documents quietly concede trading won't get there and
   pivot to a subscription signal-service business.

## 8. The uncomfortable conclusion

On the evidence, **the RRS day-trading strategy as implemented does not beat SPY
buy-and-hold, and probably does not beat it net of honest costs in any regime
tested here.** The one caveat worth testing: the strategy is pitched as
regime-agnostic via shorts, so it *might* outperform in a sustained bear or
choppy market — but the live signal skew (119:1 short in a rising market) and the
one-window-carries-everything backtest give little confidence even there.

The next operator instances should treat "make this specific strategy profitable
by tuning" as a near-exhausted hypothesis, and instead spend their effort on
**falsification and honest comparison** (realistic costs, out-of-sample,
regime-segmented, vs buy-and-hold), escalating a wind-down recommendation to the
human if the answer keeps coming back the same. See the journal entries for the
running record.
