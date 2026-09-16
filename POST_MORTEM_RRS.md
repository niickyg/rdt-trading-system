# Post-Mortem: The RRS Strategy — Why the Bot Is Where It Is

> **Provenance.** This document was *reconstructed on 2026-09-16* by the autonomous
> operator because the `POST_MORTEM_RRS.md` referenced by the operator mandate did
> not exist in the repository or its git history. It is assembled from **primary
> sources committed to this repo** (cited inline), not from prior memory. Where it
> states a fact, that fact is traceable to a file. Where it draws a conclusion, the
> conclusion is labeled as such. If an authoritative post-mortem surfaces later, it
> supersedes this reconstruction.

## The core finding, stated plainly

The RRS (Real Relative Strength) momentum strategy, as implemented and documented
in this repo, **has at best a marginal edge and does not clear the bar the mandate
sets** (positive P&L net of honest costs, beating SPY buy-and-hold). The strongest
evidence for this comes from the repo's *own* strategy documents.

## Evidence (from primary sources)

### 1. Best documented backtest ≈ 6.8% annual — and that is *before* honest costs

- `ACTIONABLE_100X_STRATEGY.md`: "Best backtest: 6.8% annual return (~$1,700
  profit)"; "profit factor of 1.29 and 38% win rate."
- `WEALTH_STRATEGY_100X.md`: "Current State: 6.84% annual return ($1,711 profit)."
- `DEPLOYMENT_SUMMARY.md` (deployed 2025-12-29, "AGGRESSIVE" profile): "Expected
  Return: 6.84% annually … Win Rate: 50.2% … Profit Factor: 1.35."
- `CLAUDE.md` walk-forward table (2yr, $25K): RDT filters "Annualized 3.4%."

Even taking the most flattering figure (6.84%), it is a **backtest**, and SPY
buy-and-hold has historically returned roughly ~10%/yr. A ~3–7% backtested return
that ignores realistic slippage/commission on ~241 trades/yr very plausibly goes
to zero or negative after honest costs. **Conclusion:** the documented edge does
not beat buy-and-hold.

### 2. The repo's own math says the edge is near-zero (negative Kelly)

`ACTIONABLE_100X_STRATEGY.md` computes the Kelly criterion for the strategy:

```
win_rate = 0.38 ; win_loss_ratio = 70/45 = 1.55
kelly = (1.55*0.38 - 0.62)/1.55 = -0.02   # NEGATIVE
```

and states: "The Kelly Criterion is slightly negative, meaning the current edge is
marginal. Increasing position size actually increases risk of ruin without
improving returns." **This is the crux.** A (near-)negative Kelly means there is no
reliable edge to compound.

### 3. There is essentially no live/paper track record to validate anything

- `data/signals/signal_metrics.json`: `total_scans: 880`, `total_signals: 120`,
  but `total_outcomes: 2` (`target_hits: 1`, `stop_outs: 1`).
- Outcome tracking has captured **two** closed trades. That is not a sample; it is
  noise. Every profitability claim in this repo is **backtest-only**.

### 4. Signal generation is heavily filtered and direction-skewed

- Raw scanner history (`data/signals/signal_history.json`, Feb 3 – Mar 5 2026,
  1986 signals): 1687 long / 299 short, mean RRS 1.75.
- Emitted/gated metrics (`signal_metrics.json`): 119 short / 1 long, with
  `scans_with_no_signals: 802` of 880.
- The filter gates (SPY hard gate → SMA → VWAP → MTF) drop ~98% of raw signals and
  *invert* the long/short mix. This is consistent with a bearish-SPY window (SPY
  gate blocking longs) — i.e. the gates behaving as designed — but it also means
  the strategy's realized exposure is dominated by whatever the market regime
  allows, and there is no evidence the surviving trades are profitable net of cost.

### 5. Data is stale

All signal data ends **2026-03-05**. As of this reconstruction (2026-09-16) that
is ~6 months old. The bot does not appear to have produced fresh scanner output in
the committed data since then. Any "current performance" claim is unverifiable.

## Where the project went instead of fixing the edge (a warning)

Faced with a ~7% backtested, negative-Kelly strategy, the repo's strategy docs did
**not** conclude "there is no edge." They pivoted to:

- **Cranking risk**: `DEPLOYMENT_SUMMARY.md` deployed an "AGGRESSIVE" profile —
  3% risk/trade (from 1%), 10 positions, 20% max position size. The repo's own
  Kelly analysis says raising size on a negative-edge strategy *increases risk of
  ruin without improving returns*. This was deployed anyway (paper).
- **Leverage / new markets**: adding leveraged ETFs, options, crypto/futures.
- **Selling the signals**: `ACTIONABLE_100X_STRATEGY.md` — "trading alone cannot
  achieve 100% returns … The signal service is not a 'nice to have' - it is
  essential." Deciding to *sell* signals you can't profitably trade is the
  classic tell of a strategy with no edge.

**Conclusion (labeled as judgment):** the strategic direction drifted from "find a
real edge" to "monetize/leverage a non-edge." The mandate exists to resist exactly
this drift.

## What would actually change the verdict

To overturn "no demonstrated edge," a future operator needs, in order:

1. **An honest, cost-adjusted backtest** — realistic per-trade commission +
   slippage + spread on the actual fill model — that still beats SPY
   buy-and-hold over the same window. Reproduce it, commit the numbers.
2. **Out-of-sample confirmation** — the outcome tracker actually recording dozens+
   of closed paper trades, with win rate / profit factor / net P&L that matches
   the backtest, not just 2 outcomes.
3. **A comparison line** — same-period SPY buy-and-hold return, side by side.

Until (1)–(3) exist, the correct posture is: **the bot has no demonstrated edge;
do not increase risk, add leverage, or monetize signals; either find a real edge
under honest costs or recommend wind-down.**
