# POST-MORTEM: The RRS / r/RealDayTrading Strategy

> **Provenance:** This file was reconstructed on 2026-11-05 by the genesis
> operator instance. The scheduled operator task referenced `POST_MORTEM_RRS.md`
> as pre-existing history, but no such file existed in the repo or its git
> history. What follows is reconstructed **only from evidence in the repository
> itself** (`CLAUDE.md`, `DEPLOYMENT_SUMMARY.md`, the backtest engines, the
> walk-forward scripts, the "100X" strategy docs) plus authoritative SPY price
> history pulled read-only from the IBKR market-data connector. Where a claim is
> the project's own documentation rather than something I reproduced, it is
> labeled as such. Nothing here is invented.

---

## 1. What the system is

A long/short, intraday-to-swing directional trading bot built around **Real
Relative Strength (RRS)**:

```
RRS = (Stock %Change − SPY %Change) / ATR%
```

It scans a watchlist, keeps names that are strong/weak relative to SPY, pushes
them through four sequential filter gates (SPY regime, 50/200 SMA, VWAP, a
lightweight multi-timeframe check), then layers on VIX/sector/regime/intermarket
adjustments and an advisory ML stack (87 features, a stacked ensemble). Average
holding per `DEPLOYMENT_SUMMARY.md` is ~3.7 days; ~240 trades/yr.

The philosophy is taken from the r/RealDayTrading subreddit: trade stocks showing
real relative strength vs the market ("market first"), take fewer but
higher-quality momentum setups, avoid mean reversion.

## 2. The central problem: it does not beat buy-and-hold

The project's **own best documented result** is the 2-year, 3-way walk-forward in
`CLAUDE.md` ("Walk-Forward Backtest Results, 2 Years, $25K Capital", Feb 2024 –
Nov 2025):

| Metric | Baseline | Old Filters | **RDT Filters (best)** |
|---|---|---|---|
| Total Return | $815 (3.3%) | $1,267 (5.1%) | **$1,716 (6.9%)** |
| Win Rate | 47.7% | 47.9% | **49.5%** |
| Profit Factor | 1.12 | 1.12 | **1.24** |
| Annualized | 1.6% | 2.5% | **3.4%** |

Now the benchmark, over the **identical window** (SPY, from IBKR daily/monthly
history):

- SPY 2024-02-01 open **$484.63** → 2025-11-03 close **$683.39**
- **Price return: +41.0%** over the window; **~+43%** including dividends
- **Annualized: ~21.7%**

**The strategy's best out-of-sample configuration returned +6.9% while simply
holding SPY returned ~+41%.** The system underperformed its own benchmark by
roughly **34 percentage points** over the test window. By the mandate's success
criterion (beat SPY buy-and-hold net of costs), this is a decisive failure — and
it fails using the project's *most favorable* documented number.

For context, SPY has since run to ~$766.81 (2026-11-02), **+58% from the Feb 2024
start**. The opportunity cost of running this system instead of holding the index
has only widened.

## 3. Why the real picture is worse than 6.9%

Two compounding problems make even the modest +6.9% optimistic:

### 3a. The backtest ignores transaction costs entirely
The intraday engine computes P&L as raw `(exit_price − entry_price) × shares`
(`backtesting/engine_intraday.py`, the `_close_*` paths). A grep of the
walk-forward scripts and the engine finds **no commission, slippage, spread, or
fee modeling** anywhere. For a strategy doing ~240 round-trips/yr, realistic
costs (bid-ask spread crossing + slippage, even at IBKR's low commissions) plausibly
consume a large fraction of a 6.9%-over-2-years gross edge. **Every backtest
number this project has produced is GROSS.** The net number is unknown and is
almost certainly lower — possibly near zero or negative.

### 3b. In-sample optimism vs out-of-sample reality (overfitting)
`DEPLOYMENT_SUMMARY.md` (dated 2025-12-29, "AGGRESSIVE" profile) advertises:

> Expected Return: **6.84% annually** · Win Rate 50.2% · Profit Factor 1.35 · Max DD 2.0%

That "6.84%/yr" sits next to a tuned parameter set (RRS threshold dropped 2.0→1.75,
take-profit 2.0x→1.5x ATR, "relaxed criteria") — i.e. an optimized, in-sample
figure. The more rigorous **out-of-sample walk-forward says 3.4%/yr.** The gap
between the optimistic single-backtest number and the honest walk-forward number
is the classic signature of overfitting, and the aggressive risk profile
(3%/trade, 6% daily loss, 20% position size) amplifies the downside if the edge
is not real.

### 3c. The ML layer does not add measurable edge
Per `CLAUDE.md`'s own "ML Model Status": the exit predictor is 43.3% accurate
("SKIP — barely above random"), and the conclusion is stated outright:
**"Rule-based filters provide all measurable improvement; ML is advisory-only."**
So the 87-feature ensemble, Murphy intermarket layer, regime detection, etc., are
machinery that has not been shown to pay for itself out-of-sample.

## 4. Structural reason it struggles

The strategy is directional day/swing trading with ~50% win rate and profit
factor ~1.2. That is a thin edge even if real. Over 2024–2025 it was competing
against a strongly trending market (SPY +21%/yr). A near-coin-flip directional
system with modest payoff asymmetry, frequent trades, and uncounted costs is
structurally unlikely to beat simply owning the up-trending index. The "fewer,
higher-quality trades" framing (98% of raw signals filtered out) improved trade
quality at the margin (PF 1.12→1.24) but nowhere near enough to close a 34-point
gap to the benchmark.

## 5. Honest status

As of the genesis assessment: **there is no evidence in this repository that the
system is profitable net of costs, and clear evidence that it badly underperforms
SPY buy-and-hold on its own best documented out-of-sample test.** This is not yet
grounds for wind-down — no one has yet (a) measured it net of honest costs or
(b) exhausted credible improvement hypotheses — but it sets a high bar: any future
change must be judged against the +41% (now +58%) the owner would have made doing
nothing.

## 6. What this means for future operator runs

1. **Honest cost accounting is prerequisite #1.** Until the engine deducts
   commission + slippage + spread, no number is trustworthy. (See the next
   journal entry's agenda.)
2. **Always report vs the SPY benchmark** for the same window. A positive return
   is not a win if SPY beat it.
3. **Suspect every in-sample number.** Demand walk-forward / out-of-sample.
4. **Simplify before adding.** The ML/options/"100X" machinery has not earned its
   keep in the evidence. Removing unvalidated complexity is a legitimate path.
5. **Keep a running honest scoreboard** in the journal so a stateless successor
   can see the trend across runs, not just a point-in-time claim.
