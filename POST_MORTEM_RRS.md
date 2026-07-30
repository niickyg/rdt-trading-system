# POST-MORTEM: RRS Trading System — State of the Edge

> **Bootstrapped 2026-07-30** by the first autonomous operator instance. No prior
> post-mortem existed in the repo, so this is the first honest reconstruction of
> "why the bot is in its current state" built from the evidence actually present
> in the repository. It is a living document — later operators should append,
> correct, and date their additions.

## TL;DR (2026-07-30)

- The repo contains **no committed record of realized trade P&L.** The system
  generated signals but never persisted what happened to the *emitted* ones.
  Therefore no one can currently claim, from the repo alone, that the strategy
  made or lost money. Prior claims of profitability (e.g. the walk-forward table
  in `CLAUDE.md`) are **not reproducible from committed data.**
- To get a real read, this operator forward-tested the **1,986 actual signals**
  the bot emitted (Feb 3 – Mar 5 2026) against **real IBKR daily bars**, using a
  first-touch (target vs stop) simulation with honest costs. Results are in the
  "Forward-test" section below.
- Two structural problems surfaced that undermine trust in the live signal stream:
  1. **71% of signals were generated outside US regular trading hours** — RRS
     computed on stale/after-hours prices.
  2. **Outcome tracking only covers *rejected* signals** (and only in the live DB,
     not the repo). Emitted signals have no feedback loop.

## What the repo evidence actually shows

### Signal stream (`data/signals/signal_history.json`)
- 1,986 signal records, Feb 3 – Mar 5 2026, 48 symbols.
- Collapses to **88 distinct daily setups** (same names re-scanned every ~20 min).
- Direction skew in the raw stream: 1,687 long / 299 short.
- `signal_metrics.json`: **880 scans, 120 signals, but only 2 recorded outcomes**
  (1 target hit, 1 stop out). Outcome capture was essentially absent.

### Timing problem (data quality)
- US RTH is 14:30–21:00 UTC. Only **574 / 1,986 (29%)** of signals were generated
  in-session; **1,412 (71%)** were generated overnight / pre-market / after-hours,
  spread evenly across all 24 UTC hours.
- RRS = (stock %chg − SPY %chg) / ATR is an intraday relative-momentum measure.
  Computed when the market is closed, it runs on stale last-prints and is not
  meaningful. A large majority of the "signals" are therefore suspect at the
  point of generation.

### No P&L feedback loop
- `agents/outcome_tracker.py` only tracks **rejected** signals (to test whether
  rejection thresholds are too strict), and writes to the live DB — which is not
  in the repo. There is no mechanism persisting the realized outcome of the
  signals the bot actually acted on.

## Forward-test: did the emitted signals have edge?

**Method.** For each of the 88 distinct daily setups, assume a fill at the
signal's `entry_price` on the first daily bar on/after the signal date, then scan
forward up to a fixed horizon of trading days for the first touch of the signal's
own stop or target (conservative tie-break: if one daily bar spans both, count it
a stop). If neither is touched by the horizon, exit at the last bar's close
(mark-to-market). Real IBKR daily OHLC bars. A friction charge of 0.05R is
subtracted from every trade for commission + slippage. Dollar P&L assumes each
trade risks 1.5% of a static $25,000 (Config C), i.e. $375 per 1R.

**Caveats stated up front.** Daily bars can't resolve intra-bar ordering (hence
the conservative tie-break); entry prices for after-hours signals may be
unrealistic fills; N=88 is a small, one-month, single-regime sample. This is a
directional read, not proof.

**Results (real IBKR daily bars; reproducible in `research/forward_test/`).**

First-touch R-multiples (each trade risks 1R; planned R:R ≈ 1.98):

| Horizon | Win rate | Avg R (net 0.05R cost) | Profit factor |
|--------|---------|------------------------|---------------|
| 5 days | 50.0% | +0.14R | 1.39 |
| 10 days | 45.5% | +0.25R | 1.58 |
| 20 days | 44.3% | +0.27R | 1.57 |

Per-trade **alpha vs SPY** over identical holding windows (10-day horizon, 0.10%
friction/trade):

| Subset | n | Mean trade ret | Mean SPY ret (same window) | Mean alpha | Median alpha | % beat SPY |
|--------|---|----------------|-----------------------------|-----------|--------------|-----------|
| All | 88 | +0.59% | −0.17% | +0.76% | −1.35% | 47% |
| **Long** | 73 | +0.97% | −0.11% | **+1.08%** | **+1.92%** | 52% |
| **Short** | 15 | −1.27% | −0.46% | **−0.81%** | −1.99% | 20% |

**What this does and does NOT show.**

- The aggregate numbers look positive: net-positive expectancy at every horizon,
  and the **long** signals produced positive alpha (median +1.92%) even though SPY
  was flat-to-negative over the same windows — i.e. it is *not* pure market beta.
- **But the sample is fatally concentrated. 79 of the 88 setups (90%) were
  generated on just two adjacent days, Feb 3 and Feb 4 2026.** The winning longs
  are dominated by an energy/materials move (DOW, DVN, SLB). The effective number
  of *independent* observations is ≈ 2, not 73. This is one good sector event, not
  evidence of a repeatable edge. Statistically, we cannot reject "luck."
- The **short** signals are a consistent, unambiguous drag: −1.3% mean return,
  −0.81% alpha, only 20% beat SPY, across the sample. Shorting relatively-weak
  names in this chop lost money.
- Conclusion: **insufficient, over-concentrated evidence — leaning skeptical.**
  The long side is *interesting enough to keep testing on a much larger, multi-
  regime signal set*; the short side looks value-destroying and should be
  scrutinized or disabled; and nothing here justifies risking capital yet.

## Why the bot is where it is (interpretation)

The system is an elaborate, well-engineered *pipeline* (scanner → filters →
agents → execution → dashboards) wrapped around a signal whose **edge was never
measured on realized outcomes.** Engineering effort went into breadth (options,
ML features, dashboards, SaaS overhaul) while the core question — *do these
signals make money net of costs?* — was never answered with committed evidence.
The walk-forward numbers in `CLAUDE.md` may be real, but they cannot be
reproduced from anything in the repo, and they report only ~3.4% annualized,
which would **trail SPY buy-and-hold** — a red flag the project did not act on.

## Open questions for future operators

1. Does the first-touch forward-test show positive, cost-honest expectancy? Over
   what horizon? (See dated entry.)
2. If the RTH-only subset is isolated, is its edge different from the after-hours
   junk?
3. Can the walk-forward backtest (`scripts/run_walkforward_v2.py`) be run and
   reproduced, and does it survive a lookahead-bias audit?
4. Is there any live-DB export that contains realized fills we could analyze?
