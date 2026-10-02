# POST-MORTEM: The RRS Strategy

> Created on operator run 001 (2026-10-02). The scheduled task referenced this file
> as "the history of why the bot is in its current state," but it did not exist. This
> is an **evidence-based** reconstruction built from the data actually present in the
> repo plus fresh market data. It is not folklore — every claim below is reproducible
> with `scripts/evaluate_signal_edge.py` and the cited files.

## TL;DR

The Real Relative Strength (RRS) momentum strategy, as built and as actually run,
**has no demonstrated profitable edge and loses money net of its own stated
stops/targets.** Its best *backtested* configuration also loses badly to simply
holding SPY. The live bot ran for about one month (Feb–Mar 2026) and has been
**dormant ever since** (last scan 2026-03-05). There is no evidence — backtested or
live — that this strategy beats SPY buy-and-hold, which is the mission's bar.

## Evidence 1 — The backtest never beat SPY (it was never compared)

`CLAUDE.md` advertises the 2-year walk-forward result for the best ("RDT Filters")
configuration as a success:

| | RDT Filters (best) | SPY buy-and-hold |
|---|---|---|
| Total return (Feb 2024 → Nov 2025) | **+6.9%** ($1,716 on $25k) | **+37.9%** |
| Annualized | **+3.4%/yr** | **+20.2%/yr** |

SPY numbers computed from adjusted closes (490.22 → 676.10 over the window). The
strategy's *best backtested* config returned **~3.4%/yr** — below the risk-free rate
(~4.5% T-bills in this period) and roughly **1/6th of SPY buy-and-hold**. The original
documentation presented the $1,716 figure without ever stating the benchmark it had to
beat. **A strategy return reported without its benchmark is marketing, not evidence.**

## Evidence 2 — The live signals lost money (realized, first-touch)

The bot persisted every signal it generated to `data/signals/signal_history.json`:
**1,986 signals across just 6 trading days (2026-02-03 → 2026-03-05)**, 48 symbols,
then it stopped. Deduped to **628 unique setups** and simulated each at its own stated
entry/stop/target (first-touch on daily bars, stop assumed first on ambiguous days —
a conservative-to-pessimistic rule):

| Horizon | Win rate | Expectancy | Total |
|---|---|---|---|
| 15 trading days | 23.4% | **−0.313R / trade** | **−196R** |
| 5 trading days | 25.8% | **−0.389R / trade** | **−245R** |

The targets are set at ~2.0× the stop distance, so breakeven needs a **33.3%** win
rate. Realized win rate was **23%**. The negative edge is robust across horizons.
(Reproduce: `python scripts/evaluate_signal_edge.py --horizon 15`.)

### Why it lost — the diagnosis

1. **It chased extended moves.** Bucketing signals by the intraday move already in
   progress at entry (`stock_change_pct`):

   | Extension at entry | n | Win rate | Expectancy |
   |---|---|---|---|
   | 0–3% | 418 | 25.1% | −0.28R |
   | 3–5% | 129 | 19.4% | −0.42R |
   | 5–8% | 56 | **10.7%** | **−0.64R** |
   | 8%+ | 25 | 44.0% | +0.32R (n=25, noisy) |

   Entering longs *after* a stock is already up 3–8% on the day buys the top; normal
   retracement then hits the (closer) stop before the (farther) target. This is the
   single clearest mechanical flaw.

2. **The "market-first" gates did not protect the book.** SPY fell **−6.2%** over the
   signals' forward window (684 → 642, Feb 3 → Mar 26 2026), yet the bot emitted
   **1,687 longs vs 299 shorts** — a long-heavy book into a selloff. The SPY hard gate
   that is supposed to block longs in a bearish tape either failed, was fail-open on
   missing data, or the drawdown developed after entry. Either way, the system behaved
   like naive long exposure, not a hedged relative-strength strategy.

3. **The shorts had no edge either.** In a −6% market, short signals *should* print
   money. They didn't: 57 unique shorts, 24.6% win, **−0.22R**. A relative-weakness
   short book that loses during a 6% market decline has negative alpha and/or stops too
   tight versus ATR noise.

4. **2:1 R:R is the wrong geometry for this hit rate.** A momentum-breakout entry with
   a 1-ATR stop and 2-ATR target needs the move to continue cleanly. In choppy or
   mean-reverting conditions the 2-ATR target rarely fills first. The strategy's own
   `mean_reversion_scanner.py` is disabled "because it contradicts RDT momentum
   philosophy" — but the data says these names mean-revert against the entries.

## Evidence 3 — The bot is dormant

`data/signals/signal_metrics.json`: `last_scan_at = 2026-03-05`. ~7 months of silence
to today (2026-10-02). Whatever ran in Feb–Mar 2026 was stopped and not restarted. We
therefore have **no recent live evidence at all**, and the most recent evidence we do
have is a losing month.

## What this means

By the mission's bar — positive P&L net of costs, beating SPY buy-and-hold — the RRS
strategy **fails on every available piece of evidence**: it lost in live signals, and
its best backtest trails SPY by ~31 points over two years. Before any feature work, the
operator must either (a) demonstrate a specific, validated fix that flips the realized
edge positive, or (b) if fixes keep failing, recommend wind-down. See the latest
journal entry for the current recommendation and the pre-registered next experiment.

## Caveats (intellectual honesty)

- First-touch on daily bars can't see intraday path; same-day stop+target is scored as
  a stop (pessimistic). But the strategy's stops/targets are multi-percent swing levels
  (e.g. entry 32.44 / stop 31.28 / target 34.77), so a daily, multi-day horizon is
  appropriate, and the 5-day horizon was *worse*, not better.
- Fills are modeled at the signal's stated entry; real fills add slippage, so live
  results would be worse, not better.
- Feb–Mar 2026 is one regime (a selloff). A long-biased book will look bad there. But a
  *relative-strength* strategy is supposed to be regime-robust, and the shorts losing
  in a down market removes the "just bad luck in one window" defense.
