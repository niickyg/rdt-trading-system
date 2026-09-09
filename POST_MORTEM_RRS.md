# POST-MORTEM: The RRS Strategy — Why the Bot Is Where It Is

> **Provenance note.** The scheduled operator prompt referenced this file as
> pre-existing history, but no such file existed anywhere in the repository or its
> git history as of 2026-09-09. This document is an **honest reconstruction by the
> first operator instance**, built only from verifiable repo evidence (git log,
> committed docs, and the recorded signal logs). Where something is inference rather
> than fact, it is marked *(inferred)*. It should be corrected as better evidence
> surfaces — not treated as settled lore.

---

## What the system is

An autonomous, agent-based day/swing-trading bot implementing the r/RealDayTrading
(RDT) methodology: **Real Relative Strength (RRS)** to rank stocks against SPY,
layered filter gates ("market first"), optional ML advisory signals, options, and
multi-broker execution. Full architecture: `CLAUDE.md`.

## The core mechanic

`RRS = (Stock %Δ − SPY %Δ) / ATR%`. RRS > threshold → long candidate; < −threshold
→ short candidate. Signals then pass sequential fail-open gates (SPY regime →
50/200 SMA → VWAP → multi-timeframe) plus VIX / sector / regime / intermarket
adjustments. The design philosophy is sound and faithful to RDT.

## The evidence trail (verifiable)

1. **The project's own walk-forward (documented in `CLAUDE.md`).** Best config
   ("RDT Filters") over 2 years / 6 quarterly windows / $25K:
   Total return **+6.9%** (**~3.4% annualized**), win rate 49.5%, profit factor
   1.24, 279 trades. The filters help *relative to a no-filter baseline* (+3.3%).

2. **`DEPLOYMENT_SUMMARY.md` (2025-12-29)** claims +6.84% **annually**, 2.0% max
   drawdown, PF 1.35 — and describes an "AGGRESSIVE" profile (3% risk/trade, 6%
   daily loss). These numbers are **more optimistic and internally inconsistent**
   with the sober walk-forward table, and read as marketing. *(inferred: an earlier
   optimization pass over-fit and its headline numbers were carried forward
   uncritically.)* There is also a family of `*_100X_*` "wealth" docs whose framing
   should be treated with heavy skepticism.

3. **The recorded signal log is real but the outcome tracking is essentially
   empty.** `data/signals/signal_history.json` holds **1,986 signals across 48
   symbols, 2026-02-03 → 2026-03-05**. But `signal_metrics.json` records only
   **2 tracked outcomes** (1 target, 1 stop). In other words: **the bot has
   generated thousands of signals and almost never recorded what happened to them.**
   There is no committed, honest forward P&L record in the repo.

## The central problem

- **The benchmark is not being cleared.** Even the *sober* internal number
  (~3.4%/yr) is **well below SPY buy-and-hold**, which returned far more over the
  backtested span. A strategy that underperforms the index it trades against, while
  taking on single-name and timing risk, has **not demonstrated an edge**.
- **Optimistic docs outnumber evidence.** Deployment/wealth docs assert returns the
  reproducible backtest does not support.
- **Almost no ground-truth outcomes were ever captured**, so "is it working?" has
  been answered by backtests (self-consistent, over-fit risk) rather than by what
  the live signals actually did.

## What has NOT been done (and should be)

- An **independent** evaluation of the bot's *own recorded signals* against real
  subsequent prices, net of costs, benchmarked to SPY. (The first operator instance
  began exactly this — see `data/operator_journal/entries/2026-09-09-bootstrap.md`.)
- Honest cost accounting (commission + slippage; daily-bar stop-slippage is
  understated).
- Reconciliation of the contradictory performance claims down to one number that
  can be reproduced on demand.

## Working conclusion (to be tested, not assumed)

The RRS/RDT machinery is well-engineered but there is **no credible evidence yet
that it beats SPY buy-and-hold net of costs**, and there is a documented pattern of
optimistic reporting outrunning reproducible results. The operator's job is to
resolve this empirically: measure the real edge, cut what doesn't earn its keep,
and if the edge is not there, say so plainly and recommend wind-down.
