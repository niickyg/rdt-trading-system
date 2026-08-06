# 2026-08-06 — Bootstrap the operator system + first honest edge test

**Operator instance:** claude-opus-4-8[1m] (first-ever run)
**Branch:** operator/2026-08-06
**Mission verdict:** NO EDGE FOUND YET (a suggestive long-momentum signal exists in-sample, but "beats SPY net of costs" is unproven; the numbers that looked spectacular are a fill artifact)

---

## Context: this was the bootstrap run

None of the files the scheduling prompt told me to read existed in the repo or its git
history: no `data/operator_journal/`, no `MANDATE.md`, no `LATEST.md`, no `POST_MORTEM_RRS.md`.
I am the first operator instance. So this session did two things:
1. **Built the missing operator infrastructure** (constitution + post-mortem + journal).
2. **Did one real piece of edge measurement** — the highest-value thing available.

## What I assessed

The one decision-relevant question (per the mandate): *does the RRS signal actually have
positive expectancy on real forward prices, net of costs — enough to beat SPY?* The repo had
1,986 emitted signals (`data/signals/signal_history.json`, 2026-02-03 → 2026-03-05) with
entry/stop/target but **no outcome tracking**, and only **2 realized trade outcomes** ever
recorded system-wide. That untapped signal set + IBKR daily price history = the cheapest way to
get a real answer.

## What I did

Fetched daily OHLC bars for all 48 signal symbols + SPY via the IBKR MCP tools (a subagent;
artifact committed), then ran three deterministic tests I wrote:
- `scripts/evaluate_signal_edge.py` — naive first-touch barrier test entering at the signal's
  stated `entry_price`.
- `scripts/edge_control_test.py` — adversarial control: same symbol/direction/geometry, random
  entry DATE, to strip out market drift.
- `scripts/edge_realistic_fill.py` — enters at the **next bar's open** (an achievable fill).

## Evidence / numbers (method + limitations stated)

**Window:** 2026-02-09 → 2026-08-05. **SPY buy-and-hold over window: +10.9%.** (Strong bull tape.)

| Test | All | Long | Short |
|------|-----|------|-------|
| **Naive** (enter at signal `entry_price`) | **+1.04R** | +1.38R | −0.89R |
| Random-date control (drift baseline) | +0.03R | +0.04R | −0.03R |
| **Realistic fill** (next-day open, fresh 2:1) | **+0.34R** | **+0.51R** | **−0.63R** |

Net of a 0.10R round-trip cost assumption, realistic-fill longs are ~**+0.41R**.

**The naive +1.0R is a fill artifact — do not trust it.** Diagnostics: only **23.9%** of
signals have a *fillable* `entry_price`; **59%** of the time the stock has already gapped past
it favorably, mean gap **+10.5%**. Mechanism: signals are re-emitted every scan as a stock runs
intraday (e.g., DOW = 93 emissions over 2 days, entry 28.88→32.67), so `entry_price` is an
early-intraday price while the forward test starts next day. Entering at that price is fantasy.

**The realistic long edge looks real, not just beta:** the random-date control (same names, same
bull window) prints ~0R, so +0.51R for longs is above drift. It's **broad, not outlier-driven**
(top 5% of long trades = only 20% of total R; median long outcome +1.38R). This is the RDT
thesis — strong relative-strength stocks show short-horizon continuation.

**The short side is a net loser here (−0.6 to −0.9R)** — shorting relative-weak stocks in a
+10.9% market. Shorts drag the blended expectancy down.

**Limitations (why this is NOT yet "beats SPY"):**
- One regime only (Feb–Aug 2026, strong bull). Long-momentum is expected to work here.
- Effective sample is small & redundant: 48 names × ~20 days, heavily repeated emissions.
- Per-trade +0.5R ≠ account return: with max 8–10 concurrent positions and heavy signal
  overlap, realized compounding is far lower. No capacity/overlap-constrained portfolio sim run.
- Prices were fetched+normalized by an LLM subagent → treat as estimates (0.35% of bars have a
  minor OHLC inconsistency; aggregate direction is robust to this, precise cents are not).
- Daily bars can't resolve true intrabar stop-vs-target sequencing (I counted same-bar both-touch
  as a loss — conservative).

## What changed in the repo

- **NEW** `data/operator_journal/MANDATE.md` — the operator constitution.
- **NEW** `POST_MORTEM_RRS.md` — reconstructed history from repo artifacts.
- **NEW** `scripts/evaluate_signal_edge.py`, `scripts/edge_control_test.py`,
  `scripts/edge_realistic_fill.py` — the edge tests (reproducible).
- **NEW** `data/operator_journal/artifacts/2026-08-06_ibkr_daily_prices.json` — price data.
- **NEW** this entry + `LATEST.md`.
- **No `risk/` changes.** No config, broker, or trading-logic changes. Nothing that could affect
  live behavior — this session was measurement + docs only. `AUTO_TRADE`/`PAPER_TRADING`
  untouched.

## Honest read

The system's own docs (negative Kelly, ~6.8% backtest) were too pessimistic about the *long*
signal and the naive `signal_history` backtest is too optimistic (fantasy fills). The truth is
in between: **RRS longs show a genuine short-horizon momentum edge (~+0.4R net) in a bull
regime; RRS shorts lose in that regime.** That is a real thread — but a per-trade R edge in one
bull month is a long way from "realized P&L that beats SPY buy-and-hold net of costs across
regimes," which remains **unproven**. SPY returned +10.9% over the window for free.

There is also a concrete, actionable defect worth the human's attention: **any backtest or live
logic that keys off the emitted `entry_price` is using an unfillable price** (stale intraday
value on re-emitted signals). This likely explains part of the gap between rosy backtests and
the near-empty live track record.

## Hand-off to the next instance

**Build a capacity-constrained, cost-realistic portfolio backtest** over ≥2 years / multiple
regimes:
- Entries at next-bar open (or modeled intraday fill), NOT `entry_price`.
- Enforce max concurrent positions, one-position-per-name, and realistic spread/slippage/commission.
- Test **longs-only** vs long+short (the data says gate shorts off when SPY is bullish).
- Produce a realized equity curve and compare directly to SPY buy-and-hold over the same dates.

If that curve beats SPY net of costs out-of-sample → flip verdict to ON TRACK and move toward
sizing. If it doesn't → the honest call is RECOMMEND WIND-DOWN. Do not add filters/ML before
this question is settled.
