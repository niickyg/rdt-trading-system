# POST-MORTEM: The RRS Strategy and How This Bot Got Here

> **Status:** BOOTSTRAPPED 2026-09-14 by the first operator run. The operator
> prompt assumes this file already exists ("the history of why the bot is in its
> current state"); it did not. This is an honest reconstruction from the
> artifacts in the repo (`CLAUDE.md`, `DEPLOYMENT_SUMMARY.md`,
> `ACTIONABLE_100X_STRATEGY.md`, `data/signals/*`, backtest scripts). Future
> operators should correct it where they find better evidence.

---

## The one-paragraph version

This project set out to systematize the r/RealDayTrading "Real Relative Strength"
(RRS) discretionary methodology into an autonomous bot. Over many iterations it
accumulated enormous machinery — a 4-gate filter pipeline, an 87-feature ML
stack, options trading, intermarket analysis, regime models, a full web
dashboard and SaaS overhaul. What it never produced is **evidence of a tradable
edge**. The best honest backtest returns ~3.9%/yr; SPY buy-and-hold returned
~15%/yr over the same era. The system's own strategy doc admits the Kelly
criterion is **negative**. And the live/paper system has recorded a grand total
of **2 trade outcomes**. The complexity grew; the edge never arrived.

## What RRS is, and why it's seductive

```
RRS = (Stock % Change - SPY % Change) / ATR
```

The idea (from r/RealDayTrading): a stock that holds up or advances while SPY
falls — or advances *more* than SPY on an up day — is showing "real relative
strength," and is a higher-probability long. It's intuitive, it matches how good
discretionary traders talk, and it produces lots of signals. That's exactly why
it's dangerous to systematize: it *feels* like edge, so it's easy to keep
building on top of it without ever checking whether the edge is real, net of
costs, out of sample.

## The evidence, laid out honestly

### 1. The backtests never beat buy-and-hold
`CLAUDE.md` (Walk-Forward, 2 years, $25K) reports three configs; the best,
"RDT Filters," made **$1,716 (6.9%)** over ~21 months — **~3.9% annualized**.
`DEPLOYMENT_SUMMARY.md` claims **6.84%/yr** for an "aggressive" profile.

Benchmark (verified 2026-09-14 via IBKR, SPY conid 756733, monthly closes):
SPY **573.76 → 764.29** over the trailing 2 years = **+33.2% (~15.4%/yr
price-only, ~16.6%/yr total return).** The strategy's *best case* loses to a
one-line SPY buy-and-hold by **roughly 5x** — and does so while taking on
single-name risk, trading costs, and the operator's time. No backtest doc in the
repo puts the strategy next to SPY buy-and-hold. That omission is the whole story.

### 2. The strategy's own math says the edge is negative
`ACTIONABLE_100X_STRATEGY.md`, to its credit, does the arithmetic:
- Win rate ~**38%**, profit factor ~**1.29**, ~215 trades/yr at 1% risk.
- Kelly = (1.55 × 0.38 − 0.62) / 1.55 = **−0.02 → NEGATIVE.**
A negative Kelly means there is no positive fraction of capital to bet; sizing up
raises risk of ruin without raising expected return. The document's honest
conclusion ("the current edge is marginal") is then abandoned in favor of a plan
to hit "100% annual return" partly by **selling a signal-service subscription** —
i.e. revenue from other people, not from trading. That pivot is the tell: when
the trading edge isn't there, the plan quietly becomes something other than
trading.

### 3. The live/paper record is empty
`data/signals/signal_metrics.json` (last scan 2026-03-05):
- 880 scans, 120 signals generated, **802 scans produced nothing**.
- **`total_outcomes: 2`** — one `target_hit`, one `stop_out`.
Two outcomes is not a track record; it's noise. Every "deployed / ACTIVE /
optimized" claim in the repo rests on backtests and hope, not realized paper P&L.

### 4. A structural short bias in a rising market
Of 120 signals: **119 short, 1 long.** In an era where SPY rose ~15%/yr, a bot
that almost exclusively shorts is positioned to bleed. Whatever the RRS logic was
doing in early 2026, it was fighting the primary trend — the opposite of the RDT
"trade with the market" principle the system claims to follow.

### 5. Complexity as a substitute for edge
The repo shows a clear pattern: each time returns disappointed, the answer was
*more machinery* — lightweight MTF, VIX regime, sector RS, regime-adaptive
thresholds, intermarket (Murphy), 17 new "murphy_features," an options module, a
SaaS front-end. None of it is tied, in any document, to a measured improvement in
net-of-cost P&L versus SPY. This is the core anti-pattern: adding parameters to a
model with no demonstrated edge fits noise and *feels* like progress.

## Why the bot is "in its current state"

Because the loop was: *build → backtest → get a low single-digit return →
attribute the shortfall to a missing feature → build that feature → repeat*,
without ever once asking "does this beat just owning SPY, net of costs, out of
sample?" The honest answer to that question — available from the artifacts the
whole time — is **no**.

## What would actually change the verdict

1. **An honest edge harness**: walk-forward, out-of-sample, net of realistic
   commission + slippage + spread, reported side-by-side with SPY buy-and-hold on
   identical capital and dates. Until this exists, no strategy claim is credible.
2. **Working outcome tracking**: the paper system must record real fills and P&L,
   so future operators reason from data, not from `signal_metrics.json` with 2
   rows.
3. **A pre-registered hypothesis discipline**: state the edge and its success
   criterion *before* testing, to stop the p-hacking that produced 87 features.
4. Absent all that, the correct move is **wind-down of the systematic-trading
   ambition** and an honest recommendation to the owner that this methodology, as
   built, does not beat index buy-and-hold.

## Bottom line

RRS is a fine *lens* for a discretionary human trader. As the core of an
autonomous bot it has, across every artifact in this repo, failed to demonstrate
an edge that beats owning SPY — and its own arithmetic says the edge is negative.
The bot's state is not a bug to be patched; it's the accumulated cost of never
running the one comparison that mattered.

*— Reconstructed by the first operator run, 2026-09-14. Correct me with better data.*
