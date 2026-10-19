# POST-MORTEM: The RRS Strategy vs. Reality

> **STATUS: BOOTSTRAPPED BY THE OPERATOR ON 2026-10-19.**
> The scheduled operator task instructed reading `POST_MORTEM_RRS.md`, but no
> such file existed. This is the first honest post-mortem, assembled from the
> repository's own documents and code plus one freshly computed benchmark. It
> should be extended (not overwritten) by future operator instances.

---

## TL;DR

The RDT Real Relative Strength (RRS) intraday strategy, by the repository's own
best backtest, returns roughly **+6.9% over ~21 months (~3.4% annualized)** on
$25,000 — and that figure **does not deduct trading costs** and was **never
compared to a benchmark.**

Over the **exact same window** (2024-02-01 → 2025-11-28), **SPY buy-and-hold
returned +42.7% (+21.5% annualized)** — turning $25,000 into ~$35,682 vs. the
strategy's ~$26,716. (SPY figure computed fresh on 2026-10-19 via yfinance,
adjusted close.)

**The strategy loses to one-click buy-and-hold by ~36 percentage points over
the period it was tuned on, while taking on hundreds of round-trip trades,
intraday execution risk, and large operational complexity.** This is the
central fact the mission exists to confront.

## The numbers, side by side

Window: 2024-02-01 → 2025-11-28 (~666 calendar days). Capital: $25,000.

| | RDT strategy (best "Config C") | SPY buy-and-hold |
|---|---|---|
| Total return | +6.9% (~+$1,716) | **+42.7% (~+$10,682)** |
| Annualized | +3.4% | **+21.5%** |
| Round-trip trades | 279 | 1 |
| Execution/slippage cost | **not modeled** | ~0 |
| Benchmark comparison | **absent** | — |
| Ending value on $25k | ~$26,716 | ~$35,682 |

Sources: strategy figures are from `CLAUDE.md` ("Walk-Forward Backtest Results,
2 Years") and `ACTIONABLE_100X_STRATEGY.md` / `WEALTH_STRATEGY_100X.md`
("Current State: 6.84% annual return"). SPY figure computed fresh.

## Why the reported strategy numbers are optimistic

1. **No cost model in the canonical walk-forward.** `scripts/run_walkforward_v2.py`
   — the script that produced the headline 3-way comparison — applies **no
   commissions and no slippage** anywhere. A momentum strategy taking 279
   round-trip intraday trades pays real slippage; omitting it inflates the
   reported edge.
2. **No benchmark.** Neither `run_walkforward_v2.py` nor `run_walkforward.py`
   computes a buy-and-hold comparison. SPY appears only as the RRS denominator
   and as a trend gate — never as the yardstick the mission actually cares
   about. `run_backtest.py` *mentions* slippage but, by its own comments, only
   "documents" it in reporting and does not apply it to fills.
3. **The win rate is structurally low** (~38% in the aggressive configs, ~49.5%
   in Config C) with a profit factor near 1.2. Even taken at face value, the
   math caps returns in the low single digits at 1% risk — a point the repo's
   own `ACTIONABLE_100X_STRATEGY.md` concedes ("mathematically caps returns
   around 7%").

## How the repo proposed to "fix" this (and why it's out of scope)

The `*_100X.md` / `WEALTH_*.md` documents target a 100% annual return ($25k →
$50k). Their proposed mechanisms are **not honest trading profitability** and
are designated anti-goals in `MANDATE.md`:

- **Risk inflation**: `QUICK_START_100X.md` advertises an `.env` with
  `MAX_RISK_PER_TRADE=0.03`, `MAX_DAILY_LOSS=0.06`, `MAX_POSITION_SIZE=0.20`.
  Raising variance on a ~1.2-profit-factor strategy raises the odds of ruin, not
  the expected return.
- **Leverage / leveraged ETFs**: manufacturing returns from an unproven edge.
- **"Signal service revenue"**: both 100X docs assume ~half the dollar target
  comes from *selling signals / API access / education* to other people.
  Selling signals from a strategy that loses to buy-and-hold transfers the
  shortfall to customers; it is not the bot becoming profitable.

The committed `.env.example` currently holds the **conservative** defaults (1%
risk, 3% daily loss, 10% position, `AUTO_TRADE=false`) — good. But the 100X docs
actively push the aggressive profile, and `DEPLOYMENT_SUMMARY.md` records an
"AGGRESSIVE" profile as having been deployed. A future operator should verify the
*live* `.env` (not committed here) has not drifted to the aggressive values.

## What would actually change the verdict

For the strategy to earn its complexity it must, net of honest costs, beat SPY
buy-and-hold (or at least beat a simple SPY trend-following rule) **out of
sample, across multiple independent windows.** Nothing in the repository
currently demonstrates this, because the comparison has never been run. The
first, cheapest, highest-integrity step is therefore to **make the comparison
measurable** — see the 2026-10-19 journal entry. Until a net-of-cost,
benchmark-relative backtest exists and is favorable, every "optimization" is
tuning an unmeasured quantity.

## Open questions for the human

1. Is the goal honest trading alpha, or is the "signal service" pivot actually
   desired? (It changes everything about what this operator should build.)
2. What execution assumptions are realistic for the intended broker/account
   (IBKR commissions ≈ $0.005/share or tiered; expected slippage on the traded
   universe)? This operator used a conservative 0.10% round-trip placeholder.
3. Has the live `.env` drifted to the aggressive risk profile?
