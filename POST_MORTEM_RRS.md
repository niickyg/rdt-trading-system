# POST-MORTEM: The RRS Strategy — Why The Bot Is Where It Is

> Bootstrapped 2026-10-21 (operator run 001). This file did not exist; the
> scheduled operator prompt referenced it as "the history of why the bot is in
> its current state," so run 001 created the first version from the evidence
> available in the repo plus independent backtesting. Future runs should append
> dated sections rather than rewrite history.

## The one-paragraph version

The system was built around Real Relative Strength (RRS) as an **intraday**
day-trading signal, then wrapped in an ever-growing stack of filters (SPY gate,
50/200 SMA gate, VWAP gate, lightweight MTF, VIX regime, sector RS, regime-
adaptive thresholds, news, intermarket/Murphy layers) plus an ML overlay. The
repo's own walk-forward result for the best filter configuration is **+6.9% over
~2 years (+3.4% annualized)** — while SPY buy-and-hold over the *identical*
window returned **+40.3% (+20.9% annualized)**. In other words the deployed bot
captured roughly **one-sixth** of what doing nothing would have earned, while
taking on the risk and cost of 279 trades. An in-repo strategy doc
(`ACTIONABLE_100X_STRATEGY.md`) concluded from this that the edge is marginal
("negative Kelly," PF 1.29, 38% win rate) and pivoted toward selling a signal
subscription service and cranking leverage/risk — i.e. away from trading its way
to profit.

**Run 001's independent finding flips the diagnosis:** the *core RRS signal is
not the problem — the filter complexity is.* A dead-simple DAILY swing
implementation of the same RRS formula, with none of the intraday machinery,
produced **+36% to +78%** over 2022–2025 depending on exit style, beating SPY
buy-and-hold on both return and drawdown in the better configs.

## What the repo's own numbers say (CLAUDE.md, walk-forward V2)

| Metric | Baseline | Old Filters | RDT Filters (best) |
|---|---|---|---|
| Total return (2yr) | +3.3% | +5.1% | **+6.9%** |
| Annualized | 1.6% | 2.5% | **3.4%** |
| Win rate | 47.7% | 47.9% | 49.5% |
| Profit factor | 1.12 | 1.12 | 1.24 |
| Trades | 235 | 457 | 279 |

"98% of raw signals are filtered out." The filters reduce trade count and nudge
win rate, but the absolute return stays trivial and **far** below buy-and-hold.

## Independent verification (operator run 001, 2026-10-21)

Tool: `data/operator_journal/research/rrs_validation_harness.py` (yfinance daily
bars, repo's RRS formula, repo core 50-stock watchlist, 10 bps round-trip cost,
1% risk/trade, long-only, SPY hard gate).

**Benchmark omission found:** the repo never compares its backtest to SPY
buy-and-hold. Over the documented window (Feb 2024–Nov 2025) SPY B&H was +40.3%.
The bot's +6.9% is not "a modest edge" — it is a large *under*performance.

**Multi-regime results (1% risk, net of 10 bps costs):**

| Window | SPY B&H | base daily RRS | trend-ride | thr1.75 |
|---|---|---|---|---|
| 2022 bear | **-18.6%** | -5.4% | -8.1% | -12.2% |
| 2022–2023 | +2.6% | +3.5% | +4.1% | +13.6% |
| 2024–2025 (bull) | +43.6% | +35.8% | +57.7% | +58.1% |
| **Full 2022–2025** | **+46.6%** | +37.7% | **+59.8%** (DD 15.8%) | **+77.8%** (DD 22%) |

Every configuration tested was profitable over the full window (PF 1.27–1.90).
The SPY hard gate kept the strategy mostly in cash during 2022, cutting the bear
drawdown to ~1/3 of SPY's. Two simple configs beat buy-and-hold outright.

**Why the deployed system underperforms its own signal (hypothesis):**
- It trades intraday (5m) and exits same-day, giving up the multi-day drift that
  the daily swing version harvests.
- Each added filter gate removes trades near the signal threshold — exactly where
  much of the edge lives — and "98% filtered out" means the surviving 2% is not
  obviously the profitable 2%.
- Complexity was added faster than it was validated against buy-and-hold.

## Honest caveats on the run-001 finding (do NOT overclaim)

1. **Survivorship bias.** The test universe is today's S&P-500 mega-caps —
   survivors. Long-only results are biased upward. A point-in-time universe is
   required before trusting absolute return levels.
2. **Daily proxy, not production.** The production engine is intraday with a DB
   and broker. The harness validates the *signal premise*, not the exact code.
3. **Hindsight in parameters.** `trend_ride` / `thr1.75` were chosen after seeing
   the data. They held up across the larger 2022–2025 window, but a true
   walk-forward (tune on A, test untouched on B) has NOT yet been done.
4. **Bull-market tailwind.** 2024–2025 was strongly up; long-only momentum
   flatters. The 2022 result (beats B&H by preserving capital) is the reassuring
   counter-evidence, but one bear is a small sample.

## Current best estimate of the mission

The bot as deployed does **not** make money worth the risk (loses badly to
buy-and-hold). But there is a **credible, evidence-backed path**: simplify toward
a daily/swing RRS strategy with trend-riding exits and keep only the filters that
demonstrably help (the SPY hard gate does). This is NOT a wind-down situation —
it is a "you built the wrong thing around a good signal" situation. The next step
is rigorous out-of-sample walk-forward validation (with survivorship-bias-aware
universe) before changing any production code.
