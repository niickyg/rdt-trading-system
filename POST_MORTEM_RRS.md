# POST-MORTEM: RRS Trading System — Honest State of the Bot

> Bootstrapped by the operator agent on **2026-09-28**. The operator task referenced
> this file as required reading, but it did not exist in the repo. This is a first,
> evidence-based reconstruction from what is actually in the repository. It is not a
> record of a single blow-up event — it is an honest audit of where the bot stands
> and why. Future instances should append dated sections rather than rewrite history.

## TL;DR

- The bot has **effectively no real trading track record.** The persistent metrics
  file (`data/signals/signal_metrics.json`) shows **2 tracked outcomes ever**
  (1 target hit, 1 stop-out) across 880 scans and ~2,000 recorded signals.
- The signal data is **stale by ~6 months.** Last scan timestamp is
  `2026-03-05`; today is `2026-09-28`. Nothing here reflects recent market behavior.
- The only performance evidence is **backtests**, and by the repo's own documented
  numbers the best configuration returns **~6.9% total over ~2 years (~3.4%
  annualized)** on $25K (CLAUDE.md, walk-forward V2 table).
- **Over that same 2024–2025 window, SPY buy-and-hold returned far more** (SPY rose
  strongly through that period). The walk-forward scripts **never computed the
  buy-and-hold benchmark**, so the mission's explicit success bar — *beat SPY
  buy-and-hold* — has, until now, never actually been measured in-repo.
- The repo's "100X" strategy docs (`ACTIONABLE_100X_STRATEGY.md`, `WEALTH_*_100X.md`)
  openly concede the trading strategy "mathematically caps returns around 7%" and
  pivot to **selling signals as a service** to hit the return target. That is not
  trading edge; it is a different business, with its own risks.

## What the evidence actually says

### 1. No forward validation
Backtests are the entire evidence base. There are 2 tracked live/paper outcomes.
A strategy with no forward track record and only in-sample/backtested numbers must
be treated as **unproven**, not as "profitable but small."

### 2. The core math problem (from the repo's own analysis)
`ACTIONABLE_100X_STRATEGY.md` states: profit factor ~1.29, win rate ~38%, ~215
trades/year at 1% risk → ~$1,700/yr on $25K. This is a **thin, momentum
signal-limited** edge. Even taken at face value, it does not clear buy-and-hold in a
bull market, and thin edges are exactly the ones that evaporate once realistic
commissions, slippage, and spread are subtracted — costs the walk-forward harness
does not fully model.

### 3. The success bar was never measured
The mission is to **beat SPY buy-and-hold, net of honest costs.**
`scripts/run_walkforward_v2.py` loaded full SPY price data but reported only the
strategies' returns and a naive annualized estimate — never the buy-and-hold
return over the same windows. You cannot claim to beat a benchmark you never compute.
*(Fixed 2026-09-28: a `spy_buy_hold()` benchmark + verdict was added to that script.)*

### 4. Risk-as-returns temptation
The path from ~3–7% to a "100X"/100%-return goal in the repo's own docs runs through
**more risk or non-trading revenue**, not better edge. This is the classic failure
mode. The mandate now hard-blocks using risk increases as a profitability lever.

## Why the bot is in this state (hypotheses, not certainties)

1. **Momentum RRS on a 30-large-cap watchlist in a strong-trend bull market is
   dominated by beta.** Relative-strength stock-picking adds little over just owning
   the index when the index itself is ripping. The filters (SPY gate, SMA, MTF, VIX)
   improve risk-adjusted numbers but shrink trade count, capping absolute return.
2. **The system has been engineered heavily (agents, ML, options, dashboards, a SaaS
   overhaul) without first establishing that the underlying signal beats buy-and-hold.**
   Effort went into infrastructure and productization ahead of proving edge.
3. **No closed feedback loop.** With 2 tracked outcomes, the "learning" and
   "adaptive" components have essentially no data to learn from.

## What would actually change the verdict

- Run the walk-forward with the new buy-and-hold benchmark and **honest cost model**
  (per-trade commission + slippage + half-spread). Report net-of-cost alpha vs SPY.
- If net alpha is ≤ 0 across windows, the honest recommendation is **wind down active
  trading in favor of index buy-and-hold**, or restrict the bot to regimes where RRS
  has demonstrated out-of-sample alpha (e.g. flat/choppy or bear markets, where
  relative strength should matter more than in a trending bull run).
- Build a real paper-trading feedback loop and accumulate forward outcomes before
  trusting any backtest.

## Append log
- **2026-09-28** — Initial post-mortem written. Operator infra bootstrapped;
  buy-and-hold benchmark added to walk-forward V2. No trading/risk logic changed.
