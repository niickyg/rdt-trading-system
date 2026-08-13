# POST-MORTEM: The RRS Strategy and Why This Bot Is Where It Is

> Reconstructed on 2026-08-13 (operator genesis run) from git history, `CLAUDE.md`,
> the backtest code, and live IBKR account state. No prior post-mortem existed;
> this is the first honest accounting. Future operators: extend this file as the
> story develops — do not rewrite history, append to it.

## The premise

The system implements the **r/RealDayTrading (RDT)** methodology: trade
individual stocks that show **Real Relative Strength (RRS)** against SPY —

```
RRS = (Stock % Change − SPY % Change) / ATR%
```

The thesis: stocks outperforming the index intraday on strong volume tend to
keep outperforming; go long strength, short weakness, in the direction of the
broader market ("market first"). The repo is a genuinely large, competent build
around this idea: a scanner, a 4-gate filter stack (SPY gate → 50/200 SMA → VWAP
→ multi-timeframe alignment), VIX/sector/regime/intermarket overlays, an 87-
feature ML layer, an options module, multi-broker execution, and a web dashboard.

## What the evidence actually says

The headline result carried in `CLAUDE.md`, from a 2-year (Feb 2024–Nov 2025),
6-window walk-forward on a $25K account:

| Config | Total Return | Win Rate | Profit Factor | Annualized |
|--------|-------------|----------|---------------|------------|
| A) Baseline (no filters) | +$815 (3.3%) | 47.7% | 1.12 | 1.6% |
| B) Old filters | +$1,267 (5.1%) | 47.9% | 1.12 | 2.5% |
| C) RDT filters | **+$1,716 (6.9%)** | **49.5%** | **1.24** | **3.4%** |

The filter engineering *is* real and additive: config C beats baseline, takes
fewer/higher-quality trades, and has a shallower worst day. On its own terms the
work is sound. **But the benchmark is not zero — it is SPY.**

Over that same window SPY buy-and-hold returned roughly **+37% (~17–20%/yr)**
(verified via IBKR monthly bars during the genesis run: SPY 563.68 → 772.49 over
~24 months, +37%, plus ~1.3%/yr dividends). The strategy's best configuration
returned **3.4%/yr**. That is a **~5x underperformance versus doing nothing but
holding the index** — and it accepts single-name risk, drawdowns, and active
management to get there.

## The two things that made the picture worse than it looked

1. **The backtests were frictionless.** Until the genesis run, the enhanced
   backtest engine filled every entry, scale-out, and exit at the *exact* signal
   price — zero commission, zero slippage, zero spread. With scaled exits, one
   "trade" is 3–4 fills per side. Modeling realistic costs (5 bps/fill +
   $0.005/share, $1 min) is a meaningful drag on ~279 trades: on the order of
   ~$1–2K over the window — i.e. plausibly most or all of the +$1,716 "edge."
   The genesis run added a default cost model so this can never be hidden again.

2. **There is no live track record.** The IBKR account reachable via MCP during
   the genesis run held **$5 net liquidation, zero positions, and zero trades in
   90 days.** Whatever the docs say about a funded DUP995654 paper account, the
   account the operator can actually see is empty. Every profitability claim in
   this repo rests on *simulation*, not realized paper P&L.

## The ML footnote

Per `CLAUDE.md`'s own status notes, the ML layer is advisory-only and mostly
non-predictive: the exit predictor scored 43% accuracy (barely above random),
and "rule-based filters provide all measurable improvement; ML is advisory-only."
So the ML build, while large, is not the source of any edge.

## Why the bot is "in its current state"

Enormous, careful engineering effort went into refining a strategy whose best
honest result still loses decisively to a passive index — and the tests were
built in a way (frictionless, no live comparison to SPY) that kept that
conclusion from being obvious. The system is not *broken*; it is *beaten by its
own benchmark*. That is the central fact every future operator must confront
before adding one more filter.

## What would actually change the verdict (open questions for successors)

- Does **any** configuration survive honest costs with positive net P&L? (Run
  the now-cost-aware walk-forward when market data is available.)
- Even if net-positive, can it beat SPY **risk-adjusted** over a full cycle, not
  a cherry-picked window? A bull-market sample flatters trend-following longs.
- Is there a lower-frequency or concentration variant whose edge isn't eaten by
  turnover costs? Turnover is the enemy here.
- If the answer stays "no" after an honest, cost-aware, benchmark-relative test:
  the correct recommendation is **wind down active trading and hold SPY.**
