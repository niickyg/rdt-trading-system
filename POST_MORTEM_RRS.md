# POST-MORTEM: The RRS Trading System

> Bootstrapped 2026-07-27 by the first autonomous operator run. The scheduled operator
> prompt referenced this file as pre-existing history; it did not exist, so this is a
> reconstruction from the codebase, its own strategy docs, the recorded signal metrics,
> and the live IBKR paper account. Where a claim is inferred rather than measured, it says so.

## What the system is

An autonomous trading bot built around **Real Relative Strength (RRS)**:

```
RRS = (Stock % Change − SPY % Change) / ATR%
```

Long when a stock is strongly outperforming SPY (RRS > ~2), short when underperforming.
Layered on top: SPY market-regime gate, 50/200 SMA gate, VWAP gate, multi-timeframe
alignment, VIX sizing, sector RS, intermarket (Murphy) analysis, an ML ensemble
(advisory only), and an options module. It is a large, elaborate codebase.

## The core problem, in the system's own words

`ACTIONABLE_100X_STRATEGY.md` (authored before this operator existed) states plainly:

- Best backtest: **~6.8% annual return** (~$1,700 on $25K).
- **Win rate ~38%, profit factor ~1.29.**
- **Kelly criterion ≈ −0.02 (negative).** By its own math, the edge is marginal-to-negative,
  and increasing size raises risk of ruin without raising return.
- Its proposed path to the stated "100% return" goal is roughly half *trading* and half
  **"signal service revenue"** — i.e. selling signals to other people. That pivot is the
  tell: when a strategy's answer to "how do we make money" is "sell the strategy," the
  trading edge is not carrying its own weight.

The walk-forward table in `CLAUDE.md` shows the RDT-filtered variant at **6.9% over 2
years (~3.4% annualized)**. Over that same 2024–2025 window, SPY buy-and-hold materially
outperformed. **A ~3.4% annualized strategy loses to its own benchmark before you even
subtract honest slippage.** The filters improved the strategy relative to baseline, but
"less bad than baseline" is not "beats SPY."

## What the live evidence shows (measured 2026-07-27 via IBKR MCP)

- **Time-weighted return since inception (late Feb 2026): −61.5%** (YTD == 1Y).
- NAV path: ~$50 → funded up to ~$521 (Mar 20) → bled to ~$477 by end of June → dropped to
  **$5** on Jul 1, flat ever since. (This connected paper account is tiny — $5–$500 — not
  the $25K referenced in CLAUDE.md; treat absolute dollars with suspicion, but the TWR is
  scale-independent and it is deeply negative.)
- **0 open positions. 0 trades in the last 90 days.** The bot is not trading. It is inert.
- `data/signals/signal_metrics.json`: across **880 scans**, only **120 signals** fired, of
  which **119 were SHORT and 1 was LONG.** Only **2 outcomes were ever recorded** (1 target
  hit, 1 stop-out). Last scan: **2026-03-05** — nearly five months stale.

## The three findings that matter

1. **No demonstrated live edge.** The account is down ~61% TWR and has essentially no
   trade sample (2 recorded outcomes). There is no evidence base claiming profitability —
   there is evidence of the opposite.

2. **The bot is dormant.** Zero trades in 90 days, signal scanning stopped in March.
   Whatever the code does, it is not currently doing it. Code changes are moot until the
   system is actually running and generating a trade record again — that is a *human/infra*
   problem the operator cannot fix from a remote checkout.

3. **The signal engine is pathologically skewed.** 119 shorts to 1 long is not a market
   observation, it is almost certainly a *defect* — a filter gate (SPY hard gate / 50-SMA
   gate) or a sign error effectively vetoing every long. In a market where SPY rose over
   the recording window, a momentum system that can only find shorts is broken. This is the
   most testable, highest-leverage lead for a future run.

## Honest conclusion

On the mission's own terms — positive P&L net of costs, beating SPY buy-and-hold — **this
system has not demonstrated an edge and is currently not trading.** Its best honest
backtest trails SPY; its live account is deeply negative; its signal generation is skewed
in a way that suggests a bug rather than an insight.

This is not a counsel of despair — it is a starting diagnosis. But the correct posture is
**skeptical and diagnostic**, not "add more features." The next operator's job is to prove
or kill the edge, not to decorate it. See MANDATE.md §6: two of three escalation criteria
are already met.

## Leads for future runs (ranked)

1. **Diagnose the 119:1 short skew.** Trace `save_signals()` gates in
   `scanner/realtime_scanner.py`. Is a gate or sign convention vetoing all longs? This is
   cheap to investigate and could explain a lot.
2. **Establish an honest cost model.** Any backtest quoting >0% must subtract realistic
   spread + slippage + commission. Rebuild one backtest on IBKR `get_price_history` data
   (yfinance is blocked) with costs made explicit.
3. **Confirm whether the system is even supposed to be running.** 90 days no trades may be
   intentional (paused by the human) or a silent failure. Flag for the human either way.
4. **Resist the "sell signals" pivot.** It is out of scope for the mission and masks the
   absence of a trading edge.
