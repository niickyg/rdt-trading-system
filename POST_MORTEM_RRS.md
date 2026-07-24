# POST-MORTEM: The RRS Day-Trading Strategy

> Authored 2026-07-24 by the first autonomous operator instance during bootstrap.
> This is the honest history of why the bot is in its current state, grounded in
> the live account and the system's own documents — not in its marketing copy.
> Append corrections below; do not rewrite.

## The one-paragraph version

The system implements the r/RealDayTrading "Real Relative Strength" (RRS)
methodology: scan for stocks moving stronger/weaker than SPY, normalize by ATR,
gate with a stack of filters (SPY trend, SMA, VWAP, MTF, VIX, sector, regime,
intermarket), and day-trade the survivors. After ~5 months of paper trading the
result is unambiguous: **the live paper account is down ~61.5% time-weighted since
inception and now holds ~$5, while SPY buy-and-hold returned ~+7.6% over the same
window.** The strategy's own optimization output shows a **negative Kelly
criterion**. The bot has placed **no trades in the last 90 days** — it is dormant.
The strategy does not have a demonstrable edge, and the repo's plans to reach
"100% annual returns" quietly depend on non-trading revenue and on increasing
risk — the exact behavior that drew the account down.

## Evidence (2026-07-24)

### 1. The live paper account (IBKR, ground truth)
- `get_account_summary`: net liquidation **$5**, no margin, 3/3 day-trades unused.
- `get_account_positions`: **empty**.
- `get_account_trades` (DAYS_90): **empty** — no trading activity for 3+ months.
- `get_pa_performance_all_periods` (TWR): cumulative return from 2026-02-26
  inception to 2026-07-24 is **−0.615 (−61.5%)**. The equity path shows an early
  bleed (50 → 21), a deposit up to ~521, a slow monthly decline (521 → 477.5),
  then a drop to 5 around 2026-07-01 where it has sat flat since. The −61.5% is
  time-weighted, so it reflects investment performance, not the cash flows.

### 2. SPY buy-and-hold benchmark, same window
- `get_price_history` (SPY conid 756733): ~685.99 (late Feb) → 738.18 (Jul 24) =
  **+7.6%** over the window (~+19% annualized).
- **The bot underperformed the trivial alternative by ~69 percentage points** and
  destroyed the account, while doing nothing for the last quarter.

### 3. The system's own documents admit the problem
From `ACTIONABLE_100X_STRATEGY.md` (in this repo):
- "Best backtest: 6.8% annual return." "profit factor of 1.29 and 38% win rate."
- Its own Kelly calculation: `kelly = -0.02 # NEGATIVE!`, with the note "the
  current edge is marginal. Increasing position size actually increases risk of
  ruin without improving returns."
- Its "path to 100%" allocates the **majority of the target to non-trading
  revenue** (Signal Service $10k, API Access $2k of a $25k goal) and to a
  **3× risk** "aggressive config" (`max_risk_per_trade: 0.03`, `max_daily_loss:
  0.06`) plus leveraged ETFs (TQQQ/SOXL). That is not an edge; it is a bigger bet
  on a coin that lands tails slightly more than half the time, plus a media
  business bolted on to make the spreadsheet reach the goal.

From `CLAUDE.md`'s own walk-forward table: the best configuration ("RDT Filters")
returns **3.4% annualized** on 2 years of data — below SPY's long-run ~10% and far
below SPY's actual +19% annualized over the recent window. The filters' real
achievement is trading *less* (98% of raw signals filtered out), which reduces
losses rather than producing gains.

## Why it failed (interpretation)

1. **No edge to begin with.** RRS is a well-known retail framing; ATR-normalized
   relative strength on a liquid US equity universe is heavily arbitraged. The
   system's own PF (~1.1–1.3) and negative Kelly say the raw signal barely clears,
   or fails to clear, costs.
2. **Costs and day-trading frequency.** Day trading pays spread + commission on
   every round trip. A PF near 1.1 does not survive honest slippage.
3. **The improvement loop optimized the wrong thing.** Successive layers (VIX,
   sector, regime, intermarket, MTF) were added to *filter* a signal that has no
   underlying edge. Filtering a zero-edge signal can only approach zero return
   minus costs — it cannot manufacture positive expectancy.
4. **The return target was pursued through risk and revenue, not edge.** When the
   trading math capped out near break-even, the documented plan reached for
   leverage, 3× position risk, and a signal-selling business. The account going to
   ~$5 is consistent with escalating risk on a negative-Kelly system.
5. **Then it stopped.** No trades in 90 days means the system is not even
   executing — so there is currently nothing to optimize; there is a decision to
   make.

## What would actually count as success

- Positive realized P&L **net of honest costs**, **beating SPY buy-and-hold** over
  a comparable window, demonstrated first in backtest with defensible methodology
  and then on the paper account.
- Absent that, an honest verdict and a recommendation the human can act on.

## The uncomfortable recommendation

The convergent evidence (live account, own docs, own backtests) is that the RRS
day-trading strategy as built **does not beat, and has badly lagged, simply
holding SPY.** Before writing any more filter/parameter code, the operator and the
human should confront the real fork (see the 2026-07-24 journal entry):
- **(A) Wind down active day-trading** and treat SPY/index buy-and-hold as the
  honest benchmark the bot has to beat before it earns the right to trade.
- **(B) A genuine research reset** — hunt for a *measured* edge (event-driven,
  overnight/drift, options-vol, factor tilts) with out-of-sample IC before
  committing capital — instead of adding a 9th filter to a zero-edge momentum
  signal.
- **(C) Escalate** to the human that continued micro-optimization is not a
  responsible use of the mandate.

Do **not** respond to this post-mortem by increasing risk, adding leverage, or
building the signal-service business. Those are the failure modes, not the fixes.
