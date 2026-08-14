# POST-MORTEM: The RRS Trading System — Why It Is Where It Is

> Bootstrapped 2026-08-14 by the first autonomous operator instance. This file is the
> honest history of how the bot reached its current state, written to save future
> instances from re-learning it. It will be extended by later operators; append, don't
> rewrite, unless a fact is proven wrong.

## The one-paragraph summary

This is a relative-strength momentum day-trading bot built on the r/RealDayTrading
"RRS" concept: buy stocks outperforming SPY (RRS > 2), short those underperforming,
with ATR-based stops and a fixed 2:1 reward:risk. Over ~two years of development it
accreted an enormous amount of machinery — multi-timeframe gates, VIX/sector/regime/
intermarket/news overlays, an 87-feature ML stack, an options module, a multi-broker
layer, and even a SaaS product skin — **without ever establishing, with reproducible
out-of-sample evidence, that the core signal has a cost-surviving edge.** As of this
writing there is no P&L ledger in the repo and only 2 tracked trade outcomes. The
headline backtest the docs cite (~3.4% annualized) would, if accurate, still lose to
simply holding SPY (~10%/yr; ~+15% over Feb–Aug 2026).

## How it got here (the pattern)

The git and CLAUDE.md history shows a consistent loop:

1. Start with a plausible idea (RRS momentum, per r/RealDayTrading).
2. Backtest is disappointing or ambiguous.
3. **Instead of concluding "weak/no edge," add another layer** — a filter, an overlay,
   a regime model, an ML feature set, a new asset class.
4. Re-backtest (in-sample, one regime, via yfinance), get a slightly better number,
   declare victory in the docs.
5. Repeat. Complexity compounds; validated edge does not.

Symptoms visible in the tree today:
- **No outcome tracking that matters.** `signal_history.json` (1986 signals, Feb–Mar
  2026) records signals but *zero* realized outcomes. `signal_metrics.json`: 2 outcomes.
- **Backtests can't be reproduced in the agent environment.** `run_walkforward*.py`
  and `train_from_history.py` depend on yfinance, which is blocked by the egress proxy.
  Every headline number in CLAUDE.md is therefore currently *unverified*.
- **Complexity that no edge measurement justifies.** 87 ML features, 5+ overlay
  filters, options, SaaS — none tied to a demonstrated, cost-net edge on the base signal.
- **Direction instability.** One metrics snapshot showed 119 shorts / 1 long; the
  larger history is 1687 long / 299 short. The signal's directional bias is regime-
  driven and unvalidated.

## What "profitable" has to mean here

Per the operator mandate: **positive P&L net of honest costs (commission + slippage +
spread), beating SPY buy-and-hold.** Win rate alone is meaningless; a 2:1 R:R needs
only ~33% wins to break even *gross*, but costs raise that bar, and beating SPY raises
it further.

## Empirical results — independent first-touch backtest (2026-08-14)

The first operator ran an independent evaluation of all 1986 historical signals
against **real IBKR daily price data** (yfinance being blocked). Methodology: enter at
each signal's stated entry_price, walk subsequent daily bars, first-touch of stop vs
target (daily high/low), same-bar ambiguity → stop (conservative), timeout →
mark-to-market. Costs modeled at ~$0.05/share round trip. Full script and outputs are
referenced in the 2026-08-14 journal entry.

**Dataset:** 1,838 of the 1,986 raw signals (45 of 48 symbols resolved to correct IBKR
contracts; DD excluded for a wrong-contract mismatch, WEC/XEL not fetched). Signals were
generated Feb 3 – Mar 5 2026; forward window is Feb–Mar 2026, a **rising market**
(SPY +12.5% Feb→Aug; up over the evaluated windows too). Reproduce with the script
referenced in the journal entry.

**Headline (MAX_HOLD=10 trading days, same-bar ambiguity → stop, costs ~$0.05/sh RT):**

| Slice | n | Target-hit (win) rate | Net expectancy | Profit factor |
|-------|---|----------------------|----------------|---------------|
| All signals | 1,838 | 60.3% | **+0.76 R** | 3.05 |
| **Longs only** | 1,562 | **69.0%** | **+1.03 R** | 4.49 |
| **Shorts only** | 276 | **2.3%** | **−0.74 R** | 0.075 |

**Directional alpha test (hold 10d, no stop/target, vs SPY over the same dates):**
- Longs: stock +1.97% vs SPY +0.57% → **+1.40% alpha**, 64% beat SPY.
- Shorts: shorted names *rose* +2.39% while SPY was flat → **−2.35% alpha**, only 26% beat SPY.

**What this does and does NOT prove:**
- ✅ The SHORT side is a clear money-loser in a non-bearish tape (robust: 2.3% hit rate,
  −0.74R, negative alpha, spread broadly across symbols). This *empirically validates the
  design of the SPY "Market First" gate* — do not take counter-trend shorts.
- ⚠️ The LONG edge is real *in this sample* (broad: median 87% win rate across 28 symbols)
  **but is not beta-adjusted and coincides with a rising market.** The +1.4% "alpha"
  controls for SPY *direction* but not for *beta magnitude* — high-beta momentum names
  outrun SPY in an up-tape by construction. It cannot be distinguished from leveraged long
  beta without a bear-market / multi-regime, beta-adjusted test. **No such test exists yet.**
- ❌ This is NOT evidence the *deployed* (post-gate) system is profitable, and NOT evidence
  it beats SPY buy-and-hold risk-adjusted. `signal_history.json` is a *raw pre-gate* log
  (only 4/1,986 rows carry post-gate metadata), so this measures raw signal edge, not
  realized performance. The bot still has no P&L ledger.

**Action taken (2026-08-14):** Closed the SPY gate's fail-open holes — see journal entry
`2026-08-14-first-honest-edge-measurement.md`.

## Guidance for future instances

- **Do not add a new overlay/feature/model to "fix" a weak edge.** That is the exact
  failure loop above. First prove the base signal has edge, out-of-sample, net of cost.
- **Measure before you build.** The IBKR MCP tools give you real data. Use them.
- **If the edge isn't there, say so** and follow the mandate's escalation/wind-down
  path. That is the valuable outcome, not another green in-sample number.
