# POST-MORTEM: The RRS Strategy — Why This Bot Is Where It Is

> This file explains, for each future stateless operator instance, how the RDT
> Trading System arrived at its current state and what the accumulated evidence
> says about its core premise. It is maintained alongside the operator journal.
> Read it after `MANDATE.md` and `LATEST.md`.

**First authored:** 2026-10-13 (bootstrap operator session).
**Status of the core trading premise:** **NO DEMONSTRATED EDGE NET OF HONEST COSTS.**

---

## 1. What this system is

An autonomous trading bot built around the r/RealDayTrading (RDT) methodology:
Real Relative Strength (RRS) = `(stock %Δ − SPY %Δ) / ATR`, scanning a watchlist of
large-cap US equities for RRS > 2.0 (long) / < −2.0 (short), gated by a stack of
"market-first" filters (SPY trend gate, 50/200 SMA, VWAP, multi-timeframe alignment,
VIX regime, sector RS, intermarket analysis), with an ML advisory layer, options
overlay, multi-broker (IBKR/Schwab/paper) execution, and a Flask dashboard. Built
Dec 2025 → Mar 2026. It has since accreted a SaaS layer (landing/pricing/login,
subscription tiers).

It is large and well-engineered. **Engineering quality is not the problem. The
premise is.**

## 2. The core problem, stated plainly

The strategy has, at best, a **marginal gross edge that does not survive honest
transaction costs, and that underperforms simply holding SPY.**

This is not a new discovery — the repo's own planning documents already admitted it:

- `ACTIONABLE_100X_STRATEGY.md` and `WEALTH_STRATEGY_100X.md` both state the best
  backtest is **~6.8% annual (~$1,711 profit on $25K)**, with **profit factor ~1.29,
  win rate ~38%**, and — critically — compute the **Kelly criterion at ≈ −0.02
  (NEGATIVE)**. A negative Kelly means there is no fraction of capital you can bet to
  grow the account; the edge is at or below zero.
- Faced with that, those same documents do **not** conclude "the strategy has no
  edge." Instead they pivot: the "path to 100% annual return" becomes **40–60% from
  selling a $49–$499/month signal-service subscription**, plus leverage and options.
  That is monetizing an audience, not trading profitably. **This pivot is the central
  intellectual failure the operator must not repeat.** (See MANDATE §4.)

## 3. Independent verification (this session)

The documented backtests are **gross and frictionless.** Confirmed by reading
`backtesting/engine_enhanced.py`: trade P&L is computed as raw
`(exit_price − entry_price) × shares` with perfect fills at the exact computed
stop/target prices. **There is no commission, slippage, bid/ask spread, or borrow
cost anywhere in the backtest engines** (`grep -niE "commission|slippage|borrow|fee|
spread" backtesting/*.py` → zero hits). Every headline return in this repo is a
best-case, zero-friction number.

To quantify the gap, this session added `scripts/run_cost_sensitivity.py`, which runs
the documented "best" configuration (Config C, full RDT filter stack) over a
continuous out-of-window period and applies realistic round-trip costs.

**Result — RDT Config C, 30 mega-caps, 2025-02-05 → 2026-10-12 (423 trading days),
$25,000 start:**

| Slippage/side | Commission | Slippage | Net P&L | Net Return | Win Rate | Profit Factor |
|---------------|-----------|----------|---------|-----------|----------|---------------|
| 0 bps (gross) | $486      | $0       | **$2,577**† | **10.31%** | 53.1% | 1.30 |
| 5 bps         | $486      | $505     | $1,586  | 6.34%     | 52.3%    | 1.22 |
| 10 bps        | $486      | $1,010   | $1,081  | 4.32%     | 52.3%    | 1.15 |
| 20 bps        | $486      | $2,019   | $71     | 0.28%     | 49.0%    | 1.01 |

† Gross includes the $486 commission line but 0 slippage; the pure frictionless P&L
is $2,577 + commission handling as reported by the script. (Reproduce:
`python3 scripts/run_cost_sensitivity.py`.)

- **Gross edge per trade: $10.60** on **~$2,071 average notional** (≈5 bps of
  notional per trade).
- **Break-even slippage: ~20.7 bps per side.** Above that, the strategy loses money
  net of costs. For a **stop-based swing strategy** — where exits are adverse-side
  stop fills that slip, targets require actual trade-through, and the daily-bar
  backtest assumes fills at the exact close/stop/target — 10–20 bps/side all-in is a
  realistic, not pessimistic, estimate.

**The benchmark:** SPY buy-and-hold over the **identical window returned $7,827
(31.31%)**.

### What this means
- **Even frictionless**, the active strategy (10.31%) earns **less than a third** of
  passively holding SPY (31.31%). The cost argument is not even needed to show
  failure — it only deepens it.
- **Net of realistic 10 bps/side costs**, the strategy returns **$1,081 (4.32%)** —
  about **14% of buy-and-hold** — while taking **243 round-trip trades** of
  concentrated single-name directional risk, requiring active infrastructure, data
  feeds, and uptime.
- At plausibly higher real-world costs (20 bps/side), expectancy is **indistinguishable
  from zero.**
- This independently reproduces the repo's own "~negative-to-marginal edge" finding,
  on a **different, more recent 20-month window** than the documented tests — so it is
  not a one-window artifact, and it is partially out-of-sample relative to the docs.

## 4. Honest caveats (stated so the next instance doesn't over- or under-claim)

- The test window (Feb 2025 → Oct 2026) was a **strong bull market** (SPY +31%),
  which flatters buy-and-hold. But the RDT thesis claims day-trading earns its keep
  through relative-strength selection and downside/short protection — none of which
  let it keep pace here. Failing to beat a buy-and-hold in an up market, *and* having
  near-zero expectancy after costs, is the worst of both worlds.
- Daily-bar backtests **cannot** simulate the intraday VWAP gate or first-hour
  filter; the live intraday system may behave differently. But the burden of proof is
  on demonstrating edge, and the daily-bar backtest is the very tool this repo uses to
  claim its 6.9%. A filter that can only be validated by trading it live is not yet
  evidence.
- This is one configuration (the documented best) on 30 mega-caps. Other universes or
  parameters might differ — but note that more parameters + a tiny, re-used sample is
  the recipe for overfitting, not for finding a durable edge.
- Short-borrow costs were **not** modeled (mega-caps, short holds → negligible). A
  broader/retail-hard-to-borrow short book would be worse, not better.

## 5. The verdict for the mission

Per `MANDATE.md`, the mission is positive P&L **net of honest costs, beating SPY
buy-and-hold** — or an honest verdict that no strategy works. On the current evidence:

> **The RRS day-trading premise, as implemented, has no demonstrated edge net of
> honest costs and badly underperforms SPY buy-and-hold. The repo's own Kelly
> analysis (−0.02) and this session's cost-sensitivity test agree. The historical
> "path to profitability" in this repo quietly substituted *selling signals* for
> *trading profitably*.**

This does not yet prove that *no* variant could ever work. It does mean the honest
next steps are **measurement- and falsification-driven, not feature-driven**:

1. **Do not add features, tune parameters, or chase the 100× narrative** until a
   variant clears the bar: positive expectancy **net of ≥10 bps/side costs**, **out
   of sample**, **beating SPY buy-and-hold**, reproducible from committed code.
2. Any serious attempt to find an edge should test a **specific, falsifiable
   hypothesis** on a **larger universe and multiple regimes** (including a down
   market), with costs modeled from the first run.
3. If successive honest tests keep returning "no edge," the correct recommendation is
   **wind down the active-trading premise** (keep paper-only for research) and
   escalate to the human — not to pivot into selling the strategy to others.

## 6. Reproduce the evidence

```bash
cd /home/user/rdt-trading-system
python3 -m pip install numpy pandas yfinance loguru pydantic pydantic-settings pyarrow
python3 scripts/run_cost_sensitivity.py         # cost-aware, vs SPY buy-and-hold
python3 scripts/run_walkforward_v2.py           # the repo's documented 3-way backtest
grep -niE "commission|slippage|borrow|fee|spread" backtesting/*.py   # → confirms: none
```

_Numbers above are reproducible from committed code and yfinance data as of
2026-10-13. Later runs will extend the window; re-run before relying on them._
