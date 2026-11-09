# POST-MORTEM: The RRS Strategy and Why This Bot Is Where It Is

**Author:** Autonomous operator (first session, 2026-11-09)
**Status:** Living document. Append, do not rewrite history.

> This file was created on the first operator run. The scheduled operator prompt
> referenced it as pre-existing history, but it did not exist in the repo (nor did
> `data/operator_journal/` or `MANDATE.md`). This is a reconstruction from the
> evidence actually present in the codebase on 2026-11-09, plus one independent
> backtest I ran this session. Where I am inferring rather than citing a file, I
> say so.

---

## 1. What the bot is supposed to do

Implement the r/RealDayTrading (RDT) methodology: scan a watchlist for **Real
Relative Strength** (`RRS = (stock %chg − SPY %chg) / ATR%`), go long the
strongest / short the weakest names, gated by a "market first" filter stack
(SPY trend gate → 50/200 SMA gate → VWAP gate → multi-timeframe gate) plus VIX,
sector, regime, and intermarket overlays. Paper trading on a $25K account.

The **mandate's success criterion** is explicit and narrow: *actual positive P&L,
net of honest costs, that beats SPY buy-and-hold.* Not "good metrics." Not
"follows the methodology faithfully." Beat the index, net of costs.

## 2. The core problem: the edge is real but tiny, and smaller than SPY

Every artifact in this repo that reports performance agrees on the shape of the
result, and my own backtest reproduces it:

| Source | Best-config return | Notes |
|---|---|---|
| `CLAUDE.md` (documented 2yr walk-forward) | $1,716 / 6.9% total (~3.4% annualized) | "RDT filters" config C |
| `WEALTH_STRATEGY_100X.md` | 6.84% annual (sic) / $1,711 | same run, mislabeled as annual |
| `WEALTH_OPTIMIZATION.md` | 2.85% annual baseline | pre-optimization |
| **My re-run, 2026-11-09, config C, gross** | **9.81% total over 2yr ≈ 4.8% annualized** | Sharpe ~0.25 |

Whatever the exact number, it sits in the **3–10% total-over-two-years** range.
Over the *same* two-year window (2024-07 → 2026-07), **SPY buy-and-hold returned
+39.3% total / ~18% annualized** (I computed this from yfinance adjusted closes
this session). The bot captured roughly **one quarter of SPY's return** while
making ~280 trades.

**The one thing the strategy does well:** drawdown. Bot max drawdown was ~2.8%
vs SPY's ~18.8% over the window. It is a low-volatility product, not a
SPY-beating one.

### Why it caps so low (the math, from `ACTIONABLE_100X_STRATEGY.md` and confirmed)
- Win rate ~49–51%, profit factor ~1.3, ~140 trades/year at ~1% risk.
- A strategy with those stats at 1% risk mathematically caps near single-digit
  annual returns. It is **signal-limited, not capital-limited** — there simply
  aren't many high-quality RRS setups, and 96–98% of raw signals are filtered out.
- You cannot fix a thin edge by trading bigger; you just scale the variance.

## 3. The response so far was to drift away from the mandate

Instead of concluding "this does not beat the index," the repo accumulated a set
of documents proposing to hit a **100% annual return ($25K→$50K)** target:
`WEALTH_STRATEGY_100X.md`, `QUICK_START_100X.md`, `ACTIONABLE_100X_STRATEGY.md`,
`WEALTH_OPTIMIZATION.md`, `DEPLOYMENT_SUMMARY.md`. Their proposed path:

1. **Crank risk.** `DEPLOYMENT_SUMMARY.md` (dated 2025-12-29) set an "AGGRESSIVE"
   profile: 3% risk/trade (from 1%), 6% daily loss (from 2–3%), 20% max position
   (from 10%), RRS threshold lowered 2.0→1.75 to force more signals. This raises
   variance and ruin risk without improving the edge — the exact failure the math
   above warns against.
2. **Add leverage, options, crypto, futures, extended hours** to manufacture
   returns the underlying signal doesn't produce.
3. **Sell the signals.** The 100X docs explicitly plan for **40–60% of the
   "profit" target to come from a signal-service / API subscription business**,
   not from trading. That is abandoning the mandate (profitable *trading*) and
   relabeling "sell shovels" as success.

This is the central lesson of this post-mortem: **when a strategy's honest edge
is below the benchmark, the temptation is to add leverage and side-hustles rather
than accept the finding. The mandate exists to resist exactly that.**

## 4. The live scanner evidence is also bad

From `data/signals/signal_metrics.json` (last scan 2026-03-05, nothing since):
- 880 scans → 120 signals, of which **119 short, 1 long**. A ~99% short bias is
  pathological for a momentum strategy and strongly suggests a directional bug or
  a badly skewed gate, not a real market read. (Unverified: I did not root-cause
  it this session.)
- **Only 2 signals ever had tracked outcomes** (1 target, 1 stop). There is
  effectively **no live paper-trading P&L record**. All "results" are historical
  simulation, not realized paper fills. We are flying on backtests.

## 5. The backtests were not even net of costs

As of this session, `backtesting/engine_enhanced.py` and `engine.py` modelled
**zero commissions and zero slippage**. Every headline number in this repo is
**gross**. For a ~280-trade strategy that is a material omission. (I added an
opt-in cost model this session — see the 2026-11-09 journal entry.)

## 6. Honest verdict

On the mandate's own terms, **the RRS strategy as built does not work**: it
underperforms SPY buy-and-hold by a wide margin, gross — and costs only widen the
gap. Its single merit is low drawdown.

This does not mean "delete everything." It means the strategic question is no
longer "which parameter tweak?" but one of three honest options, which belong to
the human owner, not the operator:

- **(A) Reframe the product.** Accept the bot as a low-volatility, low-return
  diversifier and stop pretending it targets 100%/yr. Benchmark it against
  low-vol / market-neutral peers, not SPY. (But then it still fails *this*
  mandate, which is to beat SPY.)
- **(B) Hunt for a genuinely different edge.** Use the `research/` factor-testing
  framework to find factors with real out-of-sample IC, net of costs, before
  committing capital. Treat everything as unproven until it survives walk-forward
  *with costs*.
- **(C) Wind down the trading ambition.** If months of evidence keep saying no
  edge beats the index net of costs, the honest move is to stop and say so.

The operator's standing recommendation until evidence changes: **do not increase
risk, do not add leverage/options/crypto to chase returns, do not treat a
signal-selling business as fulfilling the trading mandate.** Revert the aggressive
risk profile toward the conservative defaults. Keep everything in paper.

---

## Appendix: how to reproduce the key numbers

```bash
# Gross (historical, default):
python3 scripts/run_walkforward_v2.py

# Net of honest costs (IBKR-style: $0.005/share, $1 min, 5bps slippage/fill):
BACKTEST_COSTS=1 python3 scripts/run_walkforward_v2.py

# SPY buy-and-hold over the same window:
python3 -c "import yfinance as yf; c=yf.download('SPY',start='2024-07-26',end='2026-07-29',auto_adjust=True)['Close']['SPY']; print((c.iloc[-1]/c.iloc[0]-1)*100, '% total')"
```
