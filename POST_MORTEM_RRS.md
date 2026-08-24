# POST-MORTEM: The RRS Strategy and Why This Bot Is Where It Is

> Bootstrapped 2026-08-24 by the first operator instance. This file did not exist
> before; the scheduled mandate referenced it as if it should, so it is being
> created from a first-principles audit of the committed code and documents.
> It is intended as durable memory for future operator instances. Correct it if
> you find it wrong — but only with evidence.

---

## TL;DR

The RDT / Real Relative Strength (RRS) strategy in this repository has **no
demonstrated edge over SPY buy-and-hold net of honest costs.** Every headline
"profitability" number in the repo traces back to a single backtest that
contains three compounding artifacts — same-bar look-ahead entries, idealized
exit fills, and zero transaction costs — plus a survivor-picked universe. When
those artifacts are removed (or even just stress-tested arithmetically), the
edge disappears. The strategy's own supporting documents are internally
inconsistent and quietly concede the trading edge is near-zero (they pivot the
"path to profit" to selling a signal-service subscription).

This is not a claim that RRS *cannot* work. It is a claim that **nothing
currently in this repo shows that it does**, and the numbers that say otherwise
are not trustworthy.

---

## What the RRS strategy is

`RRS = (stock % change − SPY % change) / (ATR as % of price)`

- RRS > +2.0 → relative strength → long candidate
- RRS < −2.0 → relative weakness → short candidate

Entries also require a daily "strength"/"weakness" trend filter (EMA alignment,
recent green/red days, higher-lows / lower-highs). A large scanner stack layers
on further gates (SPY gate, 50/200 SMA, VWAP, MTF, VIX, sector, intermarket,
regime). The thesis (from r/RealDayTrading): buy the strongest stocks relative
to the market, in the market's direction, and momentum carries them.

## The headline numbers, and where they come from

`CLAUDE.md` and the "100X" docs advertise, among others:
- Walk-forward "RDT Filters": **$1,716 (6.9%)** over 2 years on $25k.
- "Best backtest: **6.8% annual** (~$1,700)", **profit factor 1.29**, **38% win
  rate**, ~**215 trades/year**, 1% risk.

Facts about these numbers:
1. They are **transcribed snapshots**, not stored results. Nothing in the repo
   persists an equity curve, trade log, or result file. `scripts/run_walkforward_v2.py`
   re-downloads data and prints to stdout; the quoted table is one past run
   pasted into a markdown file, not reproducible without re-running.
2. `scripts/run_signal_research.py`, cited in `CLAUDE.md`, **does not exist**.
3. The realized paper-trading track record is **2 outcomes over 880 scans**
   (`data/signals/signal_metrics.json`): one target hit, one stop-out. That is
   not a track record; it is noise.

## The three artifacts that manufacture the edge

Verified by reading `backtesting/engine_enhanced.py`:

1. **Same-bar look-ahead entry.** The signal is computed from the daily *close*
   (`:462`, `stock_close = ...['close'].iloc[-1]`), then the position is entered
   *at that same close* (`:494`, `entry_price=stock_close`). In reality you only
   know the close after the bar is over and the session has ended — you cannot
   trade it. A momentum signal that "buys the close of the up day" harvests the
   overnight/next-day drift for free. This alone can invent an edge.

2. **Idealized exit fills.** Stops fill *exactly* at `stop_price` (`:313`) and
   targets *exactly* at `target_price` (`:374`). Real stops gap through on the
   fast, high-momentum names this strategy selects — you get filled worse, often
   much worse, precisely on the losing trades.

3. **Zero costs.** There is no commission, no spread, and no slippage anywhere
   in `backtesting/`. For a strategy taking ~215 trades/year on a thin edge,
   costs are not a rounding error — they are the whole question.

Plus a fourth, universe-level problem:

4. **Survivorship / selection bias.** The walk-forward universe is a fixed list
   of today's mega-cap winners (AAPL, NVDA, META, …) applied retroactively to
   2024–2025. Of course a basket of the last two years' biggest winners looks
   good in the last two years.

## The arithmetic (no data required)

`scripts/edge_sensitivity.py` stress-tests the strategy's OWN reported numbers:

- The docs' stated avg win $70 / avg loss $45 / 38% win rate imply a **profit
  factor of 0.95 and a negative per-trade expectancy (−$1.30)** — *not* the
  claimed PF 1.29. The supporting numbers are mutually inconsistent.
- Granting the most favorable self-description (PF 1.29, $1,700/yr gross), the
  edge is **$7.91 per trade**. Break-even round-trip cost is therefore $7.91.
- A defensible round-trip cost for a ~$12.5k-notional momentum trade (IBKR
  commission + ~2bp spread each side + ~10bp gap slippage on stop-outs) is
  **~$14.75**, which turns the year into **~−$1,471 (−5.9%)**.
- **Even at literally zero cost**, 6.8%/yr trails SPY buy-and-hold (~10–25%/yr
  over the same window) by roughly **$800–$4,550 per year**.

The strategy's own author reached a compatible conclusion: `ACTIONABLE_100X_STRATEGY.md`
computes `kelly = -0.02  # NEGATIVE!` and then routes 40–60% of its "path to
$25k" through a **signal-subscription SaaS business**, i.e. not trading alpha.

## Why the bot is in its current state

The codebase is large, competent, and heavily engineered — dozens of agents,
gates, ML features, dashboards, a full options module. But the engineering
investment went into **breadth and infrastructure**, not into **honestly
measuring whether the core signal makes money**. The one thing that would settle
the question — a costed, look-ahead-free, out-of-sample backtest benchmarked
against SPY — was never the thing being built. In its absence, optimistic
backtest snapshots became "the numbers," and everything downstream assumed the
edge was real.

## What would change this conclusion

Concrete, falsifiable bars to clear before believing RRS has an edge:
1. A backtest with **next-bar-open (or later) entries**, modeled commission +
   spread + slippage, and **gap-through** stop fills. (`scripts/honest_backtest.py`
   implements exactly this; it just needs a reachable data source.)
2. A **point-in-time universe** (or at minimum a broad, non-cherry-picked one)
   to kill survivorship bias.
3. **Out-of-sample** results (train/validate/test split or true walk-forward)
   that still beat SPY *net* on the test window.
4. A **paper-trading track record** with realized P&L over enough trades
   (hundreds, not two) to be more than noise.

Until at least #1–#3 are cleared, the honest status is: **no edge demonstrated;
do not risk capital; SPY buy-and-hold is the benchmark to beat and currently
wins.**

## Operator's standing recommendation

- Treat all existing "profitability" numbers as unproven until reproduced under
  honest conditions.
- The highest-value work is **measurement, not more features**: make the
  backtest honest, then let it tell the truth about RRS.
- If honest testing keeps saying "no edge," that is a valid, reportable outcome —
  escalate for a strategy rethink or wind-down rather than adding complexity.
