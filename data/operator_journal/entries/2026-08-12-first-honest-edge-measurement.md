# 2026-08-12 — First honest edge measurement (and journal bootstrap)

**Operator instance:** first run. **Branch:** `operator/2026-08-12`.
**One-line verdict:** Built the operator infrastructure from scratch and ran the
first cost-aware, SPY-benchmarked measurement of the bot's own signals. **Result:
no demonstrable edge. Under realistic entry fills the signals slightly
*underperform* SPY, and the "sample" is really just two trading days.** This
confirms the post-mortem's prior. Recommend: do NOT add features; either find a
genuinely different, testable edge or wind the trading down.

---

## State of play (Step 1 assessment)

- The operator journal, `MANDATE.md`, and `POST_MORTEM_RRS.md` **did not exist**.
  I am the first instance. I bootstrapped all three this session.
- The bot has been **dormant since 2026-03-05** (~5 months). It last scanned then.
- `signal_metrics.json`: 880 scans, 120 signals emitted, **2 outcomes ever
  recorded** (1 win, 1 loss). The scan→signal→trade→result→learn loop was never
  closed. There is no live P&L track record at all.
- `signal_history.json`: 1,986 raw signal rows, all `RRS_Momentum`, **zero** with
  any outcome/exit/pnl field.
- The repo's own docs disagree on the strategy's win rate (38% vs 49.5%) and none
  compare to SPY buy-and-hold. `ACTIONABLE_100X_STRATEGY.md` concedes the strategy
  "mathematically caps returns around 7%" then proposes filling the gap to a 100%
  target by **selling a signal service** — goal displacement, not an edge.

## The one bet this session (Step 2)

Hypothesis to test: *Do these RRS signals, acted on, beat SPY buy-and-hold net of
realistic costs?* This is the prior question the whole project skipped. I built the
measurement instead of adding anything.

## What I did (Step 3)

1. Bootstrapped `data/operator_journal/` + `MANDATE.md` (constitution/protocol) and
   `POST_MORTEM_RRS.md` (honest reconstructed history).
2. Wrote `scripts/measure_signal_edge.py` — stdlib, dependency-free. It dedupes the
   re-logged signals, evaluates each distinct signal's planned bracket (target vs
   stop) against **real daily OHLC bars pulled from IBKR** (yfinance is blocked in
   this env), nets out slippage, and compares every trade to SPY over the identical
   window. Two fill assumptions (planned entry vs realistic entry-day open), a
   naive t-stat, and a capital-capped portfolio sim.
3. Pulled 1yr daily bars for SPY + the top 25 signal symbols (85% of signals) via
   the IBKR MCP tools (data saved under `research/2026-08-12/hist/` for reproducibility).

## Evidence (Step 4) — the numbers I actually measured

Window: signals 2026-02-03/04 → outcomes over following weeks. **SPY buy&hold over
the full period 2026-02-03 → 2026-08-12: +11.0%.** 550 distinct signals evaluated
across 25 symbols (505 long / 45 short). Hold cap 20d (avg hold 3.4d). Slippage 5bps/side.

| Metric (net of costs) | Fill @ **planned entry** (optimistic) | Fill @ **entry-day open** (realistic) |
|---|---|---|
| Win rate (target before stop) | 36.4% | 36.8% |
| Avg return / trade | +0.085% | **−1.467%** |
| Avg SPY over same window | −0.957% | −0.957% |
| **Avg excess vs SPY** | **+1.042%** (t=+6.4) | **−0.510%** (t=−0.8) |
| Long-only avg excess | +1.177% (t=+7.1) | −0.539% (t=−0.8) |

Capital-capped portfolio (realistic fills, 1% risk/trade): **−3.8% / −3.7% / −5.7%**
for max 3 / 5 / 8 concurrent positions, vs **SPY +11.0%**. Only 6–15 trades are even
takeable because the signals all fire at once.

### Why the apparent edge is not real
1. **The whole "edge" was a fill artifact.** Filling at the signal's *planned*
   entry_price (a level often set at the prior close, below where the stock actually
   opened) manufactured a +1% excess with a gaudy t=6.4. Filling at the **realistic
   entry-day open** erases it: **−0.5% excess, t=−0.8 (not significant)**. Overnight
   momentum signals gap; you don't get the planned price.
2. **The sample is two days, not 674 trades.** Distinct signals by date: **Feb 3:
   412, Feb 4: 247**, then a handful. Effective independent sample ≈ 2 market days.
   Every "trade" is a correlated bet on one Feb-3/4 event. No statistical power; no
   regime diversity.
3. **Profile is a lottery ticket.** 36% win rate, median trade negative; any positive
   mean rides on a few fat-tail winners from that one event. Fragile.
4. Even the optimistic-fill illustrative compounding ("+24.7%") is fantasy — it
   assumes 550 sequential non-overlapping trades that in reality all fire on 2 days.
   (The realistic capped portfolio is the honest version, and it loses.)

> Caveat on the harness: under open-fill, the printed "Avg R-multiple +2.955" is a
> junk artifact (tiny risk denominators when open ≈ stop); ignore R there and read
> the return% / excess, which are stable. Noted for the next instance.

## Verdict (Step 5)

**Does it beat SPY? No — it underperforms once fills are realistic, and the data is
too thin to claim anything anyway.** The post-mortem's prior holds and is
strengthened with actual numbers: **there is no validated, SPY-beating edge in this
system.** A truthful null result, which the mandate treats as a successful session.

## Next (single most valuable thing for the next instance)

The current evidence base is 2 days of a dormant bot. Before any strategy verdict is
final, get **more out-of-sample signals across regimes**. Concretely, in priority order:
1. **Generate a proper out-of-sample signal set.** The scanner logic exists; run it
   (offline, against historical daily/intraday bars via the IBKR MCP or a data dump)
   over many months of 2025–2026 so we have hundreds of signals on *different* days,
   not 2. Feed them through `measure_signal_edge.py` (extend it to batch-load a
   generated signal file). Only then is the edge question answerable.
2. **Model realistic fills properly** (next-open or entry_price-only-if-touched, plus
   gap-through-stop handling). The plan-vs-open gap this session shows fills are the
   whole ballgame for an overnight-signal strategy.
3. If, with a real multi-month sample and realistic fills, excess-vs-SPY stays ≤ 0
   and not significant → **recommend winding down live trading and stopping feature
   work.** If it turns clearly positive → build the capital-aware portfolio layer and
   re-verify before trusting it.

Do not add ML, options, dashboards, or "signal service" work until step 1 answers the
edge question on real data. That is the project's core unanswered question.

## Open risks / flags

- **No changes to `risk/`** this session. No trading code changed. Only added: journal
  infra, post-mortem, and a read-only measurement script. Nothing here can place an order.
- **PAPER-only respected.** `AUTO_TRADE` untouched; no broker creds touched.
- IBKR MCP historical bars are **delayed/EOD daily** — fine for this daily-bar analysis;
  intraday fill realism will need intraday bars.
- Coverage is 25/48 symbols (85% of signals); the 23 tail symbols are unmeasured but
  can't change a 2-day-sample conclusion.
- Branch naming: the harness scaffolding referenced `claude/adoring-feynman-nrlxek`;
  the operator mandate specifies `operator/YYYY-MM-DD`. I followed the mandate and
  used `operator/2026-08-12`. Flagging for the human reviewer.

## Reproducibility

- Harness: `scripts/measure_signal_edge.py` (run from repo root).
- Data + full output: `data/operator_journal/research/2026-08-12/` (26 price files,
  `edge_measurement_output.txt`, `edge_summary.json`).
