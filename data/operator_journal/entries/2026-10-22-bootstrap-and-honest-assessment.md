# 2026-10-22 — Bootstrap + first honest profitability assessment

**Operator run #1 (effectively).** Branch: `operator/2026-10-22`.
Model: Opus 4.8 (1M). Fresh checkout, no live infra.

---

## 0. Situation on arrival

The scheduling prompt told me to read, in order: `data/operator_journal/MANDATE.md`,
`LATEST.md`, `POST_MORTEM_RRS.md`, `CLAUDE.md`, and the 3 latest journal entries.

**None of the operator-journal files or `POST_MORTEM_RRS.md` exist** — not in the
working tree and not anywhere in git history (checked `git log --all`). The journal
infrastructure the mission depends on had never been created. So this run is a
bootstrap: I created the journal scaffolding and wrote the mandate (`MANDATE.md`,
flagged as a draft for the owner to ratify — it encodes the scheduling prompt's
intent, nothing more), then did the one thing that matters most on run #1:
**an honest, evidence-based answer to "is this bot profitable / can it beat SPY
buy-and-hold?"**

Branch note: the harness system prompt names a default branch
(`claude/adoring-feynman-kcjy3k`); the scheduling prompt explicitly instructs
`operator/YYYY-MM-DD` and that the human merges. I followed the scheduling prompt's
explicit, intentional instruction and used `operator/2026-10-22`.

## 1. Assessment — what the repo itself claims

- `CLAUDE.md` headline backtest ("RDT Filters", Config C, 2yr walk-forward,
  $25k): **+6.9% total, +3.4% annualized**, win rate 49.5%, profit factor 1.24.
- `ACTIONABLE_100X_STRATEGY.md` (the repo's own strategy doc) states the core RRS
  strategy has **profit factor 1.29, 38% win rate, and a slightly NEGATIVE Kelly
  criterion** — i.e. by its own math the edge is marginal-to-nonexistent, and it
  pivots the "$25k→$50k" plan away from trading toward **selling signal
  subscriptions**. When a trading plan's path to its return target is "sell
  subscriptions," that is a tell that the trading edge isn't there.
- The committed walk-forward harness (`scripts/run_walkforward_v2.py`) has two
  honesty gaps for this mission: **(a) zero transaction costs** (entry just
  subtracts `shares*price`; no commission/slippage/spread anywhere), and
  **(b) no SPY buy-and-hold benchmark** — it only compares filter variants to each
  other, never to the thing the mission says to beat.

## 2. Decision

Independently verify the edge and, crucially, **compute the missing benchmark and
the missing costs** over the identical window. Hypothesis to falsify: *"the RRS
strategy beats SPY buy-and-hold net of honest costs."*

## 3. Execution — independent backtest

Wrote a self-contained backtest (`evidence/2026-10-22/honest_bt.py`, output in
`results.txt`) that is deliberately faithful to the repo's own logic
(`shared/indicators/rrs.py` RRS formula + relaxed strength/weakness; the scanner's
SPY 50/200-EMA gate and stock 50-SMA gate; Config-C 1.5×ATR stop / 2.0×ATR target /
1.5% risk / max 8 positions / 12-day time stop), but adds what the mission requires:

- **No lookahead:** signal at `close[T]`, enter at `open[T+1]`, exits via intrabar
  high/low with stop priority.
- **Transaction-cost sensitivity:** gross, realistic ($0.005/sh + 2bps), higher
  ($0.01/sh + 5bps).
- **SPY buy-and-hold over the identical window** (dividend-adjusted).
- Same 30 large-cap universe + SPY as the committed harness.
- No leverage; single-position cap 25% of equity.

(Found and fixed a NumPy bug mid-run — `np.bool_ + np.bool_` is logical OR, not
integer addition — which had silently produced 0 trades. Documented in the mandate.)

## 4. Findings (window Feb 2024 → Oct 2025, 1.72 yr, $25k)

| Scenario | Trades | Win% | PF | Total | Annualized |
|---|---|---|---|---|---|
| RRS gross (no costs) | 97 | 49.5% | 1.24 | +9.7% | +5.5% |
| RRS realistic costs | 99 | 49.5% | 1.22 | +8.7% | +5.0% |
| RRS higher costs | 98 | 50.0% | 1.18 | +7.1% | +4.1% |
| **SPY buy-and-hold** | — | — | — | **+40.2%** | **+21.7%** |

**My independent PF (1.24) and win rate (49.5%) match `CLAUDE.md`'s committed
numbers exactly** — strong cross-validation that both engines capture the same
edge. My total return is a touch higher (9.7% vs 6.9%) because I omit the
threshold-raising VIX/regime/MTF layers (so I trade slightly more in a bull tape).

### Verdict
**The strategy does NOT beat SPY buy-and-hold. It is not close.** Over this window
it captured roughly **one-quarter** of buy-and-hold's return (+5% vs +21.7%
annualized) while taking on single-stock selection risk, intraday leverage, and
drawdown risk that buy-and-hold does not carry. The edge is *real but marginal*
(PF ~1.2) and does survive this low-frequency daily-proxy's modest costs — but it
loses decisively on the only comparison the mission cares about.

### Three caveats I will not paper over
1. **Regime dependence.** Feb 2024–Oct 2025 was a strong bull; buy-and-hold is
   flattered. A long-short relative-strength strategy earns its keep in
   bear/sideways tapes (e.g. 2022). A fair verdict needs a down-market window —
   that is the top task for run #2.
2. **The backtest is a daily-bar proxy of an intraday strategy.** The live bot
   trades **5-minute RRS**; **nothing in the repo backtests the actual intraday
   strategy.** Intraday means more round-trips → costs bite far harder than the
   ~100-trade daily proxy shows. The real edge is **unverified**.
3. My dollar accounting is a clean-room reimplementation; the PF/WR agreement with
   the repo is the load-bearing result, not the exact % (which depends on sizing
   and the bull window).

## 5. What I changed this run

- Created `data/operator_journal/` (MANDATE.md draft, entries/, evidence/, LATEST.md).
- Added evidence: `evidence/2026-10-22/honest_bt.py` + `results.txt`.
- **No production code touched. `risk/` untouched. No config changed.** This run is
  assessment + infrastructure only.

## 6. Recommendation to the owner

- **Do not treat this bot as a wealth engine on current evidence.** Over the tested
  period, holding SPY beat it ~4×, net of costs, with less risk.
- Before any further tuning, the mission needs two things that don't exist yet:
  **(a)** an honest **intraday** backtest with realistic costs (the real strategy),
  and **(b)** evaluation across a **bear/sideways** window where a market-neutral
  RRS approach could actually justify itself. If neither produces a net-of-cost
  edge over buy-and-hold, the honest move is **wind-down of the trading objective**
  (keep the scanner/dashboard if useful) per MANDATE §6.

## 7. Next run (run #2) — concrete

1. Build a cost-aware **intraday** backtest, or prove (with data) that one can't be
   built from available data — the daily proxy is not good enough to bet on.
2. Re-run the daily backtest over a **bear window** (e.g. 2022, or Jan–Oct 2022)
   and a **sideways** window; report RRS vs buy-and-hold in each. This is the
   fairest test of whether the long/short RRS idea has standalone merit.
3. Compute buy-and-hold-relative (alpha) and max drawdown, not just total return.
4. Owner: please **ratify or edit `MANDATE.md`**, and confirm the branch convention.

---
*Evidence: `data/operator_journal/evidence/2026-10-22/` (honest_bt.py, results.txt).
Run on Python 3.13 + yfinance via agent proxy. Reproducible.*
