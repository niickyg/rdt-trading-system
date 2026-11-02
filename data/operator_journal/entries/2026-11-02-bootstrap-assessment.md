# 2026-11-02 — Bootstrap assessment (first operator run)

**Operator:** autonomous run, routine `rdt-bot-operator`
**Branch:** `claude/adoring-feynman-i4xh1f` (harness-designated; see note)
**Mode:** research / assess / journal. No live trading. No changes to `risk/`.

---

## 0. Why this entry exists / what I found on arrival

The scheduled task told me to read, in order: `data/operator_journal/MANDATE.md`,
`LATEST.md`, `POST_MORTEM_RRS.md`, `CLAUDE.md`, and the 3 most recent journal entries.

**None of the operator-journal files existed.** `data/operator_journal/` was absent from the
working tree and from all of git history (checked `git log --all`). `POST_MORTEM_RRS.md` did not
exist anywhere either. So this is a genuine **bootstrap / first run**: there is no prior-instance
memory to inherit.

Actions taken to bootstrap:
- Created `data/operator_journal/{MANDATE.md, LATEST.md, entries/}`.
- `MANDATE.md` transcribes the constraints + mission from the scheduled-task prompt (the only
  authoritative source available), clearly marked as reconstructed. If a human-authored mandate
  exists elsewhere, it supersedes mine.

**Branch note:** the task prompt asks for `operator/YYYY-MM-DD`, but the harness environment for
this run designated `claude/adoring-feynman-i4xh1f` and instructed never to push elsewhere without
explicit permission. I followed the harness rail and used that branch. Not merged to `main`.

---

## 1. The one question that matters

Per the mandate: **does this bot, net of honest costs, beat SPY buy-and-hold?**
I answered it with runnable evidence this session, not stale docs.

### Independent walk-forward backtest (ran `scripts/run_walkforward_v2.py` fresh)

Rolling 2-year, 6-window walk-forward, 30-stock core watchlist, $25k. Aggregate:

| Config | Gross P&L | Total | Annualized | Trades | Profit factor |
|--------|-----------|-------|------------|--------|---------------|
| A) Baseline (RRS 2.0, no filters) | $1,709 | 6.8% | **3.4%/yr** | 234 | 1.27 |
| B) Old filters | $2,101 | 8.4% | **4.2%/yr** | 430 | 1.21 |
| C) RDT filters (current prod) | **$2,472** | **9.9%** | **4.9%/yr** | 280 | 1.33 |

This *confirms and slightly exceeds* the number documented in CLAUDE.md (6.9% → now 9.9% for C on
a window that includes 2025–2026 data). Config C remains the best and lowest-drawdown variant.

### The benchmark (ran `scripts/benchmark_vs_spy.py`, new this session)

Over the **exact** walk-forward span 2024-07-08 → 2026-07-20 (2.03 years):

| | Total | Annualized | $ on $25k |
|---|------|-----------|-----------|
| **SPY buy & hold** | **+36.8%** | **+16.7%** | **$9,202** |
| Risk-free (~4.5% T-bills) | +9.4% | +4.5% | $2,338 |
| Bot config C — **frictionless** | +9.9% | +4.8% | $2,472 |

### The cost problem (confirmed, not assumed)

`grep` of the whole `backtesting/` tree: **zero** commission / slippage / borrow terms. P&L is
raw `(exit − entry) × shares` (`engine_enhanced.py:423-428`), entries at the signal-bar close,
exits at modeled stop/target prices intrabar on **daily** bars. So every headline number is a
**gross, frictionless, optimistically-filled** figure. Cost sensitivity on config C:

| Friction / round-trip | Net P&L | Annualized | Beats SPY? | Beats risk-free? |
|----|----|----|----|----|
| $3 | $1,632 | ~3.2%/yr | no | no |
| $5 | $1,072 | ~2.1%/yr | no | no |
| $8 | $232 | ~0.5%/yr | no | no |
| $12 | −$888 | ~−1.8%/yr | no | no |

---

## 2. Honest conclusion

**The bot does not beat SPY buy-and-hold, and this is not close.**

- At its *best* and with *no costs*, config C returns **4.8%/yr vs SPY's 16.7%/yr** — it captures
  under one-third of the benchmark.
- Even frictionless, it only ties the **risk-free rate** (~4.5%). Net of any realistic trading
  friction it falls **below risk-free** — i.e. a rational actor does strictly better holding
  T-bills (zero effort, zero single-stock risk) or SPY (2–3× the return).
- This is consistent with the project's own prior analysis: `ACTIONABLE_100X_STRATEGY.md` already
  admits the **Kelly fraction is ≈ −0.02 (negative)** — there is no exploitable edge to scale.
- The one genuine virtue is low volatility: max drawdown ~2.8% ($696) vs SPY's ~19% peak-to-trough
  in the Apr-2025 selloff. But a ~2% net return is a very expensive way to buy that, when T-bills
  deliver more with no drawdown at all.

The filter stack (SPY gate / SMA / VWAP / MTF / VIX / sector / intermarket) is well-engineered and
*does* improve the strategy relative to baseline (+$763 vs A). But it is polishing a vehicle whose
ceiling is far below the benchmark. **The problem is the premise (discretionary-style RRS
swing/day trading on a 30-name watchlist produces alpha), not the tuning.**

---

## 3. Recommendation — ESCALATE (do not manufacture activity)

The mandate is explicit: when the evidence says no strategy works, say so and recommend escalation
or wind-down rather than inventing optimizations. I am **escalating a strategic decision to the
human**, with three honest options:

1. **Wind down active trading of the core RRS strategy.** Park paper capital in SPY and/or T-bills
   as the honest benchmark-beating "do nothing" baseline. Keep the bot as a research harness.
2. **Accept it as a low-return / ultra-low-drawdown sleeve** *only if* a hard net-of-cost backtest
   (with the cost model below) still clears the risk-free rate — current evidence says it won't.
3. **Rethink the edge from scratch.** The current factor set has Kelly ≈ 0. Candidate directions
   that are *testable in this repo* and not yet honestly evaluated: (a) the options premium-selling
   overlay the code already scaffolds (theta, not direction), (b) longer-horizon factor tilts from
   `research/`, (c) reducing trade frequency drastically (costs scale with turnover). None should
   be pursued without a cost-aware backtest proving edge first.

I did **not** change any strategy parameters, risk limits, or execution code this session. Doing so
would be "manufacturing activity" against an edge that the evidence says is absent.

---

## 4. What I changed this session (all low-risk, reviewable)

- `data/operator_journal/` created (MANDATE.md, LATEST.md, entries/, this file).
- `scripts/benchmark_vs_spy.py` — new, read-only. One-command honest comparison of any
  frictionless backtest P&L vs SPY + risk-free + a cost-sensitivity table. So every future
  operator run starts from the honest question instead of a gross number.

Nothing in `risk/`, `brokers/`, execution, or config was touched. `AUTO_TRADE` untouched.

---

## 5. Hand-off to the next instance (READ THIS)

1. **The headline finding stands until disproven:** gross backtest ≈ 4.9%/yr, SPY ≈ 16.7%/yr,
   risk-free ≈ 4.5%/yr. Re-run `scripts/run_walkforward_v2.py` and
   `scripts/benchmark_vs_spy.py` to refresh — don't trust any doc's gross number in isolation.
2. **Do not "optimize" the RRS parameters** hoping to close a 12-point annualized gap. Kelly ≈ 0;
   tuning cannot manufacture alpha that isn't there. That path is explicitly disallowed by the
   mandate ("do not manufacture activity").
3. **Highest-value next task (if the human wants to keep going):** add a real cost model to
   `backtesting/engine_enhanced.py` (per-share commission + bps slippage + short borrow), so all
   future numbers are net by default. I scoped it but did not implement it to avoid changing
   shared backtest outputs without human review. The exact P&L lines are `engine_enhanced.py:423-428`.
4. **If the human has not responded to the escalation and the evidence is unchanged, do not keep
   running optimization passes.** Record "no change, still escalated" and stop. Repeated identical
   runs are the "keeps saying no" condition in the mandate → recommend wind-down explicitly.

**Bottom line for the human:** this paper bot is safe and well-built, but as a *money-making*
proposition it loses decisively to doing nothing (SPY) or almost-nothing (T-bills). Please pick a
direction from §3 before further engineering effort is spent.
