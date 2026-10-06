# Operator Journal — 2026-10-06 (Bootstrap Run)

**Instance:** First autonomous operator run. No prior state existed.
**Branch:** `claude/adoring-feynman-vduw2o` (see "Branch discrepancy" below).

---

## 0. Situation on arrival

The task prompt instructed me to read `data/operator_journal/MANDATE.md`,
`LATEST.md`, `POST_MORTEM_RRS.md`, and recent entries **before doing anything
else**. None of these files existed — the entire operator infrastructure was
absent from the repo. This is therefore a bootstrap run: I had to (a) create the
mandate/journal/post-mortem that future runs depend on, and (b) perform the first
honest assessment they refer to.

## 1. Hypothesis under test

> "Is there evidence the RDT/RRS strategy, in any configured variant, beats SPY
> buy-and-hold net of costs?" — the foundational question of the mandate.

## 2. What I did

1. Installed backtest deps (pandas, numpy, yfinance, scipy, pyarrow, pydantic).
   yfinance works through the agent proxy.
2. Ran the project's own 2-year, 3-way walk-forward backtest **unchanged**:
   `python scripts/run_walkforward_v2.py`.
3. Wrote and committed `scripts/benchmark_vs_spy.py` to compute SPY buy-and-hold
   over the **exact same active span** (trading-day index 91..600 of the loaded
   daily series = 2024-06-10 → 2026-06-23). This comparison did not exist anywhere
   in the codebase — the backtest only ever used SPY for *gating*, never as a
   benchmark.
4. Verified the backtest engine (`backtesting/engine_enhanced.py`) models **no
   commission and no slippage** — so strategy returns are *gross*.
5. Created the operator infrastructure: `MANDATE.md`, `POST_MORTEM_RRS.md`,
   this entry, and `LATEST.md`.
6. Corrected the aspirational backtest table in `CLAUDE.md` to show the SPY
   benchmark alongside strategy returns.

## 3. Results (verified, reproducible)

Span: 2024-06-10 → 2026-06-23 (~2.03 yrs), $25,000, 30-stock watchlist.

| | Total Return | Annualized | $ Profit | Trades | Win Rate | PF |
|---|---|---|---|---|---|---|
| **SPY buy-and-hold** | **+40.64%** | **+18.25%** | **+$10,161** | 1 | — | — |
| C) RDT Filters (best, gross) | +9.53% | +4.7% | +$2,383 | 277 | 49.5% | 1.33 |
| B) Old Filters | +8.01% | +4.0% | +$2,003 | 424 | 47.4% | 1.20 |
| A) Baseline | +6.07% | +3.0% | +$1,517 | 235 | 47.7% | 1.23 |

- Best variant underperforms SPY by **~13.5 pp/year**.
- Filters discard **98.1%** of signals (14,671 → 277) to add ~3.5pp *total* over
  two years vs baseline — negligible against the gap.
- Returns are **gross**; net of costs on 277 trades the gap widens.

## 4. Verdict

**NEGATIVE / FOUNDATIONAL.** On the strongest available evidence, the strategy
does not come close to beating SPY buy-and-hold — and it failed to do so *during
a bull market*, the most favorable environment for a long-biased momentum system.
The large ML/options/agent surface area does not contribute measurable P&L
(ML is already documented as advisory-only). This is the central fact every
future instance must reason from. Full detail + caveats in `POST_MORTEM_RRS.md`.

This is not a bug to fix by tuning. It is a strategy-level result.

## 5. Repo changes this session

- `+ data/operator_journal/MANDATE.md` — the constitution.
- `+ POST_MORTEM_RRS.md` — the foundational finding + caveats + backlog.
- `+ scripts/benchmark_vs_spy.py` — reproducible SPY benchmark (was missing).
- `+ data/operator_journal/entries/2026-10-06-bootstrap.md` — this entry.
- `+ data/operator_journal/LATEST.md` — pointer for next instance.
- `~ CLAUDE.md` — added SPY benchmark to the walk-forward results section.
- `~ .gitignore` — ignore `data/backtest_cache/` (regenerable, large).
- **`risk/` directory: NOT touched.**

## 6. Backlog for the next instance (prioritized)

1. **[High value / low effort] Add a real cost model** (commission + per-side
   slippage) to `engine_enhanced.py` so all future numbers are net, not gross.
   This will likely push every variant's return *down* and make the verdict
   starker — but honesty demands it.
2. **[Diagnostic] Split the backtest by direction.** If longs carry all the
   return and shorts are a net drag, the "relative weakness → short" half of the
   thesis is unsupported and should be cut.
3. **[Diagnostic] Compute beta-adjusted alpha.** Determine whether the strategy's
   small return is just low-beta market exposure (in which case SPY+cash also
   beats it).
4. **[Test] Highest-conviction subset.** Does trading only the top RRS decile
   ever beat SPY, or merely trade less?
5. **[Framing] Bear/flat-market window (e.g. 2022).** Test whether the honest
   case for this system is "downside protection," and quantify it. If even that
   fails, escalate toward wind-down.
6. **Resist feature sprawl.** Do not add new modules. The gap is strategic, not
   a missing feature.

## 7. Branch discrepancy (flag for the human)

The task prompt's branch strategy says to use `operator/YYYY-MM-DD`. The harness
environment, however, designates `claude/adoring-feynman-vduw2o` and states
"NEVER push to a different branch without explicit permission." I honored the
harness (safer, and it is the branch set up for your review) and did all work on
`claude/adoring-feynman-vduw2o`. If you want future operator runs on dated
`operator/*` branches, please reconcile the harness branch designation with the
cron prompt.

## 8. Safety attestation

PAPER context only. No broker orders placed. `AUTO_TRADE`/`PAPER_TRADING` not
changed. No live credentials touched. `risk/` not modified.
