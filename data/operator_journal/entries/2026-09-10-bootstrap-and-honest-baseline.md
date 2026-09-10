# 2026-09-10 — Bootstrap the operator, establish the honest baseline

**Instance:** operator #1 (bootstrapping instance)
**Branch:** `operator/2026-09-10`
**Session type:** cold start — no prior journal existed

---

## 0. Orientation finding (important)

The scheduled task told me to read, first, `data/operator_journal/MANDATE.md`,
`LATEST.md`, `POST_MORTEM_RRS.md`, and the 3 most recent journal entries.
**None of these files existed** — not in the working tree, not on `main`, not anywhere
in git history. The entire operator-journal infrastructure was missing.

Interpretation: I am the **first** operator instance, and part of my job is to bootstrap
the system so future (stateless) instances aren't blind. The scheduled prompt itself
contains the mission and the hard safety constraints, so I used it as the authoritative
source for the constitution.

## 1. What I created

- `data/operator_journal/MANDATE.md` — the constitution, derived faithfully from the
  scheduled-task prompt (mission = beat SPY net of costs; PAPER ONLY; never
  `AUTO_TRADE=true`; flag any `risk/` change; journal every session; no fabricated or
  gross-as-net numbers; don't pivot to SaaS/leverage).
- `POST_MORTEM_RRS.md` — the honest history, backed by fresh measurements below.
- `data/operator_journal/entries/` — this entry.
- `data/operator_journal/LATEST.md` — points here.
- `scripts/net_cost_analysis.py` — a reusable, reviewable net-of-cost + SPY-benchmark
  tool that does **not** modify the shared backtest engine.

## 2. State assessed (all numbers reproduced this session)

Environment had no deps; installed pandas/numpy/yfinance/pyarrow/loguru/pydantic to run
the existing backtest. Ran `scripts/run_walkforward_v2.py` (2yr daily, $25k) and my new
`scripts/net_cost_analysis.py`. Window span: **2024-05-15 → 2026-05-28 (~2.03 yr)**.

**Config C (RDT filters), GROSS:** 274 trades, 48.9% WR, PF 1.34, **+9.85% total
(~4.9% ann)**. The engine models **zero** commission/slippage/spread — every headline
number in the repo is gross.

**Config C, NET of realistic costs** (IBKR-style $0.005/sh $1 min + 1–5 bps one-way):
- 1 bps → +7.2% total / **+3.5% ann**
- 3 bps → +6.4% total / **+3.1% ann**
- 5 bps → +5.5% total / **+2.7% ann**

**SPY buy & hold, identical span: +45.9% total / +20.4% ann. $25k → $36,476.**

The bot nets ~$1,400–1,800 over 2 years; SPY made **+$11,476**. It captures **~1/6 of
the index.**

### Structural findings
- **Avg entry notional = $1,977** (~8% of account/trade). Capital is chronically
  under-deployed → absolute returns are structurally tiny regardless of win rate.
- **~98% of raw signals are filtered out** by the RDT gate stack.
- The backtest is **daily-bar**; the core intraday gates (VWAP, first-hour) that define
  the live system **cannot be simulated** here. The tested strategy ≠ the deployed one.
- Repo has drifted off-mission toward a **SaaS signal service** and **3x leveraged ETFs /
  higher risk** (the "100X" docs + latest commit). Both are off-mandate; leverage on a
  near-zero/negative-Kelly edge raises ruin risk, not expected return.

## 3. Decision / hypothesis this session

Because I found no infrastructure and no trustworthy measurement, the highest-value bet
was **not** to tune parameters (which would just curve-fit gross numbers) but to
**establish an honest measurement baseline and the operator scaffolding**. Hypothesis
tested: *"Does the strategy beat SPY net of honest costs?"* → **Falsified.** It does not,
by a wide margin.

## 4. What I changed (files)

- **NEW** `data/operator_journal/MANDATE.md`, `LATEST.md`, this entry.
- **NEW** `POST_MORTEM_RRS.md`.
- **NEW** `scripts/net_cost_analysis.py` (reuses `run_walkforward_v2` machinery; no engine
  edits). Compile-checked; runs clean.
- **No changes to `risk/`.** No changes to any trading/execution code. No config changes.
  Nothing that could affect live behavior.

## 5. Honest verdict

The bot is **not profitable in the sense the mandate requires** — it trails the trivial
alternative (hold SPY) by ~17 points annualized, net of costs, during a strong bull
market. It is not broken; it is simply a poor use of capital versus the index.

## 6. Recommendation for the next instance (do these in order)

1. **Trust nothing gross.** Route every performance claim through `net_cost_analysis.py`
   (or fold its cost model into `engine_enhanced.py` — flag it clearly, it's not `risk/`).
2. **Backtest the ACTUAL intraday strategy** (5-min RRS + VWAP + first-hour) on intraday
   data. The daily approximation cannot confirm or deny the core RDT thesis. If intraday
   data/backtesting isn't feasible, treat the strategy's edge as **unfalsifiable** and say
   so.
3. **Investigate the $1,977 avg-notional / under-deployment problem** — likely the single
   biggest lever on absolute return, if a real edge exists.
4. **Do not add leverage or a subscription business** to paper over the gap. If honest
   intraday testing still trails SPY, **recommend wind-down**: hold the index. Under this
   mandate that is a valid, successful outcome — not a failure to avoid.

## 7. Safety attestation

PAPER only. `AUTO_TRADE` untouched. No broker credentials touched. `risk/` untouched.
No live orders. Work pushed to `operator/2026-09-10` for human review; not merged to main.
