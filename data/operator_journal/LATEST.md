# LATEST — handoff to the next operator instance

> Read `MANDATE.md` first, then this file, then `POST_MORTEM_RRS.md`, then
> `CLAUDE.md`, then the 3 most recent `entries/`.

**Last session:** 2026-10-13 — **Bootstrap + Cost Reality Check**
**Entry:** `entries/2026-10-13-bootstrap-and-cost-reality-check.md`
**Branch:** `claude/adoring-feynman-wbn53u` (harness-designated; the prompt's
`operator/2026-10-13` was overridden for push safety — see entry).

## State of the mission (one-paragraph truth)

The RRS day-trading premise has **no demonstrated edge net of honest costs** and
**badly underperforms SPY buy-and-hold.** Independently verified this session:
the backtest engines model **zero** transaction costs; the documented "best" config
(Config C) returns **10.31% gross** over Feb-2025→Oct-2026 vs **SPY +31.31%** over the
same window, and net of a realistic 10 bps/side it returns **4.32% ($1,081)** — ~14% of
buy-and-hold — on 243 trades. Break-even is ~20.7 bps/side. The repo's own docs already
showed **Kelly ≈ −0.02 (negative)** and then pivoted to *selling signals* instead of
trading — a trap the operator must not adopt as the mission.

## What was built this session

- `data/operator_journal/MANDATE.md` — the constitution (was missing).
- `POST_MORTEM_RRS.md` — the history + the evidence (was missing).
- `scripts/run_cost_sensitivity.py` — cost-aware backtest vs SPY buy-and-hold (the
  measurement tool the repo lacked). Run: `python3 scripts/run_cost_sensitivity.py`.

## Do next (pick ONE, cost-aware + out-of-sample from run #1)

1. Falsification test on a **larger universe (full S&P 500) across multiple regimes,
   including a down market.** If it still can't beat SPY net of ≥10 bps/side OOS →
   that is wind-down evidence.
2. Split the existing trade list by **direction** (longs vs shorts) and by **filtered
   vs baseline** *after costs* — cheap extensions to `run_cost_sensitivity.py` that
   test where (if anywhere) the edge lives.
3. If 2–3 honest tests keep saying "no edge" → **write the wind-down/escalation
   recommendation to the human and stop.** Reaching that honestly is success.

## Do NOT

- Do NOT add features, tune parameters, or chase the "100×"/signal-service narrative
  to manufacture returns.
- Do NOT enable live trading (`AUTO_TRADE`/`PAPER_TRADING=false`) or touch broker
  credentials. Flag any `risk/` change prominently.
- Do NOT trust any backtest number in this repo that isn't net of honest costs and
  compared to SPY buy-and-hold.

## Environment setup for the next instance

Default `python3` is 3.11 but deps installed into **3.13**; run analysis with
`/usr/bin/python3.13`. Install: `python3 -m pip install numpy pandas yfinance loguru
pydantic pydantic-settings pyarrow` (pyarrow needed for the data-loader parquet cache).
