# LATEST

**Most recent entry:** [`entries/2026-09-02-genesis.md`](entries/2026-09-02-genesis.md)
**Date:** 2026-09-02
**Instance:** genesis (first run)
**Branch:** `operator/2026-09-02`

## One-line status

Genesis run: bootstrapped the operator journal + built the SPY-benchmark harness,
then established the first honest baseline — **the bot does not beat SPY
buy-and-hold, and it's not close.**

## Headline evidence (fresh, real data — 2024-09-02 → 2026-09-02)

`scripts/honest_benchmark.py` on real downloaded data, net of honest costs:

| rrs | net return | SPY buy-and-hold | edge | verdict |
|-----|-----------|------------------|------|---------|
| 1.75 | +1.33% | +41.12% | −39.79% | FAIL |
| 2.00 | +2.49% | +41.12% | −38.63% | FAIL |
| 2.50 | +3.47% | +41.12% | −37.65% | FAIL |

~2 years of returns below risk-free T-bills, vs SPY +41%. ML layer is
near-random (CV AUC 0.54, overfit); regime detector is degenerate; only 2 real
tracked trade outcomes ever. No demonstrated edge.

## What the next instance should do

1. Read `MANDATE.md` → this file → `entries/2026-09-02-genesis.md` → `POST_MORTEM_RRS.md`.
2. Re-run `LOGURU_LEVEL=WARNING python3 scripts/honest_benchmark.py --days 730` — confirm the finding holds.
3. Pick ONE: (a) extend the benchmark to walk-forward + the full intraday
   filtered strategy to test whether *anything* beats SPY, or (b) if it still
   can't, draft the wind-down recommendation (mandate §4). **Do not add
   features or leverage to a zero-edge strategy.**

## Flags for the human

- **Reproducibility bug:** `utils/secrets.py` is gitignored & absent → a fresh
  clone can't `import utils` (breaks app + entire test suite until generated).
- Data-integrity: `signal_metrics.json` direction counts disagree with
  `signal_history.json`.
- No changes were made to `risk/` or to any live-trading behavior this run.
