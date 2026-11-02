# LATEST — operator hand-off

**Most recent entry:** [`entries/2026-11-02-bootstrap-assessment.md`](entries/2026-11-02-bootstrap-assessment.md)
**Date:** 2026-11-02
**Run type:** Bootstrap (first operator run — no prior journal existed)

## One-line status

🔴 **Bot does NOT beat SPY and loses to risk-free T-bills net of costs. Strategic decision escalated to human — do not manufacture optimizations.**

## Numbers that matter (walk-forward span 2024-07-08 → 2026-07-20, 2.03y, $25k)

| | Annualized | $ on $25k |
|---|---|---|
| SPY buy & hold | **+16.7%** | $9,202 |
| Risk-free (~4.5%) | +4.5% | $2,338 |
| Bot config C (best, **frictionless**) | +4.8% | $2,472 |
| Bot config C net of ~$5/round-trip | ~+2.1% | $1,072 |

Backtest models **zero** trading costs. Kelly fraction of the strategy ≈ −0.02 (negative edge).

## For the next instance

1. Read `MANDATE.md` then the bootstrap entry above, fully.
2. Re-run `scripts/run_walkforward_v2.py` + `scripts/benchmark_vs_spy.py` to refresh evidence.
3. **Do NOT tune RRS params to chase the gap** — the mandate forbids manufacturing activity against an absent edge.
4. If the human hasn't picked a direction (§3 of the entry) and evidence is unchanged: record "no change, still escalated," recommend wind-down, and stop.

## Awaiting human decision (see §3 of entry)

- (1) Wind down active RRS trading → park in SPY/T-bills, keep bot as research harness; **or**
- (2) keep only if a cost-aware backtest clears risk-free (unlikely on current evidence); **or**
- (3) rethink the edge (options theta overlay / lower turnover / `research/` factors) — cost-aware proof required first.
