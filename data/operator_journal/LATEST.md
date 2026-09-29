# LATEST — Operator Journal Pointer

**Most recent session:** [2026-09-29 — Bootstrap + First Honest Assessment](entries/2026-09-29-bootstrap-assessment.md)

## One-line status
Bootstrap run: mandate/journal infrastructure was **missing** and has been
created. **No demonstrated SPY-beating edge** — bot's best backtest +6.9% vs
SPY +34.5% over the same Feb 2024–Nov 2025 window (~5x underperformance); only
2 real tracked outcomes ever; strategy's own Kelly is negative; signals are
~99% short in an up market.

## For the next stateless instance — start here
1. Read `data/operator_journal/MANDATE.md` (your constitution — reconstructed
   this session; ratify or correct it).
2. Read `POST_MORTEM_RRS.md` (evidence-based, sourced to repo artifacts).
3. Read the latest entry linked above.

## Top next action
Re-run walk-forward with **honest costs** and a strict **SPY comparison**, and
**root-cause the 119:1 short skew** in the scanner. If no simple variant beats
buy-and-hold out-of-sample, **escalate with a wind-down recommendation** rather
than optimizing further. Details in the latest entry.

## Hard constraints (never violate)
PAPER ONLY · never `AUTO_TRADE=true` / `PAPER_TRADING=false` · never touch live
credentials · flag any `risk/` change in the journal · every session ends with a
committed journal entry + updated this file · no merge to `main`.
