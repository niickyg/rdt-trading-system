# LATEST — Operator Journal Pointer

**Most recent entry:** [`entries/2026-08-26-baseline.md`](entries/2026-08-26-baseline.md)
**Date:** 2026-08-26 · **Run:** #1 (first run) · **Branch:** `claude/adoring-feynman-7wwoko`

## One-paragraph state of the bot

First operator run. The mandated constitution files did not exist and were
bootstrapped this session (`MANDATE.md`, `POST_MORTEM_RRS.md`, this journal).
Ground truth established: the connected IBKR account is empty ($5, no positions
— not the $25K paper account), there are only 2 recorded trade outcomes ever,
and **every backtest number in the repo is gross of costs** — no engine modeled
commission/slippage/spread until `backtesting/costs.py` was added today. The
best config restated net of honest costs is ~0.9%/yr, versus **SPY buy-and-hold
~16.5%/yr** over the same 2-year window. The bot does not currently beat holding
the index. No strategy/parameter change was made — deliberately, to avoid
curve-fitting to an inflated number before the ruler was fixed.

## Next run should

1. Resolve which paper account is real; reconnect the operator's live view.
2. Wire `backtesting/costs.py` into the walk-forward harnesses (restate NET).
3. Run ONE honest net-of-cost walk-forward using **IBKR** price history
   (yfinance is proxy-blocked). Clears SPY net → pursue; doesn't → start the
   wind-down conversation per MANDATE §6.
4. Investigate the signal-count discrepancy (119S/1L metrics vs 1687L/299S raw).

## Standing recommendation

Burden of proof is on the active strategy. Fix the measurement, run it honestly,
and be willing to recommend winding down to a passive index if the edge isn't
there.
