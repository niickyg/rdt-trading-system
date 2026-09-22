# LATEST

Pointer to the most recent operator session. Read this, then the entry it names.

- **Latest entry:** [`entries/2026-09-22-genesis.md`](entries/2026-09-22-genesis.md)
- **Date:** 2026-09-22
- **Session type:** Genesis (first operator run — bootstrapped the journal)

## One-line state

RRS strategy does **not** clear the mandate bar: fresh out-of-sample
walk-forward = ~3.7% (gross, prod config) vs SPY buy-and-hold ~23.6% over the
same window; the RDT filter stack actively hurts; backtester models **zero**
transaction costs.

## Open items for the next operator

1. **Add commission + slippage to `backtesting/engine.py`** and re-baseline —
   no number is trustworthy until costs are modeled. (Highest leverage.)
2. Re-run the 3-way comparison **net of costs** over a 2-year multi-regime
   window (must include a non-bull period).
3. Decide with the human: fix the edge, or wind down to index buy-and-hold.
   Do not ship a paid signal service on a negative-Kelly strategy.
4. `MANDATE.md` is self-authored — get it **ratified** by the human owner.

## Standing constraints (from MANDATE.md)

PAPER ONLY · never `AUTO_TRADE=true` · never touch `risk/` unflagged · never
merge to `main` · every session commits a journal entry.
