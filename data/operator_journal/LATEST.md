# LATEST — pointer to the most recent operator session

**Most recent entry:** [`entries/2026-10-09-genesis.md`](entries/2026-10-09-genesis.md)

## One-line status
GENESIS run (2026-10-09): bootstrapped the operator journal and ran three honest
measurements. **Verdict: the bot does NOT beat SPY buy-and-hold** — best config
~6%/yr vs SPY ~18%/yr over the same 2 years. No real edge demonstrated; no live
outcome tracking exists.

## Carry-forward for the next instance
1. The bot's own walk-forward puts its best config at ~6%/yr (PF 1.44); SPY
   returned ~18%/yr over the same window. The gap is ~3x, against the bot.
2. `SignalMetrics.record_outcome()` has ZERO callers — production never measured
   its 1,986 signals (only 2 outcomes ever). Wiring this is the prerequisite to
   any checkable performance claim.
3. Beware the signal-backfill look-ahead trap: filling at logged `entry_price`
   fakes a +505% month. `scripts/analyze_signal_outcomes.py` now defaults to
   realistic next-open entry — keep it that way.
4. Do not add more filters/models until an out-of-sample edge over SPY is shown.
   If the next 1–2 honest tests also fail, recommend wind-down to the human.

See `MANDATE.md` for the full constitution and protocol.
