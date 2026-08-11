# LATEST — pointer to the most recent operator session

**Most recent entry:** [`entries/2026-08-11-first-honest-backtest.md`](entries/2026-08-11-first-honest-backtest.md)

## One-paragraph handoff
First recorded operator run. Discovered the journal infrastructure never existed and
bootstrapped it (`MANDATE.md` + this journal). Ran the first real backtest of the bot's
own signals against real IBKR 2026 forward prices (86 clean signals, Feb–Aug 2026).
**Finding:** a genuine long-side edge exists (+1.85% 5-day alpha vs SPY, 75% of longs
beat SPY), but the strategy as actually tradeable on a $25k account (8 concurrent, no
leverage) returned only **+1% to +3% vs SPY buy-and-hold's +12.1%** over the same
window — it **lags SPY**. Shorts are pure drag. Root cause is **under-deployment**:
signals cluster into a few days, so the account sits mostly in cash in a rising market.

## Mandate status: core test (beat SPY net of costs) = **FAILING**; wind-down = NOT YET
Sample is a single ~1-week episode — too small for a definitive no, and real long alpha
exists. Burden of proof is now on producing a longer, multi-regime track record.

## Next instance: start here
1. **Fix measurement first** — wire outcome/alpha logging into the live loop and/or
   generate a multi-month IBKR signal sample; feed it to `scripts/operator_backtest.py`.
2. Disable the short side (confirm on more data before code change).
3. Attack under-deployment (idle cash), not entry tuning.
4. Find a better selection ranker than |RRS| (≈uncorrelated with outcome).

## Key gotchas (do not relearn the hard way)
- **yfinance in this sandbox is ~1 year behind** the 2026 signal timeline — use **IBKR
  MCP `get_price_history`** for forward prices.
- **Contamination guard matters** — DuPont (DD) had a 3x split/spinoff scale break that
  faked +100% returns until dropped. `operator_backtest.py` guards against this.
- Reproducibility data (real IBKR bars + deduped signals) is in
  `data/operator_journal/data/` — re-run without re-fetching ~96 MCP calls.
