# Forward-test of shipped RRS signals (operator, 2026-07-30)

Reproducible evaluation of the **1,986 real signals** the bot emitted
(`data/signals/signal_history.json`, Feb 3 – Mar 5 2026) against **real IBKR
daily bars**. Answers: did these signals make money net of costs, and did they
beat SPY buy-and-hold?

## Contents
- `setups.json` — the 88 distinct daily setups (symbol, date, direction,
  entry/stop/target) extracted from the signal history (dedup: first occurrence
  per symbol/date/direction).
- `prices/<TICKER>.json` — 6 months of daily OHLC bars per symbol + `SPY.json`,
  fetched via the IBKR MCP `get_price_history` tool (yfinance/Yahoo is blocked by
  the egress proxy). Keys: `time, open, high, low, close`.
- `first_touch.py` — first-touch (target vs stop) R-multiple simulation.
- `benchmark.py` — per-trade realized return vs SPY over matched holding windows
  (alpha).

## Run
```bash
cd research/forward_test
python3 first_touch.py 10   # horizon in trading days (also try 5, 20)
python3 benchmark.py 10
```
No third-party deps (stdlib only).

## Method & honest caveats
- Fill assumed at the signal's `entry_price` on the first daily bar on/after the
  signal date; scan forward up to HORIZON trading days for first touch of the
  signal's own stop/target. Conservative tie-break: a single daily bar spanning
  both counts as a stop. (In this dataset 0 bars were ambiguous.)
- If neither touched by the horizon, exit at the last bar's close (mark-to-market).
- Costs: `first_touch.py` charges 0.05R/trade; `benchmark.py` charges 0.10% of
  notional/trade. Both are optimistic-but-reasonable for liquid names.
- **Daily bars cannot resolve intra-bar ordering** — hence the conservative
  tie-break.
- **71% of the underlying signals were generated outside RTH**, so `entry_price`
  may be an unrealistic after-hours print; real fills would differ.
- **The sample is tiny and concentrated: 79 of 88 setups occur on Feb 3–4 2026.**
  Effective independent observations ≈ 2 days, not 88. Treat every aggregate as a
  hint, not proof.

See `POST_MORTEM_RRS.md` and `data/operator_journal/entries/2026-07-30-*.md` for
results and interpretation.
