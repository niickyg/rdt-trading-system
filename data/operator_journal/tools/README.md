# Operator tools

## backtest_firsttouch.py

Independent first-touch backtest of `data/signals/signal_history.json` against **real
IBKR daily prices** (yfinance is blocked in the remote agent environment). Measures raw
signal edge: enters at each signal's `entry_price`, walks subsequent daily bars, records
first touch of stop vs target (daily high/low), same-bar ambiguity → stop, timeout →
mark-to-market. Reports win rate vs breakeven, expectancy in R, profit factor, a
directional-alpha test vs SPY, long/short splits, and a SPY buy-and-hold benchmark.

### How to reproduce (the data-gathering step)

The script reads per-symbol price files from a `prices/` directory (one JSON per symbol,
the raw `get_price_history` response with `time/open/high/low/close/volume` arrays, plus
`SPY.json`). To regenerate them:

1. Resolve each watchlist symbol to an IBKR `contract_id` with the
   `mcp__Interactive-Brokers--IBKR-__search_contracts` tool (pick the US primary STK
   listing: `country_code == "US"`, exchange NYSE/NASDAQ/ARCA). A resolved map from the
   2026-08-14 run is in `ibkr_conids.json`.
   ⚠️ **`ibkr_conids.json` has a KNOWN-BAD entry: `DD=887235923`** resolves to the wrong
   instrument (price ~$91–158 vs DuPont's ~$45–48). Re-resolve DD before using it.
2. For each conid call `mcp__Interactive-Brokers--IBKR-__get_price_history` with
   `security_type="STK", step="ONE_DAY", period="ONE_YEAR", outside_rth=false` and save
   the raw response to `prices/<SYMBOL>.json`. Delegate this to subagents — IBKR calls are
   slow (~30–45s each); ~48 symbols took ~30 min across parallel agents.
3. Edit the `PRICES` constant at the top of `backtest_firsttouch.py` to point at your
   `prices/` dir, then `python3 backtest_firsttouch.py`.

`backtest_2026-08-14_output.txt` is the saved output from the 2026-08-14 run (45 symbols).

### Caveats (read before trusting the numbers)
- One month of signals, one (rising-market) forward regime. Not walk-forward.
- Raw pre-gate signals — measures signal edge, not deployed/post-gate performance.
- Long "alpha" is SPY-direction-adjusted but NOT beta-adjusted.
