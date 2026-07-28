# Backtest harness fixtures

Tiny, hand-verified example inputs for `scripts/signal_outcome_backtest.py`.
Outcomes are deterministic and known, so this doubles as a smoke test.

Run:

```bash
python3 scripts/signal_outcome_backtest.py \
  --signals data/operator_journal/fixtures/example_signals.json \
  --prices  data/operator_journal/fixtures/example_prices.json \
  --max-hold 5 --cost-r 0.05
```

Expected: 4 filled / 5 setups, 50% win rate, +0.45R expectancy, edge +0.50% vs SPY.
(EEE never fills; DDD's daily bar contains both stop and target -> counted as a
stop, i.e. the pessimistic intraday assumption.)

## Generating a REAL prices file

The harness needs `{SYMBOL: [{date,open,high,low,close}, ...], ...}` including SPY.
Two ways to produce it:

- **Local bot container** (has yfinance): download daily OHLC for every symbol in
  the signal window plus SPY and dump to this JSON shape.
- **Remote operator agent**: use the IBKR MCP `get_price_history` tool
  (security_type=STK, step=ONE_DAY, period=SIX_MONTHS) per symbol and reshape the
  parallel time/open/high/low/close arrays into the per-bar dict format.
