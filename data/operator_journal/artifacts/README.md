# Operator artifacts

Data files backing journal entries, kept for reproducibility.

## 2026-08-06_ibkr_daily_prices.json
Daily OHLC bars (`{SYMBOL: [{date,o,h,l,c}, ...]}`) for the 48 signal symbols + SPY, spanning
2026-02-02 → 2026-08-05, fetched from the IBKR MCP `get_price_history` tool (SIX_MONTHS / ONE_DAY,
RTH only) on 2026-08-06.

**Provenance caveat:** the bars were fetched and normalized by an LLM subagent, so treat them as
*estimates*, not authoritative tick data. Internal consistency check: 6,037 bars, 0 nonpositive,
21 bars (0.35%) with a close sitting ≤0.02 outside the session high/low (IBKR "Last"-sourced
daily bar artifact). Aggregate statistics computed from this file are robust to that noise;
per-cent-precise claims are not. Regenerate from IBKR before relying on exact values.

Consumed by: `scripts/evaluate_signal_edge.py`, `scripts/edge_control_test.py`,
`scripts/edge_realistic_fill.py`.
