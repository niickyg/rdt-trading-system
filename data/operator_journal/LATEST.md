# LATEST — Operator State Pointer

**Most recent session:** [2026-09-09 (Instance #1, bootstrap)](entries/2026-09-09-bootstrap.md)

> New instance? Read `MANDATE.md` first (full), then this file, `../../POST_MORTEM_RRS.md`,
> `CLAUDE.md`, and the 3 newest files in `entries/`. Then follow the MANDATE Protocol.

---

## Current state in one paragraph

The operator journal/MANDATE/POST_MORTEM did not exist and were bootstrapped on
2026-09-09. First honest out-of-sample test of the bot's *own* recorded signals
(Feb–Mar 2026, 1,986 signals → 50 de-duplicated trades) against real IBKR prices,
net of 10 bps costs: **positive per-trade edge (PF 1.81, +0.28R, 64% win); equal-weight
+14.9% vs SPY −4.4%** in a down-market month. **Promising but unproven** — n=50, one
month, 72% of trades on a single entry day. The bot also selects among simultaneous
signals by arrival order (no ranking); RRS-priority selection would have added ~+2pp.

## Scoreboard (best honest edge estimate vs SPY, net of costs)

| Date | Estimate | Sample | Verdict |
|------|----------|--------|---------|
| 2026-09-09 | +0.28 R/trade; +14.9% vs SPY −4.4% | 50 trades, 1 OOS month (down mkt) | Promising, **unproven** |

## Top priorities for the next instance

1. **Widen the evidence** across more OOS windows / regimes (one month ≠ an edge).
2. **Fix outcome capture** — only 2 outcomes were ever recorded; this is the core gap.
3. **Implement RRS-priority selection at the position cap** (verify with the evaluator).
4. Investigate the edgeless short side (−0.085R) at larger n.

## Reproduce the current evidence

```bash
pip install numpy pandas          # yfinance is egress-blocked; use IBKR MCP for prices
# fetch daily bars for the 48 signal symbols + SPY into prices.json via IBKR MCP, then:
python scripts/evaluate_recorded_signals.py prices.json out.md
```

## Hard constraints (see MANDATE §2)

PAPER ONLY · never AUTO_TRADE=true · never modify live creds · never touch `risk/`
without flagging · IBKR MCP is data-only · every session ends with a committed entry.
