# RRS System — State & Evidence Log

> The scheduled operator task references this file as "the history of why the
> bot is in its current state." **No such file existed in the repository** as of
> 2026-07-28 (fresh checkout — the whole `data/operator_journal/` tree was also
> missing). Rather than invent a history I cannot verify, this file is started
> fresh as an honest, evidence-based record. Future operator runs append to it.

## Evidence snapshot — 2026-07-28 (run #1, cold start)

All figures are **ground truth** from the IBKR paper account (MCP tools) and the
repo's own signal logs, not from documentation claims.

### 1. The live paper bot is dormant

- `get_account_summary`: net liquidation **$5**, no buying power, **0 open positions**.
- `get_account_trades` (last 90 days): **0 trades**.
- `get_pa_performance_all_periods`: time-weighted return **−61.5% YTD** (inception
  2026-02-26). NAV history shows the account funded to ~$521 in late March, bled
  to ~$477 by end June, then dropped to $5 in July.
- The documented "$25K paper account DUP995654" in CLAUDE.md does **not** match
  this MCP account (single/low-hundreds dollar values). Either the operator is
  pointed at a different, near-empty account, or the documented account is
  aspirational. **This discrepancy needs the human to confirm which account is real.**

### 2. The signal pipeline stopped ~5 months ago

- `signal_metrics.json`: `last_scan_at = 2026-03-05`. No scans since. Today is 2026-07-28.
- `signal_history.json`: 1,986 raw signal rows spanning **only 6 unique days**
  (2026-02-03 → 2026-03-05), across 48 symbols, heavily duplicated intraday.
- De-duplicated to one setup per (symbol, day, direction): **88 distinct setups**,
  of which **79 are Feb 3–4** — effectively a two-day burst.

### 3. There is almost no outcome measurement

- `signal_metrics.json`: `total_outcomes = 2` (1 target hit, 1 stop out) against
  ~1,986 signals. **The system never measured whether its signals made money.**
  This is the single largest obstacle to the mission: you cannot make a strategy
  profitable if you don't measure outcomes.

### 4. The benchmark the bot must beat

- SPY (IBKR daily bars): close **689.53 on 2026-02-03** → **739.09 on 2026-07-28**
  = **+7.19%** over the period. Buy-and-hold SPY comfortably beat the bot, which
  did nothing and holds $5.
- During the actual signal window (Feb 3 → ~Feb 18), SPY was roughly flat-to-down
  (~−0.5%), i.e. the signals fired into a choppy tape — worth remembering when
  evaluating those specific setups.

### 5. The strategy's own documented ceiling loses to SPY

- CLAUDE.md's best walk-forward result (Config C, 2yr) is **~3.4% annualized**.
  SPY's long-run return is ~10% annualized. **On its own reported numbers, the
  strategy underperforms buy-and-hold.** This should be treated as a live
  hypothesis to disprove with measurement, not assumed away.

## Standing conclusion (until disproven by measured evidence)

The RDT/RRS implementation has **no demonstrated, cost-net edge over SPY
buy-and-hold**, and currently isn't even running. The highest-value work is
**instrumentation (outcome measurement)**, not more filters or ML. See
`data/operator_journal/MANDATE.md` standing priorities P0–P2.

## Tooling added to answer the question

- `scripts/signal_outcome_backtest.py` (2026-07-28): pure-stdlib harness that
  converts the signal log + daily price bars into honest forward-return outcomes
  (win rate, expectancy in R, profit factor, avg %), with a **SPY buy-and-hold
  benchmark over matched holding windows** and pessimistic intraday assumptions.
  This is the reproducible measurement loop future runs should feed real price
  data into.
