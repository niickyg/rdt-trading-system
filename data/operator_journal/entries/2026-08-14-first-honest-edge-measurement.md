# 2026-08-14 — First honest edge measurement + SPY gate fail-closed

**Operator instance:** bootstrap (session 1). **Branch:** `operator/2026-08-14`.
**Model:** claude-opus-4-8[1m].

## TL;DR

- This was the **first operator run**. The journal infrastructure referenced by the
  scheduled prompt (`MANDATE.md`, `LATEST.md`, `POST_MORTEM_RRS.md`, `entries/`) **did
  not exist** — I bootstrapped all of it.
- The bot has **no P&L ledger**: `signal_history.json` is a raw pre-gate signal log with
  zero outcome fields; `signal_metrics.json` had only 2 tracked outcomes ever. So there
  was no honest evidence about profitability at all. I created some.
- I ran the **first independent backtest** of the 1,986 historical signals against **real
  IBKR daily prices** (yfinance is blocked in this environment; the repo's own backtests
  can't run here). Result on 1,838 signals: **longs win (69% hit, +1.03R net), shorts lose
  badly (2.3% hit, −0.74R net)**, in a rising market.
- The short-side loss is the robust, actionable finding. It **empirically validates the
  SPY "Market First" gate** and exposed that the gate **fails OPEN**. I made the gate
  **fail closed** (block everything when SPY trend is unknown; short only in a
  confirmed-bearish tape). Scanner change, verified, does not touch `risk/`.

## What I found (evidence)

### 1. No track record exists
`data/signals/signal_history.json`: 1,986 signals (Feb 3 – Mar 5 2026), **no exit/pnl
fields**. `signal_metrics.json`: 880 scans, 120 signals, **2 outcomes** (1 win/1 loss).
One metrics snapshot showed a direction flip to **119 shorts / 1 long** — i.e., in some
regimes the live scanner emits mostly shorts, the losing side.

### 2. Independent first-touch backtest (real IBKR data)
Tooling committed at `data/operator_journal/tools/` (script + saved output + conid map +
README with reproduction steps). 45/48 symbols resolved to correct contracts (DD dropped
for a wrong-contract price mismatch, WEC/XEL not fetched). MAX_HOLD=10d, costs ~$0.05/sh RT:

| Slice | n | Win rate | Net expectancy | PF |
|-------|---|----------|----------------|----|
| All | 1,838 | 60.3% | +0.76 R | 3.05 |
| Longs | 1,562 | 69.0% | **+1.03 R** | 4.49 |
| Shorts | 276 | 2.3% | **−0.74 R** | 0.075 |

Directional alpha vs SPY (10d, no stop/target): **longs +1.40% alpha** (64% beat SPY);
**shorts −2.35% alpha** (shorted names rose while SPY was flat). Long edge is broad
(median 87% win rate across 28 symbols), not a few-name artifact.

### 3. Honest caveats (why I did NOT declare victory)
- One month of signals, **one rising-market regime**; forward window Feb–Mar 2026. Not
  walk-forward, not multi-regime.
- Long "alpha" adjusts for SPY *direction* but **not beta magnitude** — cannot yet be
  separated from leveraged long-beta momentum.
- These are **raw pre-gate** signals → measures signal edge, not deployed performance.
- Therefore: **no claim that the system beats SPY buy-and-hold risk-adjusted.** That
  needs a bear-market sample and beta adjustment, which we don't have.

## What I changed and why

**File:** `scanner/realtime_scanner.py`, `_apply_spy_gate()` (NOT in `risk/`).

The SPY "Market First" gate blocks counter-trend signals — its design is now empirically
justified (shorts off-trend lose money). But it **failed open**: on missing SPY trend
data (`None`) it let *all* signals through, and in a "mixed" tape it let shorts through
with only a warning. Since `get_spy_daily_trend()` depends on flaky/blocked yfinance,
this is a live risk. Changes:
- **Unknown trend → block everything** ("no market context ⇒ no trades").
- **Mixed trend → block shorts, keep longs** ("short only a confirmed-weak market").
- Bullish (block shorts) / bearish (block longs) unchanged.

**Verified:** compiles; isolated behavioral test of the actual method source passes for
all four regimes (bearish 0/2, bullish 2/0, mixed 2/0, unknown 0/0). This only *tightens*
filtering (never emits a signal it didn't before), so downside risk is low; the cost is
fewer trades when SPY data is missing or choppy — acceptable given shorts' negative edge.

## What I did NOT do (and why)
- Did **not** touch `risk/`.
- Did **not** enable live trading or change any broker/credential config.
- Did **not** add ML/features/overlays — the mandate warns against "fix a weak edge by
  adding complexity." I removed an edge-negative failure mode instead.
- Did **not** block longs on missing data — longs showed positive expectancy; I kept the
  change minimal and asymmetric toward the proven-losing side.

## Open questions / recommendations for the next instance

1. **Beta-adjust the long edge.** Is the +1.4% alpha real skill or high-beta momentum in
   an up-tape? Regress each signal's forward return on its stock beta × SPY return.
2. **Get a bear/sideways sample.** The whole conclusion hinges on one bull month. Pull IBKR
   history for a 2025 drawdown window, synthesize RRS signals over it, and re-run. If longs
   only work when SPY rises, the "edge" is just beta and the honest answer is "buy SPY."
3. **Validate the short gate helps, don't just assume.** Once a bear sample exists, confirm
   shorts actually work in confirmed-bearish tapes (my data had too few bearish-tape shorts).
4. **Wire up outcome tracking.** The single biggest missing piece is a real P&L ledger. Until
   `outcome_tracker`/`execution_tracker` persist realized results, every profitability claim
   is theoretical. Prioritize making the paper broker log fills+exits to the DB.
5. **Re-resolve DD's contract id** before reusing `ibkr_conids.json` (known bad).

## Reproduce
`data/operator_journal/tools/README.md` has full steps. Price files are transient
(scratchpad, not committed) — regenerate via the IBKR MCP tools per the README.
