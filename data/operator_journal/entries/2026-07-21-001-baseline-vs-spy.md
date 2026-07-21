# Run 001 — 2026-07-21 — Baseline established: strategy loses badly to SPY

**Operator:** first instance. **Branch:** `operator/2026-07-21`.
**Status of journal on arrival:** empty. No MANDATE, no POST_MORTEM, no entries.
This session bootstrapped the journal infrastructure and established the
first honest performance baseline.

## What I did

1. Found the mandated read-list files (`MANDATE.md`, `LATEST.md`,
   `POST_MORTEM_RRS.md`, journal entries) **did not exist**. Bootstrapped them.
2. Confirmed the environment can fetch real market data via a `requests`-based
   Yahoo shim (yfinance's curl_cffi fails TLS through the agent proxy).
3. Ran the project's **own** walk-forward backtest (`run_walkforward_v2.py`)
   **unmodified** on fresh 2024–2026 data, via the shim. Then computed the
   thing the project never had: **SPY buy-and-hold over the identical window.**

## The result (reproducible)

Window 2024-03-27 → 2026-04-09 (~2.03 years, $25k, 510 trading days):

| Config | Total return | Annualized | Trades | Win rate |
|--------|-------------|-----------|--------|----------|
| A) Baseline (no filters) | +4.36% | 2.2% | 233 | 48.5% |
| B) Old filters | +4.47% | 2.2% | 417 | 48.9% |
| C) RDT filters (best) | +5.39% | 2.7% | 266 | 50.0% |
| **SPY buy-and-hold** | **+33.5%** | **15.2%** | 1 | — |

**The best strategy config captured ~1/6 of buy-and-hold, and the +5.39% is
GROSS** — `backtesting/engine*.py` model zero commissions and zero slippage
(grep-confirmed). With 266 trades, realistic friction plausibly erases most of
the gross gain. Net of honest costs, the best config is likely break-even to
negative while SPY returned +33.5%.

This directly contradicts the framing in CLAUDE.md, which cites the RDT filters
as a 2-year "win" (+6.9%). That comparison was strategy-vs-strategy; it never
included the benchmark. **Corrected: no tested config beats buy-and-hold.**

## Other findings

- **No live track record.** `signal_metrics.json` (last scan 2026-03-05): 880
  scans, 120 signals, **only 2 tracked outcomes.** The bot has never
  demonstrably made or lost measurable money.
- **119 short : 1 long** emitted signals — the SPY hard gate blocks *all* longs
  when SPY is below its 50/200 EMA (bearish Feb–Mar 2026). Working as coded, but
  it discards RDT's core "buy relative strength" edge in weak tapes.
- Data note: window is a strong SPY bull market (+33%). The strategy's only
  possible defense — "protects capital in downturns" — is untested and must be
  measured, not assumed.

## Caveats (stated honestly)

- The walk-forward is a **daily-bar proxy** of an **intraday** strategy; it
  cannot simulate VWAP or first-hour filters. An intraday edge *could* exist
  that daily bars miss — but it is asserted nowhere-demonstrated. This daily
  backtest is the project's own primary evidence, and it fails the benchmark.

## Constraints touched

- Did **not** touch `risk/`. Did **not** change any trading/execution code.
- Paper-only respected; no broker credentials touched; AUTO_TRADE untouched.
- New code is confined to `scripts/operator/` (reproducibility harness) and the
  operator journal. No behavior change to the live system.

## Recommendation to next operator

The burden of proof is inverted. **Do not add another filter.** Instead:
1. Build a **cost-aware** backtest and quantify how much of +5.39% survives.
2. Construct an **intraday** backtest (Yahoo ~60d of 5m bars) to test the
   *actual* live strategy for any edge the daily proxy can't see.
3. Measure performance specifically in **flat/bear** regimes.
4. If these keep failing to beat buy-and-hold: draft the **wind-down /
   escalation** recommendation. The user is currently better served holding SPY.

## Reproduce this

```
pip install yfinance pandas numpy pyarrow loguru pydantic pydantic-settings
python scripts/operator/run_walkforward.py
```
Full raw output archived at `data/operator_journal/entries/2026-07-21-001-wf-output.txt`.
